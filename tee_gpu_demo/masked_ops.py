"""Masked matrix multiplication primitives.

This file intentionally simulates the TEE/GPU split in one PyTorch process:

- trusted side: creates masks and removes correction terms;
- untrusted side: only receives masked tensors and performs the heavy matmul.

It is a prototype for validating the algebra. It is not a real TEE boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def _randn(
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: Optional[torch.Generator],
    scale: float,
) -> Tensor:
    return torch.randn(shape, dtype=dtype, device=device, generator=generator) * scale


def _matmul_t(lhs: Tensor, rhs: Tensor) -> Tensor:
    """Compute lhs @ rhs.T over the last two dimensions."""
    return torch.matmul(lhs, rhs.transpose(-1, -2))


def _device_or_default(device: Optional[torch.device | str], fallback: torch.device | str) -> torch.device:
    return torch.device(device) if device is not None else torch.device(fallback)


def _default_untrusted_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default_trusted_dtype(device: torch.device, dtype: torch.dtype) -> torch.dtype:
    # CPU simulates the TEE, so keep trusted math in fp32 for numerical stability.
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def _pin_if_cuda_copy(tensor: Tensor, target_device: torch.device) -> Tensor:
    if tensor.device.type != "cpu" or target_device.type != "cuda":
        return tensor
    try:
        return tensor if tensor.is_pinned() else tensor.pin_memory()
    except RuntimeError:
        return tensor


def _to_device_async(tensor: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if tensor.device == device and tensor.dtype == dtype:
        return tensor
    source = _pin_if_cuda_copy(tensor, device)
    return source.to(device=device, dtype=dtype, non_blocking=True)


def _copy_to_trusted_async(
    tensor: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Optional[torch.cuda.Stream]]:
    if tensor.device == device and tensor.dtype == dtype:
        return tensor, None
    if tensor.device.type == "cuda" and device.type == "cpu":
        try:
            output = torch.empty(tensor.shape, dtype=dtype, device=device, pin_memory=True)
            stream = torch.cuda.Stream(device=tensor.device)
            stream.wait_stream(torch.cuda.current_stream(tensor.device))
            with torch.cuda.stream(stream):
                output.copy_(tensor, non_blocking=True)
            return output, stream
        except RuntimeError:
            pass
    return tensor.to(device=device, dtype=dtype, non_blocking=True), None


def _wait_for_copy(stream: Optional[torch.cuda.Stream]) -> None:
    if stream is not None:
        stream.synchronize()


@dataclass
class MaskedLinearResult:
    output: Tensor
    masked_input: Tensor
    masked_output: Tensor
    correction: Tensor


@dataclass
class LowRankMask:
    """Low-rank mask R = left @ right for tensors shaped [..., rows, dim]."""

    left: Tensor
    right: Tensor

    @classmethod
    def random(
        cls,
        target: Tensor,
        rank: int,
        *,
        generator: Optional[torch.Generator] = None,
        scale: float = 0.02,
    ) -> "LowRankMask":
        if target.ndim < 2:
            raise ValueError("low-rank masks require tensors shaped [..., rows, dim]")
        if rank <= 0:
            raise ValueError("rank must be positive")

        *batch, rows, dim = target.shape
        # Split the scale so the materialized mask has roughly scale-sized values.
        factor_scale = math.sqrt(scale / max(math.sqrt(rank), 1.0))
        left = _randn(
            (*batch, rows, rank),
            dtype=target.dtype,
            device=target.device,
            generator=generator,
            scale=factor_scale,
        )
        right = _randn(
            (*batch, rank, dim),
            dtype=target.dtype,
            device=target.device,
            generator=generator,
            scale=factor_scale,
        )
        return cls(left=left, right=right)

    def materialize(self) -> Tensor:
        return torch.matmul(self.left, self.right)


@dataclass
class MaskedQKResult:
    output: Tensor
    masked_q: Tensor
    masked_k: Tensor
    masked_output: Tensor
    correction: Tensor
    q_mask: LowRankMask
    k_mask: LowRankMask


@dataclass
class MaskedPVResult:
    output: Tensor
    masked_p: Tensor
    masked_v: Tensor
    masked_output: Tensor
    correction: Tensor
    p_mask: LowRankMask
    v_mask: Optional[LowRankMask]


class UntrustedGPU:
    """The part that only sees masked operands."""

    @staticmethod
    def linear(masked_x: Tensor, weight: Tensor) -> Tensor:
        return F.linear(masked_x, weight, bias=None)

    @staticmethod
    def qk(masked_q: Tensor, masked_k: Tensor) -> Tensor:
        return _matmul_t(masked_q, masked_k)

    @staticmethod
    def pv(masked_p: Tensor, masked_v: Tensor) -> Tensor:
        return torch.matmul(masked_p, masked_v)


class _TokenBuffer:
    """Append-only contiguous buffer for token-major tensors shaped [tokens, dim]."""

    def __init__(self, dim: int, *, dtype: torch.dtype, device: torch.device) -> None:
        self.dim = dim
        self.dtype = dtype
        self.device = device
        self.length = 0
        self._data: Optional[Tensor] = None

    @property
    def capacity(self) -> int:
        return 0 if self._data is None else int(self._data.shape[0])

    def _reserve(self, required: int) -> None:
        if required <= self.capacity:
            return
        new_capacity = max(required, max(1, self.capacity * 2))
        new_data = torch.empty((new_capacity, self.dim), dtype=self.dtype, device=self.device)
        if self._data is not None and self.length:
            new_data[: self.length].copy_(self._data[: self.length])
        self._data = new_data

    def append(self, chunk: Tensor) -> Tensor:
        if chunk.ndim != 2 or chunk.shape[-1] != self.dim:
            raise ValueError(f"chunk must be shaped [tokens, {self.dim}]")
        if chunk.device != self.device or chunk.dtype != self.dtype:
            chunk = _to_device_async(chunk, device=self.device, dtype=self.dtype)

        rows = int(chunk.shape[0])
        if rows == 0:
            return torch.empty((0, self.dim), dtype=self.dtype, device=self.device)

        start = self.length
        end = start + rows
        self._reserve(end)
        self._data[start:end].copy_(chunk, non_blocking=True)
        self.length = end
        return self._data[start:end]

    def tensor(self) -> Tensor:
        if self._data is None:
            return torch.empty((0, self.dim), dtype=self.dtype, device=self.device)
        return self._data[: self.length]


def masked_linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    *,
    mask: Optional[Tensor] = None,
    mask_scale: float = 0.02,
    generator: Optional[torch.Generator] = None,
    correction_dtype: Optional[torch.dtype] = None,
    trusted_device: Optional[torch.device | str] = None,
    untrusted_device: Optional[torch.device | str] = None,
    trusted_dtype: Optional[torch.dtype] = None,
    untrusted_dtype: Optional[torch.dtype] = None,
    output_device: Optional[torch.device | str] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> MaskedLinearResult:
    """Compute F.linear(x, weight, bias) while offloading masked input.

    PyTorch Linear stores weight as [out_features, in_features].
    The untrusted side computes (x + r) W.T. The trusted side subtracts r W.T.
    """

    trusted_device = _device_or_default(trusted_device, x.device)
    untrusted_device = _device_or_default(untrusted_device, x.device)
    untrusted_dtype = untrusted_dtype or x.dtype
    trusted_dtype = trusted_dtype or _default_trusted_dtype(trusted_device, x.dtype)

    x_t = x.to(device=trusted_device, dtype=trusted_dtype)
    weight_t = weight.to(device=trusted_device, dtype=trusted_dtype)
    bias_t = bias.to(device=trusted_device, dtype=trusted_dtype) if bias is not None else None

    if mask is None:
        mask = _randn(
            x_t.shape,
            dtype=x_t.dtype,
            device=x_t.device,
            generator=generator,
            scale=mask_scale,
        )
    else:
        mask = mask.to(device=trusted_device, dtype=trusted_dtype)
    if mask.shape != x.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} != input shape {tuple(x.shape)}")

    # Trusted side: hide the private activation before it leaves the boundary.
    masked_input_t = x_t + mask
    masked_input = _to_device_async(masked_input_t, device=untrusted_device, dtype=untrusted_dtype)
    weight_u = _to_device_async(weight, device=untrusted_device, dtype=untrusted_dtype)

    # Untrusted side: perform only the expensive matmul on masked data.
    masked_output = UntrustedGPU.linear(masked_input, weight_u)
    masked_output_t, output_copy_stream = _copy_to_trusted_async(
        masked_output,
        device=trusted_device,
        dtype=correction_dtype or trusted_dtype,
    )

    # Trusted side: compute correction while the untrusted matmul can run asynchronously.
    trusted_out_dtype = correction_dtype or trusted_dtype
    if correction_dtype is None:
        correction = F.linear(mask, weight_t, bias=None)
    else:
        correction = F.linear(
            mask.to(correction_dtype),
            weight_t.to(correction_dtype),
            bias=None,
        )

    _wait_for_copy(output_copy_stream)
    output = masked_output_t - correction

    if bias_t is not None:
        output = output + bias_t.to(output.dtype)

    if output_device is not None or output_dtype is not None:
        output = output.to(
            device=_device_or_default(output_device, output.device),
            dtype=output_dtype or output.dtype,
        )

    return MaskedLinearResult(
        output=output,
        masked_input=masked_input,
        masked_output=masked_output,
        correction=correction,
    )


def _q_times_low_rank_t(q: Tensor, mask: LowRankMask) -> Tensor:
    # q @ (A B).T = (q @ B.T) @ A.T
    return torch.matmul(torch.matmul(q, mask.right.transpose(-1, -2)), mask.left.transpose(-1, -2))


def _low_rank_times_k_t(mask: LowRankMask, k: Tensor) -> Tensor:
    # (A B) @ k.T = A @ (B @ k.T)
    return torch.matmul(mask.left, torch.matmul(mask.right, k.transpose(-1, -2)))


def _low_rank_cross(mask_q: LowRankMask, mask_k: LowRankMask) -> Tensor:
    # (Aq Bq) @ (Ak Bk).T = Aq @ (Bq @ Bk.T) @ Ak.T
    middle = torch.matmul(mask_q.right, mask_k.right.transpose(-1, -2))
    return torch.matmul(torch.matmul(mask_q.left, middle), mask_k.left.transpose(-1, -2))


def _x_times_low_rank(x: Tensor, mask: LowRankMask) -> Tensor:
    # x @ (A B) = (x @ A) @ B
    return torch.matmul(torch.matmul(x, mask.left), mask.right)


def _low_rank_times_x(mask: LowRankMask, x: Tensor) -> Tensor:
    # (A B) @ x = A @ (B @ x)
    return torch.matmul(mask.left, torch.matmul(mask.right, x))


def _low_rank_product(mask_left: LowRankMask, mask_right: LowRankMask) -> Tensor:
    # (Al Bl) @ (Ar Br) = Al @ (Bl @ Ar) @ Br
    middle = torch.matmul(mask_left.right, mask_right.left)
    return torch.matmul(torch.matmul(mask_left.left, middle), mask_right.right)


def masked_qk(
    q: Tensor,
    k: Tensor,
    *,
    rank: int = 4,
    q_mask: Optional[LowRankMask] = None,
    k_mask: Optional[LowRankMask] = None,
    mask_scale: float = 0.02,
    generator: Optional[torch.Generator] = None,
    trusted_device: Optional[torch.device | str] = None,
    untrusted_device: Optional[torch.device | str] = None,
    trusted_dtype: Optional[torch.dtype] = None,
    untrusted_dtype: Optional[torch.dtype] = None,
) -> MaskedQKResult:
    """Compute q @ k.T with low-rank additive masks.

    q and k are shaped [..., rows, dim]. Batch dimensions must match.
    """

    if q.shape[:-2] != k.shape[:-2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have matching batch dimensions and hidden size")

    trusted_device = _device_or_default(trusted_device, q.device)
    untrusted_device = _device_or_default(untrusted_device, q.device)
    untrusted_dtype = untrusted_dtype or q.dtype
    trusted_dtype = trusted_dtype or _default_trusted_dtype(trusted_device, q.dtype)
    q = q.to(device=trusted_device, dtype=trusted_dtype)
    k = k.to(device=trusted_device, dtype=trusted_dtype)

    q_mask = q_mask or LowRankMask.random(q, rank, generator=generator, scale=mask_scale)
    k_mask = k_mask or LowRankMask.random(k, rank, generator=generator, scale=mask_scale)
    q_mask = LowRankMask(
        q_mask.left.to(device=trusted_device, dtype=trusted_dtype),
        q_mask.right.to(device=trusted_device, dtype=trusted_dtype),
    )
    k_mask = LowRankMask(
        k_mask.left.to(device=trusted_device, dtype=trusted_dtype),
        k_mask.right.to(device=trusted_device, dtype=trusted_dtype),
    )

    # Trusted side: materialize low-rank masks only for the tensors sent out.
    masked_q_t = q + q_mask.materialize()
    masked_k_t = k + k_mask.materialize()
    masked_q = _to_device_async(masked_q_t, device=untrusted_device, dtype=untrusted_dtype)
    masked_k = _to_device_async(masked_k_t, device=untrusted_device, dtype=untrusted_dtype)

    # Untrusted side: launch masked QK; trusted correction below can overlap on CUDA.
    masked_output = UntrustedGPU.qk(masked_q, masked_k)
    masked_output_t, output_copy_stream = _copy_to_trusted_async(
        masked_output,
        device=trusted_device,
        dtype=trusted_dtype,
    )

    # Trusted side: remove Q @ Rk.T, Rq @ K.T, and Rq @ Rk.T.
    correction = (
        _q_times_low_rank_t(q, k_mask)
        + _low_rank_times_k_t(q_mask, k)
        + _low_rank_cross(q_mask, k_mask)
    )
    _wait_for_copy(output_copy_stream)
    output = masked_output_t - correction

    return MaskedQKResult(
        output=output,
        masked_q=masked_q,
        masked_k=masked_k,
        masked_output=masked_output,
        correction=correction,
        q_mask=q_mask,
        k_mask=k_mask,
    )


def masked_pv(
    p: Tensor,
    v: Tensor,
    *,
    rank_p: int = 4,
    rank_v: int = 4,
    p_mask: Optional[LowRankMask] = None,
    v_mask: Optional[LowRankMask] = None,
    masked_v: Optional[Tensor] = None,
    mask_scale: float = 0.02,
    generator: Optional[torch.Generator] = None,
    trusted_device: Optional[torch.device | str] = None,
    untrusted_device: Optional[torch.device | str] = None,
    trusted_dtype: Optional[torch.dtype] = None,
    untrusted_dtype: Optional[torch.dtype] = None,
) -> MaskedPVResult:
    """Compute p @ v with additive masks on both attention probabilities and V.

    p is the softmax output shaped [..., query_rows, key_rows].
    v is the private value cache shaped [..., key_rows, head_dim].
    If masked_v is provided, v_mask must also be provided and no new V mask is generated.
    """

    if p.shape[:-2] != v.shape[:-2] or p.shape[-1] != v.shape[-2]:
        raise ValueError("p and v must have matching batch dimensions and key length")
    if (masked_v is None) != (v_mask is None):
        raise ValueError("masked_v and v_mask must be provided together")

    trusted_device = _device_or_default(trusted_device, p.device)
    untrusted_device = _device_or_default(untrusted_device, p.device)
    untrusted_dtype = untrusted_dtype or p.dtype
    trusted_dtype = trusted_dtype or _default_trusted_dtype(trusted_device, p.dtype)
    p = p.to(device=trusted_device, dtype=trusted_dtype)
    v = v.to(device=trusted_device, dtype=trusted_dtype)

    p_mask = p_mask or LowRankMask.random(p, rank_p, generator=generator, scale=mask_scale)
    p_mask = LowRankMask(
        p_mask.left.to(device=trusted_device, dtype=trusted_dtype),
        p_mask.right.to(device=trusted_device, dtype=trusted_dtype),
    )
    if v_mask is None:
        v_mask = LowRankMask.random(v, rank_v, generator=generator, scale=mask_scale)
        masked_v_t = v + v_mask.materialize()
        masked_v = _to_device_async(masked_v_t, device=untrusted_device, dtype=untrusted_dtype)
    else:
        v_mask = LowRankMask(
            v_mask.left.to(device=trusted_device, dtype=trusted_dtype),
            v_mask.right.to(device=trusted_device, dtype=trusted_dtype),
        )
        masked_v = _to_device_async(masked_v, device=untrusted_device, dtype=untrusted_dtype)

    # Trusted side: hide both P and V before offloading P @ V.
    masked_p_t = p + p_mask.materialize()
    masked_p = _to_device_async(masked_p_t, device=untrusted_device, dtype=untrusted_dtype)

    # Untrusted side: launch masked PV; trusted correction below can overlap on CUDA.
    masked_output = UntrustedGPU.pv(masked_p, masked_v)
    masked_output_t, output_copy_stream = _copy_to_trusted_async(
        masked_output,
        device=trusted_device,
        dtype=trusted_dtype,
    )

    # Trusted side: remove P @ Rv, Rp @ V, and Rp @ Rv.
    correction = (
        _x_times_low_rank(p, v_mask)
        + _low_rank_times_x(p_mask, v)
        + _low_rank_product(p_mask, v_mask)
    )
    _wait_for_copy(output_copy_stream)
    output = masked_output_t - correction

    return MaskedPVResult(
        output=output,
        masked_p=masked_p,
        masked_v=masked_v,
        masked_output=masked_output,
        correction=correction,
        p_mask=p_mask,
        v_mask=v_mask,
    )


@dataclass
class MaskedKVQueryResult:
    output: Tensor
    masked_query: Tensor
    masked_output: Tensor
    correction: Tensor


class MaskedKVCache:
    """Dynamic masked K cache for decode-time q @ K.T.

    New key chunks are masked with low-rank masks before they are appended to the
    simulated GPU cache. Queries use a small random vector u and a fixed basis G:

        q_masked = q + u @ G

    The trusted side precomputes G @ K_masked.T when chunks are appended, so the
    online correction for the query mask is only u @ precomputed.
    """

    def __init__(
        self,
        dim: int,
        *,
        key_rank: int = 4,
        query_rank: int = 4,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        trusted_device: Optional[torch.device | str] = "cpu",
        untrusted_device: Optional[torch.device | str] = None,
        trusted_dtype: Optional[torch.dtype] = None,
        mask_scale: float = 0.02,
        offload_pv: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if key_rank <= 0 or query_rank <= 0:
            raise ValueError("key_rank and query_rank must be positive")

        self.dim = dim
        self.key_rank = key_rank
        self.query_rank = query_rank
        self.dtype = dtype
        self.untrusted_device = _device_or_default(untrusted_device, device or _default_untrusted_device())
        self.trusted_device = _device_or_default(trusted_device, "cpu")
        self.trusted_dtype = trusted_dtype or _default_trusted_dtype(self.trusted_device, dtype)
        # Keep the old attribute as the GPU-side device for compatibility with demos.
        self.device = self.untrusted_device
        self.mask_scale = mask_scale
        self.offload_pv = offload_pv
        self.generator = generator

        self.query_basis = _randn(
            (query_rank, dim),
            dtype=self.trusted_dtype,
            device=self.trusted_device,
            generator=generator,
            scale=mask_scale,
        )
        # Keep private K on the trusted side and masked K in one GPU-visible buffer.
        self._private_buffer = _TokenBuffer(dim, dtype=self.trusted_dtype, device=self.trusted_device)
        self._masked_buffer = _TokenBuffer(dim, dtype=dtype, device=self.untrusted_device)
        self._key_masks: List[LowRankMask] = []
        self._basis_to_masked_buffer = _TokenBuffer(
            query_rank,
            dtype=self.trusted_dtype,
            device=self.trusted_device,
        )

    def append(self, keys: Tensor) -> Tensor:
        """Append private keys and return the masked chunk visible to the GPU."""

        if keys.ndim != 2 or keys.shape[-1] != self.dim:
            raise ValueError(f"keys must be shaped [tokens, {self.dim}]")

        keys = keys.to(device=self.trusted_device, dtype=self.trusted_dtype)
        # Trusted side: mask the newly produced K chunk before appending to GPU cache.
        mask = LowRankMask.random(
            keys,
            self.key_rank,
            generator=self.generator,
            scale=self.mask_scale,
        )
        masked_t = keys + mask.materialize()
        masked = _to_device_async(masked_t, device=self.untrusted_device, dtype=self.dtype)

        self._private_buffer.append(keys)
        masked_slice = self._masked_buffer.append(masked)
        self._key_masks.append(mask)
        # Trusted side: precompute G @ K_masked.T so query correction stays small.
        basis_to_masked = torch.matmul(self.query_basis, masked_t.transpose(-1, -2))
        self._basis_to_masked_buffer.append(basis_to_masked.transpose(0, 1).contiguous())
        return masked_slice

    @property
    def seq_len(self) -> int:
        return self._masked_buffer.length

    @property
    def private_keys(self) -> Tensor:
        return self._private_buffer.tensor()

    @property
    def masked_keys(self) -> Tensor:
        return self._masked_buffer.tensor()

    @property
    def basis_to_masked(self) -> Tensor:
        return self._basis_to_masked_buffer.tensor().transpose(0, 1)

    def query(self, q: Tensor) -> MaskedKVQueryResult:
        if self.seq_len == 0:
            raise RuntimeError("append at least one key chunk before querying")
        if q.shape[-1] != self.dim:
            raise ValueError(f"query hidden size must be {self.dim}")

        q = q.to(device=self.trusted_device, dtype=self.trusted_dtype)
        # Trusted side: use a fresh small random vector for each private query.
        u = _randn(
            (*q.shape[:-1], self.query_rank),
            dtype=self.trusted_dtype,
            device=self.trusted_device,
            generator=self.generator,
            scale=self.mask_scale,
        )
        query_mask = torch.matmul(u, self.query_basis)
        masked_query_t = q + query_mask
        masked_query = _to_device_async(masked_query_t, device=self.untrusted_device, dtype=self.dtype)

        # Untrusted side: launch masked QK against the masked K cache.
        masked_keys = self.masked_keys
        masked_output = torch.matmul(masked_query, masked_keys.transpose(-1, -2))
        masked_output_t, output_copy_stream = _copy_to_trusted_async(
            masked_output,
            device=self.trusted_device,
            dtype=self.trusted_dtype,
        )

        # Trusted side: subtract the per-key mask terms chunk by chunk.
        q_key_mask_terms = []
        for mask in self._key_masks:
            q_key_mask_terms.append(_q_times_low_rank_t(q.unsqueeze(-2), mask).squeeze(-2))
        q_key_mask = torch.cat(q_key_mask_terms, dim=-1)

        # Trusted side: subtract the query mask term using the precomputed basis product.
        query_mask_term = torch.matmul(u, self.basis_to_masked)

        correction = q_key_mask + query_mask_term
        _wait_for_copy(output_copy_stream)
        output = masked_output_t - correction
        return MaskedKVQueryResult(
            output=output,
            masked_query=masked_query,
            masked_output=masked_output,
            correction=correction,
        )

    def baseline_query(self, q: Tensor) -> Tensor:
        q = q.to(device=self.trusted_device, dtype=self.trusted_dtype)
        return torch.matmul(q, self.private_keys.transpose(-1, -2))


@dataclass
class MaskedAttentionQueryResult:
    output: Tensor
    scores: Tensor
    probabilities: Tensor
    qk_masked_output: Tensor
    pv_masked_output: Tensor
    correction: Tensor


class MaskedAttentionCache:
    """Decode-time attention cache with masked QK and masked PV offload.

    GPU-visible state:
        K_masked, V_masked

    Trusted state:
        private K/V, low-rank K/V mask factors, and the query-mask table.

    Query flow:
        1. GPU computes masked q @ K.T.
        2. Trusted side corrects scores and runs softmax.
        3. GPU computes masked P @ V.
        4. Trusted side corrects the final attention output.
    """

    def __init__(
        self,
        dim: int,
        *,
        key_rank: int = 4,
        query_rank: int = 4,
        prob_rank: int = 4,
        value_rank: int = 4,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        trusted_device: Optional[torch.device | str] = "cpu",
        untrusted_device: Optional[torch.device | str] = None,
        trusted_dtype: Optional[torch.dtype] = None,
        mask_scale: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if prob_rank <= 0 or value_rank <= 0:
            raise ValueError("prob_rank and value_rank must be positive")

        self.dim = dim
        self.prob_rank = prob_rank
        self.value_rank = value_rank
        self.dtype = dtype
        self.untrusted_device = _device_or_default(untrusted_device, device or _default_untrusted_device())
        self.trusted_device = _device_or_default(trusted_device, "cpu")
        self.trusted_dtype = trusted_dtype or _default_trusted_dtype(self.trusted_device, dtype)
        self.device = self.untrusted_device
        self.mask_scale = mask_scale
        self.generator = generator

        self.k_cache = MaskedKVCache(
            dim,
            key_rank=key_rank,
            query_rank=query_rank,
            dtype=dtype,
            device=self.untrusted_device,
            trusted_device=self.trusted_device,
            untrusted_device=self.untrusted_device,
            trusted_dtype=self.trusted_dtype,
            mask_scale=mask_scale,
            generator=generator,
        )
        self._private_value_buffer = _TokenBuffer(dim, dtype=self.trusted_dtype, device=self.trusted_device)
        self._masked_value_buffer = _TokenBuffer(dim, dtype=dtype, device=self.untrusted_device)
        self._value_masks: List[LowRankMask] = []

    def append(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        """Append private K/V chunks and return the masked chunks visible to GPU."""

        if keys.shape != values.shape:
            raise ValueError("keys and values must have the same shape")
        if values.ndim != 2 or values.shape[-1] != self.dim:
            raise ValueError(f"values must be shaped [tokens, {self.dim}]")

        masked_keys = self.k_cache.append(keys)

        values = values.to(device=self.trusted_device, dtype=self.trusted_dtype)
        if self.offload_pv:
            # Trusted side: V gets its own dynamic low-rank mask before GPU caching.
            value_mask = LowRankMask.random(
                values,
                self.value_rank,
                generator=self.generator,
                scale=self.mask_scale,
            )
            masked_values_t = values + value_mask.materialize()
            masked_values = _to_device_async(masked_values_t, device=self.untrusted_device, dtype=self.dtype)
            self._value_masks.append(value_mask)
        else:
            value_mask = None
            masked_values = torch.empty((values.shape[0], self.dim), dtype=self.dtype, device=self.untrusted_device)

        self._private_value_buffer.append(values)
        masked_value_slice = self._masked_value_buffer.append(masked_values)
        return masked_keys, masked_value_slice

    @property
    def seq_len(self) -> int:
        return self.k_cache.seq_len

    @property
    def private_values(self) -> Tensor:
        return self._private_value_buffer.tensor()

    @property
    def masked_values(self) -> Tensor:
        return self._masked_value_buffer.tensor()

    @property
    def private_keys(self) -> Tensor:
        return self.k_cache.private_keys

    @property
    def masked_keys(self) -> Tensor:
        return self.k_cache.masked_keys

    def _p_times_value_masks(self, p: Tensor) -> Tensor:
        """Compute P @ Rv chunk by chunk without materializing all Rv."""

        terms = []
        start = 0
        for value_mask in self._value_masks:
            chunk_len = value_mask.left.shape[-2]
            p_chunk = p[..., start : start + chunk_len]
            terms.append(_x_times_low_rank(p_chunk, value_mask))
            start += chunk_len
        return torch.stack(terms, dim=0).sum(dim=0)

    def _p_mask_times_value_masks(self, p_mask: LowRankMask) -> Tensor:
        """Compute Rp @ Rv for a full P mask and chunked V masks."""

        terms = []
        start = 0
        for value_mask in self._value_masks:
            chunk_len = value_mask.left.shape[-2]
            p_right_chunk = p_mask.right[..., start : start + chunk_len]
            middle = torch.matmul(p_right_chunk, value_mask.left)
            terms.append(torch.matmul(torch.matmul(p_mask.left, middle), value_mask.right))
            start += chunk_len
        return torch.stack(terms, dim=0).sum(dim=0)

    def query(
        self,
        q: Tensor,
        *,
        scale: Optional[float] = None,
        attention_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        training: bool = False,
    ) -> MaskedAttentionQueryResult:
        if self.seq_len == 0:
            raise RuntimeError("append at least one K/V chunk before querying")

        q = q.to(device=self.trusted_device, dtype=self.trusted_dtype)
        scale = scale if scale is not None else 1.0 / math.sqrt(self.dim)

        # Step 1-2: masked QK is offloaded, then scores are corrected in trusted code.
        qk_result = self.k_cache.query(q)
        scores = qk_result.output * scale
        if attention_mask is not None:
            scores = scores + attention_mask.to(device=self.trusted_device, dtype=scores.dtype)

        # Trusted side: softmax stays inside the boundary by design.
        probabilities = torch.softmax(scores.float(), dim=-1).to(self.trusted_dtype)
        if dropout_p:
            probabilities = F.dropout(probabilities, p=dropout_p, training=training)

        if not self.offload_pv:
            output = torch.matmul(probabilities, self.private_values)
            return MaskedAttentionQueryResult(
                output=output,
                scores=scores,
                probabilities=probabilities,
                qk_masked_output=qk_result.masked_output,
                pv_masked_output=output,
                correction=torch.zeros_like(output),
            )

        # Step 3: mask P and use the dynamic masked V cache for GPU-side P @ V.
        p_mask = LowRankMask.random(
            probabilities,
            self.prob_rank,
            generator=self.generator,
            scale=self.mask_scale,
        )
        masked_p_t = probabilities + p_mask.materialize()
        masked_p = _to_device_async(masked_p_t, device=self.untrusted_device, dtype=self.dtype)
        masked_output = UntrustedGPU.pv(masked_p, self.masked_values)
        masked_output_t, output_copy_stream = _copy_to_trusted_async(
            masked_output,
            device=self.trusted_device,
            dtype=self.trusted_dtype,
        )

        # Step 4: remove P @ Rv, Rp @ V, and Rp @ Rv.
        correction = (
            self._p_times_value_masks(probabilities)
            + _low_rank_times_x(p_mask, self.private_values)
            + self._p_mask_times_value_masks(p_mask)
        )
        _wait_for_copy(output_copy_stream)
        output = masked_output_t - correction

        return MaskedAttentionQueryResult(
            output=output,
            scores=scores,
            probabilities=probabilities,
            qk_masked_output=qk_result.masked_output,
            pv_masked_output=masked_output,
            correction=correction,
        )

    def baseline_query(self, q: Tensor, *, scale: Optional[float] = None) -> Tensor:
        q = q.to(device=self.trusted_device, dtype=self.trusted_dtype)
        scale = scale if scale is not None else 1.0 / math.sqrt(self.dim)
        scores = torch.matmul(q, self.private_keys.transpose(-1, -2)) * scale
        probabilities = torch.softmax(scores.float(), dim=-1).to(self.trusted_dtype)
        return torch.matmul(probabilities, self.private_values)


def batched_masked_attention_query(
    caches: Sequence[MaskedAttentionCache],
    queries: Sequence[Tensor],
    *,
    scale: Optional[float] = None,
    attention_masks: Optional[Sequence[Optional[Tensor]]] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> List[MaskedAttentionQueryResult]:
    """Query multiple masked attention caches with batched GPU QK/PV matmuls.

    The caches may have different KV lengths. The untrusted side sees padded
    masked Q/K/P/V tensors and runs one batched QK plus one batched PV matmul.
    Trusted correction, masking, and softmax remain per request because each
    request owns independent low-rank masks and cache metadata.
    """

    if len(caches) != len(queries):
        raise ValueError("caches and queries must have the same length")
    if not caches:
        return []
    if attention_masks is None:
        attention_masks = [None] * len(caches)
    if len(attention_masks) != len(caches):
        raise ValueError("attention_masks must be None or match caches length")

    dim = caches[0].dim
    trusted_device = caches[0].trusted_device
    untrusted_device = caches[0].untrusted_device
    trusted_dtype = caches[0].trusted_dtype
    untrusted_dtype = caches[0].dtype
    for cache in caches:
        if cache.dim != dim:
            raise ValueError("all caches must have the same dim")
        if cache.trusted_device != trusted_device or cache.untrusted_device != untrusted_device:
            raise ValueError("all caches must use the same trusted and untrusted devices")
        if cache.trusted_dtype != trusted_dtype or cache.dtype != untrusted_dtype:
            raise ValueError("all caches must use the same trusted and untrusted dtypes")
        if cache.offload_pv != caches[0].offload_pv:
            raise ValueError("all caches must use the same PV offload mode")
        if cache.seq_len == 0:
            raise RuntimeError("append at least one K/V chunk to every cache before querying")

    trusted_queries: List[Tensor] = []
    trusted_u: List[Tensor] = []
    masked_queries: List[Tensor] = []
    masked_keys: List[Tensor] = []
    q_lens: List[int] = []
    kv_lens: List[int] = []

    for cache, query in zip(caches, queries):
        if query.ndim != 2 or query.shape[-1] != dim:
            raise ValueError(f"each query must be shaped [query_rows, {dim}]")
        q = query.to(device=trusted_device, dtype=trusted_dtype)
        u = _randn(
            (*q.shape[:-1], cache.k_cache.query_rank),
            dtype=trusted_dtype,
            device=trusted_device,
            generator=cache.generator,
            scale=cache.mask_scale,
        )
        query_mask = torch.matmul(u, cache.k_cache.query_basis)
        masked_query_t = q + query_mask
        masked_query = _to_device_async(masked_query_t, device=untrusted_device, dtype=untrusted_dtype)

        trusted_queries.append(q)
        trusted_u.append(u)
        masked_queries.append(masked_query)
        masked_key = cache.masked_keys
        masked_keys.append(masked_key)
        q_lens.append(q.shape[-2])
        kv_lens.append(masked_key.shape[-2])

    batch_size = len(caches)
    max_q_len = max(q_lens)
    max_kv_len = max(kv_lens)
    padded_q = torch.zeros((batch_size, max_q_len, dim), dtype=untrusted_dtype, device=untrusted_device)
    padded_k = torch.zeros((batch_size, max_kv_len, dim), dtype=untrusted_dtype, device=untrusted_device)
    for index, (masked_query, masked_key, q_len, kv_len) in enumerate(
        zip(masked_queries, masked_keys, q_lens, kv_lens)
    ):
        padded_q[index, :q_len] = masked_query
        padded_k[index, :kv_len] = masked_key

    qk_masked_output = torch.bmm(padded_q, padded_k.transpose(1, 2))
    qk_masked_output_t, qk_copy_stream = _copy_to_trusted_async(
        qk_masked_output,
        device=trusted_device,
        dtype=trusted_dtype,
    )

    qk_corrections: List[Tensor] = []
    for cache, q, u in zip(caches, trusted_queries, trusted_u):
        q_key_mask_terms = []
        for key_mask in cache.k_cache._key_masks:
            q_key_mask_terms.append(_q_times_low_rank_t(q.unsqueeze(-2), key_mask).squeeze(-2))
        q_key_mask = torch.cat(q_key_mask_terms, dim=-1)
        query_mask_term = torch.matmul(u, cache.k_cache.basis_to_masked)
        qk_corrections.append(q_key_mask + query_mask_term)

    _wait_for_copy(qk_copy_stream)

    probabilities: List[Tensor] = []
    p_masks: List[LowRankMask] = []
    scores_list: List[Tensor] = []
    for index, (cache, attention_mask, q_len, kv_len, qk_correction) in enumerate(
        zip(caches, attention_masks, q_lens, kv_lens, qk_corrections)
    ):
        unscaled_scores = qk_masked_output_t[index, :q_len, :kv_len] - qk_correction
        request_scale = scale if scale is not None else 1.0 / math.sqrt(cache.dim)
        scores = unscaled_scores * request_scale
        if attention_mask is not None:
            scores = scores + attention_mask.to(device=trusted_device, dtype=scores.dtype)
        probs = torch.softmax(scores.float(), dim=-1).to(trusted_dtype)
        if dropout_p:
            probs = F.dropout(probs, p=dropout_p, training=training)
        scores_list.append(scores)
        probabilities.append(probs)

    if not caches[0].offload_pv:
        results: List[MaskedAttentionQueryResult] = []
        for index, (cache, probs, q_len, kv_len) in enumerate(zip(caches, probabilities, q_lens, kv_lens)):
            output = torch.matmul(probs, cache.private_values)
            results.append(
                MaskedAttentionQueryResult(
                    output=output,
                    scores=scores_list[index],
                    probabilities=probs,
                    qk_masked_output=qk_masked_output[index, :q_len, :kv_len],
                    pv_masked_output=output,
                    correction=torch.zeros_like(output),
                )
            )
        return results

    for cache, probs in zip(caches, probabilities):
        p_masks.append(
            LowRankMask.random(
                probs,
                cache.prob_rank,
                generator=cache.generator,
                scale=cache.mask_scale,
            )
        )

    padded_p = torch.zeros((batch_size, max_q_len, max_kv_len), dtype=untrusted_dtype, device=untrusted_device)
    padded_v = torch.zeros((batch_size, max_kv_len, dim), dtype=untrusted_dtype, device=untrusted_device)
    for index, (cache, probs, p_mask, q_len, kv_len) in enumerate(zip(caches, probabilities, p_masks, q_lens, kv_lens)):
        masked_p_t = probs + p_mask.materialize()
        padded_p[index, :q_len, :kv_len] = _to_device_async(
            masked_p_t,
            device=untrusted_device,
            dtype=untrusted_dtype,
        )
        padded_v[index, :kv_len] = cache.masked_values

    pv_masked_output = torch.bmm(padded_p, padded_v)
    pv_masked_output_t, pv_copy_stream = _copy_to_trusted_async(
        pv_masked_output,
        device=trusted_device,
        dtype=trusted_dtype,
    )

    pv_corrections: List[Tensor] = []
    for cache, probs, p_mask in zip(caches, probabilities, p_masks):
        pv_corrections.append(
            cache._p_times_value_masks(probs)
            + _low_rank_times_x(p_mask, cache.private_values)
            + cache._p_mask_times_value_masks(p_mask)
        )

    _wait_for_copy(pv_copy_stream)

    results: List[MaskedAttentionQueryResult] = []
    for index, (probs, correction, q_len, kv_len) in enumerate(zip(probabilities, pv_corrections, q_lens, kv_lens)):
        output = pv_masked_output_t[index, :q_len] - correction
        results.append(
            MaskedAttentionQueryResult(
                output=output,
                scores=scores_list[index],
                probabilities=probs,
                qk_masked_output=qk_masked_output[index, :q_len, :kv_len],
                pv_masked_output=pv_masked_output[index, :q_len],
                correction=correction,
            )
        )

    return results
