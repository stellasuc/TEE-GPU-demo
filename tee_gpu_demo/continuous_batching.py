"""Continuous batching scheduler for masked attention cache demos.

This module models the serving-side scheduling pattern used by LLM engines:
new requests can arrive while older requests are still decoding. Each request
owns its masked K/V cache; every scheduler tick admits ready requests, advances
prefill by one chunk, and decodes up to ``max_batch_size`` active requests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor

from .masked_ops import MaskedAttentionCache
from .masked_ops import batched_masked_attention_query


@dataclass
class ContinuousRequest:
    """Synthetic request data for continuous batching experiments.

    The tensors represent already-projected per-head K/V/Q vectors. During
    decode step ``t`` the scheduler appends ``decode_keys[t]`` /
    ``decode_values[t]`` to the request cache, then queries with
    ``decode_queries[t]``.
    """

    request_id: str
    prompt_keys: Tensor
    prompt_values: Tensor
    decode_queries: Tensor
    decode_keys: Tensor
    decode_values: Tensor
    arrival_step: int = 0

    def __post_init__(self) -> None:
        if self.prompt_keys.shape != self.prompt_values.shape:
            raise ValueError("prompt_keys and prompt_values must have the same shape")
        if self.decode_queries.shape != self.decode_keys.shape or self.decode_keys.shape != self.decode_values.shape:
            raise ValueError("decode query/key/value tensors must have the same shape")
        if self.prompt_keys.ndim != 2 or self.decode_queries.ndim != 2:
            raise ValueError("request tensors must be shaped [tokens, dim]")
        if self.prompt_keys.shape[-1] != self.decode_queries.shape[-1]:
            raise ValueError("prompt and decode hidden sizes must match")
        if self.arrival_step < 0:
            raise ValueError("arrival_step must be non-negative")

    @property
    def dim(self) -> int:
        return int(self.decode_queries.shape[-1])

    @property
    def prompt_len(self) -> int:
        return int(self.prompt_keys.shape[0])

    @property
    def decode_len(self) -> int:
        return int(self.decode_queries.shape[0])


@dataclass
class ContinuousBatchStep:
    step_index: int
    admitted_ids: tuple[str, ...]
    prefilled_ids: tuple[str, ...]
    decoded_ids: tuple[str, ...]
    finished_ids: tuple[str, ...]
    active_ids: tuple[str, ...]


@dataclass
class ContinuousBatchingResult:
    outputs: Dict[str, Tensor]
    steps: List[ContinuousBatchStep]

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def decode_batches(self) -> int:
        return sum(1 for step in self.steps if step.decoded_ids)

    @property
    def decoded_tokens(self) -> int:
        return sum(len(step.decoded_ids) for step in self.steps)

    @property
    def max_decode_batch(self) -> int:
        return max((len(step.decoded_ids) for step in self.steps), default=0)


@dataclass
class _RequestState:
    request: ContinuousRequest
    cache: MaskedAttentionCache
    prefill_pos: int = 0
    decode_pos: int = 0
    outputs: List[Tensor] = field(default_factory=list)

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def prefill_done(self) -> bool:
        return self.prefill_pos >= self.request.prompt_len

    @property
    def done(self) -> bool:
        return self.prefill_done and self.decode_pos >= self.request.decode_len


class ContinuousBatchingEngine:
    """Simple continuous batching engine backed by per-request masked caches."""

    def __init__(
        self,
        dim: int,
        *,
        max_batch_size: int = 4,
        max_active_requests: Optional[int] = None,
        prefill_chunk: int = 128,
        key_rank: int = 4,
        query_rank: int = 4,
        prob_rank: int = 4,
        value_rank: int = 4,
        dtype: torch.dtype = torch.float32,
        trusted_device: torch.device | str = "cpu",
        untrusted_device: torch.device | str | None = None,
        trusted_dtype: torch.dtype | None = None,
        mask_scale: float = 0.02,
        offload_pv: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if prefill_chunk <= 0:
            raise ValueError("prefill_chunk must be positive")

        self.dim = dim
        self.max_batch_size = max_batch_size
        self.max_active_requests = max_active_requests or max_batch_size
        self.prefill_chunk = prefill_chunk
        self.key_rank = key_rank
        self.query_rank = query_rank
        self.prob_rank = prob_rank
        self.value_rank = value_rank
        self.dtype = dtype
        self.trusted_device = torch.device(trusted_device)
        self.untrusted_device = torch.device(untrusted_device) if untrusted_device is not None else self.trusted_device
        self.trusted_dtype = trusted_dtype or (torch.float32 if self.trusted_device.type == "cpu" else dtype)
        self.mask_scale = mask_scale
        self.offload_pv = offload_pv
        self.generator = generator

        self._pending: List[ContinuousRequest] = []
        self._active: List[_RequestState] = []
        self._finished: Dict[str, Tensor] = {}
        self._steps: List[ContinuousBatchStep] = []
        self._step_index = 0

    def submit(self, request: ContinuousRequest) -> None:
        if request.dim != self.dim:
            raise ValueError(f"request {request.request_id!r} dim {request.dim} != engine dim {self.dim}")
        if request.request_id in self._finished or any(state.request_id == request.request_id for state in self._active):
            raise ValueError(f"duplicate request_id: {request.request_id}")
        if any(pending.request_id == request.request_id for pending in self._pending):
            raise ValueError(f"duplicate request_id: {request.request_id}")
        self._pending.append(request)
        self._pending.sort(key=lambda item: (item.arrival_step, item.request_id))

    def submit_many(self, requests: Sequence[ContinuousRequest]) -> None:
        for request in requests:
            self.submit(request)

    def _new_cache(self) -> MaskedAttentionCache:
        return MaskedAttentionCache(
            dim=self.dim,
            key_rank=self.key_rank,
            query_rank=self.query_rank,
            prob_rank=self.prob_rank,
            value_rank=self.value_rank,
            dtype=self.dtype,
            trusted_device=self.trusted_device,
            untrusted_device=self.untrusted_device,
            trusted_dtype=self.trusted_dtype,
            mask_scale=self.mask_scale,
            offload_pv=self.offload_pv,
            generator=self.generator,
        )

    def _admit_ready(self) -> tuple[str, ...]:
        admitted = []
        while self._pending and len(self._active) < self.max_active_requests:
            if self._pending[0].arrival_step > self._step_index:
                break
            request = self._pending.pop(0)
            self._active.append(_RequestState(request=request, cache=self._new_cache()))
            admitted.append(request.request_id)
        return tuple(admitted)

    def _jump_to_next_arrival_if_idle(self) -> None:
        if self._active or not self._pending:
            return
        self._step_index = max(self._step_index, self._pending[0].arrival_step)

    def step(self) -> ContinuousBatchStep:
        self._jump_to_next_arrival_if_idle()
        admitted_ids = self._admit_ready()

        prefilled = []
        for state in self._active:
            if state.prefill_done:
                continue
            start = state.prefill_pos
            end = min(start + self.prefill_chunk, state.request.prompt_len)
            if end > start:
                state.cache.append(
                    state.request.prompt_keys[start:end],
                    state.request.prompt_values[start:end],
                )
                state.prefill_pos = end
                prefilled.append(state.request_id)

        decoded = []
        decode_states = [item for item in self._active if item.prefill_done and not item.done][: self.max_batch_size]
        decode_queries = []
        for state in decode_states:
            pos = state.decode_pos
            request = state.request
            state.cache.append(request.decode_keys[pos : pos + 1], request.decode_values[pos : pos + 1])
            decode_queries.append(request.decode_queries[pos : pos + 1])

        if decode_states:
            batch_results = batched_masked_attention_query(
                [state.cache for state in decode_states],
                decode_queries,
            )
            for state, query_result in zip(decode_states, batch_results):
                state.outputs.append(query_result.output.squeeze(0))
                state.decode_pos += 1
                decoded.append(state.request_id)

        finished = []
        still_active = []
        for state in self._active:
            if state.done:
                finished.append(state.request_id)
                if state.outputs:
                    self._finished[state.request_id] = torch.stack(state.outputs, dim=0)
                else:
                    self._finished[state.request_id] = torch.empty(
                        (0, self.dim),
                        dtype=self.trusted_dtype,
                        device=self.trusted_device,
                    )
            else:
                still_active.append(state)
        self._active = still_active

        step = ContinuousBatchStep(
            step_index=self._step_index,
            admitted_ids=admitted_ids,
            prefilled_ids=tuple(prefilled),
            decoded_ids=tuple(decoded),
            finished_ids=tuple(finished),
            active_ids=tuple(state.request_id for state in self._active),
        )
        self._steps.append(step)
        self._step_index += 1
        return step

    def run(self, *, max_steps: Optional[int] = None) -> ContinuousBatchingResult:
        while self._pending or self._active:
            if max_steps is not None and len(self._steps) >= max_steps:
                raise RuntimeError(f"continuous batching exceeded max_steps={max_steps}")
            self.step()
        return ContinuousBatchingResult(outputs=dict(self._finished), steps=list(self._steps))


def plain_attention_reference(request: ContinuousRequest, *, scale: Optional[float] = None) -> Tensor:
    """Plain attention reference for one synthetic request."""

    scale = scale if scale is not None else 1.0 / math.sqrt(request.dim)
    keys = [request.prompt_keys]
    values = [request.prompt_values]
    outputs = []
    for pos in range(request.decode_len):
        keys.append(request.decode_keys[pos : pos + 1])
        values.append(request.decode_values[pos : pos + 1])
        key_all = torch.cat(keys, dim=0)
        value_all = torch.cat(values, dim=0)
        q = request.decode_queries[pos : pos + 1]
        scores = (q @ key_all.transpose(-1, -2)) * scale
        probs = torch.softmax(scores.float(), dim=-1).to(value_all.dtype)
        outputs.append((probs @ value_all).squeeze(0))
    if outputs:
        return torch.stack(outputs, dim=0)
    return torch.empty((0, request.dim), dtype=request.prompt_keys.dtype, device=request.prompt_keys.device)
