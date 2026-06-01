"""Helpers for replacing Llama Linear layers with masked Linear layers."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from types import MethodType
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

from .masked_ops import MaskedAttentionCache, masked_linear


DEFAULT_LLAMA_LINEAR_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class MaskedLinear(nn.Module):
    """Drop-in nn.Linear wrapper using additive masking before matmul."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        mask_scale: float = 0.02,
        correction_dtype: torch.dtype | None = None,
        trusted_device: torch.device | str | None = None,
        untrusted_device: torch.device | str | None = None,
        trusted_dtype: torch.dtype | None = None,
        return_to_input_device: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        # Reuse the original parameter objects so loading and memory layout stay unchanged.
        self.weight = base.weight
        self.bias = base.bias
        self.mask_scale = mask_scale
        self.correction_dtype = correction_dtype
        self.trusted_device = trusted_device
        self.untrusted_device = untrusted_device
        self.trusted_dtype = trusted_dtype
        self.return_to_input_device = return_to_input_device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_device = x.device if self.return_to_input_device else self.trusted_device
        output_dtype = x.dtype if self.return_to_input_device else self.trusted_dtype
        return masked_linear(
            x,
            self.weight,
            self.bias,
            mask_scale=self.mask_scale,
            correction_dtype=self.correction_dtype,
            trusted_device=self.trusted_device,
            untrusted_device=self.untrusted_device or x.device,
            trusted_dtype=self.trusted_dtype,
            untrusted_dtype=x.dtype,
            output_device=output_device,
            output_dtype=output_dtype,
        ).output


@dataclass
class PatchReport:
    replaced: int
    names: Tuple[str, ...]


@dataclass
class MaskedAttentionPatchConfig:
    key_rank: int = 4
    query_rank: int = 4
    prob_rank: int = 4
    value_rank: int = 4
    mask_scale: float = 0.02
    trusted_device: torch.device | str | None = "cpu"
    untrusted_device: torch.device | str | None = None
    trusted_dtype: torch.dtype | None = None
    return_to_input_device: bool = False


def _to_device(device: torch.device | str | None, fallback: torch.device) -> torch.device:
    return torch.device(device) if device is not None else fallback


def _get_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def replace_llama_linears(
    model: nn.Module,
    *,
    include_names: Iterable[str] = DEFAULT_LLAMA_LINEAR_NAMES,
    mask_scale: float = 0.02,
    correction_dtype: torch.dtype | None = None,
    trusted_device: torch.device | str | None = None,
    untrusted_device: torch.device | str | None = None,
    trusted_dtype: torch.dtype | None = None,
    return_to_input_device: bool = False,
) -> PatchReport:
    """Replace selected Llama projection layers in-place."""

    include = set(include_names)
    targets = []
    # Collect names first because replacing modules while iterating can skip children.
    for name, module in model.named_modules():
        short_name = name.rsplit(".", 1)[-1]
        if short_name in include and isinstance(module, nn.Linear):
            targets.append(name)

    for name in targets:
        parent, child_name = _get_parent(model, name)
        child = getattr(parent, child_name)
        # Swap only the selected Linear leaf; the rest of the model remains untouched.
        setattr(
            parent,
            child_name,
            MaskedLinear(
                child,
                mask_scale=mask_scale,
                correction_dtype=correction_dtype,
                trusted_device=trusted_device,
                untrusted_device=untrusted_device,
                trusted_dtype=trusted_dtype,
                return_to_input_device=return_to_input_device,
            ),
        )

    return PatchReport(replaced=len(targets), names=tuple(targets))


def _get_num_heads(module: nn.Module) -> int:
    if hasattr(module, "num_heads"):
        return int(module.num_heads)
    if hasattr(module, "num_attention_heads"):
        return int(module.num_attention_heads)
    return int(module.config.num_attention_heads)


def _get_num_key_value_heads(module: nn.Module, num_heads: int) -> int:
    if hasattr(module, "num_key_value_heads"):
        return int(module.num_key_value_heads)
    return int(getattr(module.config, "num_key_value_heads", num_heads))


def _get_head_dim(module: nn.Module, num_heads: int) -> int:
    if hasattr(module, "head_dim"):
        return int(module.head_dim)
    hidden_size = int(getattr(module.config, "hidden_size"))
    return hidden_size // num_heads


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if position_ids is not None and cos.ndim == 2:
        cos = cos[position_ids]
        sin = sin[position_ids]
    if cos.ndim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    except ImportError:
        return _apply_rotary_pos_emb_fallback(q, k, cos, sin, position_ids)

    try:
        return apply_rotary_pos_emb(q, k, cos, sin)
    except TypeError:
        return apply_rotary_pos_emb(q, k, cos, sin, position_ids)


def _compute_position_embeddings(
    module: nn.Module,
    value_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    if position_embeddings is not None:
        return position_embeddings
    rotary_emb = getattr(module, "rotary_emb", None)
    if rotary_emb is None:
        return None
    try:
        return rotary_emb(value_states, position_ids)
    except TypeError:
        return rotary_emb(value_states, seq_len=value_states.shape[-2])


def _cache_seq_length(cache, layer_idx: Optional[int]) -> int:
    if cache is None:
        return 0
    if isinstance(cache, tuple) and cache:
        return int(cache[0].shape[-2])
    if layer_idx is None:
        return 0
    for args in ((layer_idx,), ()):
        try:
            return int(cache.get_seq_length(*args))
        except (AttributeError, TypeError):
            continue
    return 0


def _update_hf_cache(
    cache,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    module: nn.Module,
    cache_position: Optional[torch.Tensor],
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, object]:
    if cache is None:
        return key_states, value_states, cache
    if isinstance(cache, tuple):
        key_states = torch.cat([cache[0], key_states], dim=2)
        value_states = torch.cat([cache[1], value_states], dim=2)
        return key_states, value_states, (key_states, value_states)

    layer_idx = getattr(module, "layer_idx", None)
    cache_kwargs = {}
    if position_embeddings is not None:
        cos, sin = position_embeddings
        cache_kwargs.update({"cos": cos, "sin": sin})
    if cache_position is not None:
        cache_kwargs["cache_position"] = cache_position

    attempts = (
        lambda: cache.update(key_states, value_states, layer_idx, cache_kwargs),
        lambda: cache.update(key_states, value_states, layer_idx),
        lambda: cache.update(key_states, value_states),
    )
    last_error = None
    for attempt in attempts:
        try:
            key_states, value_states = attempt()
            return key_states, value_states, cache
        except TypeError as exc:
            last_error = exc
    raise last_error


def _empty_or_causal_mask(q_len: int, kv_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    past_len = max(kv_len - q_len, 0)
    rows = torch.arange(q_len, device=device).unsqueeze(-1)
    cols = torch.arange(kv_len, device=device).unsqueeze(0)
    blocked = cols > (past_len + rows)
    mask = torch.zeros((q_len, kv_len), dtype=dtype, device=device)
    return mask.masked_fill(blocked, torch.finfo(dtype).min)


def _select_attention_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    batch_index: int,
    head_index: int,
    q_len: int,
    kv_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if attention_mask is None:
        return _empty_or_causal_mask(q_len, kv_len, dtype, device)

    mask = attention_mask
    if mask.ndim == 4:
        head_slot = 0 if mask.shape[1] == 1 else head_index
        selected = mask[batch_index, head_slot, -q_len:, :kv_len]
    elif mask.ndim == 3:
        selected = mask[batch_index, -q_len:, :kv_len]
    elif mask.ndim == 2:
        selected = mask[batch_index, :kv_len].unsqueeze(0).expand(q_len, kv_len)
        if selected.dtype == torch.bool:
            selected = torch.where(selected, torch.zeros_like(selected, dtype=dtype), torch.finfo(dtype).min)
        elif torch.is_floating_point(selected) and selected.numel():
            selected_max = float(selected.max().item())
            selected_min = float(selected.min().item())
            if selected_max <= 1 and selected_min >= 0:
                selected = (1.0 - selected.to(dtype)) * torch.finfo(dtype).min
        selected = selected.to(dtype) + _empty_or_causal_mask(q_len, kv_len, dtype, selected.device)
    else:
        raise ValueError(f"Unsupported attention_mask shape: {tuple(mask.shape)}")
    return selected.to(device=device, dtype=dtype)


def _new_masked_attention_cache(
    *,
    dim: int,
    dtype: torch.dtype,
    config: MaskedAttentionPatchConfig,
) -> MaskedAttentionCache:
    trusted_device = _to_device(config.trusted_device, torch.device("cpu"))
    untrusted_device = _to_device(config.untrusted_device, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    trusted_dtype = config.trusted_dtype or (torch.float32 if trusted_device.type == "cpu" else dtype)
    return MaskedAttentionCache(
        dim=dim,
        key_rank=config.key_rank,
        query_rank=config.query_rank,
        prob_rank=config.prob_rank,
        value_rank=config.value_rank,
        dtype=dtype,
        trusted_dtype=trusted_dtype,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
        mask_scale=config.mask_scale,
    )


def _masked_cache_len(cache: MaskedAttentionCache) -> int:
    return int(cache.private_keys.shape[0])


def _cache_grid(module: nn.Module, batch_size: int, num_heads: int) -> list[list[Optional[MaskedAttentionCache]]]:
    grid = getattr(module, "_tee_gpu_attention_caches", None)
    if (
        grid is None
        or len(grid) != batch_size
        or any(len(row) != num_heads for row in grid)
    ):
        grid = [[None for _ in range(num_heads)] for _ in range(batch_size)]
        setattr(module, "_tee_gpu_attention_caches", grid)
    return grid


def _module_is_llama_attention(module: nn.Module) -> bool:
    name = module.__class__.__name__
    if name in {"LlamaAttention", "LlamaSdpaAttention", "LlamaFlashAttention2"}:
        return True
    return all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj")) and (
        hasattr(module, "head_dim") or hasattr(module, "config")
    )


def masked_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    past_key_values=None,
    **kwargs,
):
    """LlamaAttention.forward replacement backed by MaskedAttentionCache."""

    config: MaskedAttentionPatchConfig = self._tee_gpu_attention_config
    input_device = hidden_states.device
    input_dtype = hidden_states.dtype
    batch_size, q_len, _ = hidden_states.shape
    num_heads = _get_num_heads(self)
    num_key_value_heads = _get_num_key_value_heads(self, num_heads)
    num_key_value_groups = int(getattr(self, "num_key_value_groups", num_heads // num_key_value_heads))
    head_dim = _get_head_dim(self, num_heads)
    hidden_shape = (batch_size, q_len, -1, head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    position_embeddings = _compute_position_embeddings(self, value_states, position_ids, position_embeddings)
    if position_embeddings is not None:
        query_states, key_states = _apply_rotary_pos_emb(
            query_states,
            key_states,
            position_embeddings[0],
            position_embeddings[1],
            position_ids,
        )

    new_key_states = key_states
    new_value_states = value_states
    cache = past_key_values if past_key_values is not None else past_key_value
    previous_cache_length = _cache_seq_length(cache, getattr(self, "layer_idx", None))
    if cache is not None:
        key_states, value_states, cache = _update_hf_cache(
            cache,
            key_states,
            value_states,
            self,
            cache_position,
            position_embeddings,
        )
    elif getattr(self, "_tee_gpu_attention_returns_cache", False) and use_cache:
        cache = (key_states, value_states)

    key_states = _repeat_kv(key_states, num_key_value_groups)
    value_states = _repeat_kv(value_states, num_key_value_groups)
    new_key_states = _repeat_kv(new_key_states, num_key_value_groups)
    new_value_states = _repeat_kv(new_value_states, num_key_value_groups)

    kv_len = key_states.shape[-2]
    scale = float(getattr(self, "scaling", 1.0 / math.sqrt(head_dim)))
    dropout_p = float(getattr(self, "attention_dropout", 0.0)) if self.training else 0.0
    cache_grid = _cache_grid(self, batch_size, num_heads) if (use_cache or cache is not None) else None

    head_outputs = []
    attn_weights = [] if output_attentions else None
    for batch_index in range(batch_size):
        batch_outputs = []
        batch_weights = [] if output_attentions else None
        for head_index in range(num_heads):
            query = query_states[batch_index, head_index]
            mask = _select_attention_mask(
                attention_mask,
                batch_index=batch_index,
                head_index=head_index,
                q_len=q_len,
                kv_len=kv_len,
                dtype=torch.float32,
                device=query.device,
            )

            if cache_grid is None:
                masked_cache = _new_masked_attention_cache(dim=head_dim, dtype=query.dtype, config=config)
                masked_cache.append(key_states[batch_index, head_index], value_states[batch_index, head_index])
            else:
                masked_cache = cache_grid[batch_index][head_index]
                expected_previous = max(kv_len - new_key_states.shape[-2], 0)
                if masked_cache is None or _masked_cache_len(masked_cache) != expected_previous:
                    masked_cache = _new_masked_attention_cache(dim=head_dim, dtype=query.dtype, config=config)
                    if expected_previous == previous_cache_length and expected_previous < kv_len:
                        masked_cache.append(
                            key_states[batch_index, head_index, :expected_previous],
                            value_states[batch_index, head_index, :expected_previous],
                        ) if expected_previous else None
                        masked_cache.append(
                            new_key_states[batch_index, head_index],
                            new_value_states[batch_index, head_index],
                        )
                    else:
                        masked_cache.append(key_states[batch_index, head_index], value_states[batch_index, head_index])
                    cache_grid[batch_index][head_index] = masked_cache
                else:
                    masked_cache.append(
                        new_key_states[batch_index, head_index],
                        new_value_states[batch_index, head_index],
                    )

            result = masked_cache.query(
                query,
                scale=scale,
                attention_mask=mask,
                dropout_p=dropout_p,
                training=self.training,
            )
            batch_outputs.append(result.output)
            if batch_weights is not None:
                batch_weights.append(result.probabilities)

        head_outputs.append(torch.stack(batch_outputs, dim=0))
        if attn_weights is not None:
            attn_weights.append(torch.stack(batch_weights, dim=0))

    attn_output = torch.stack(head_outputs, dim=0)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, q_len, num_heads * head_dim)
    attn_output = self.o_proj(attn_output)
    if config.return_to_input_device:
        attn_output = attn_output.to(device=input_device, dtype=input_dtype)

    if attn_weights is not None:
        attn_weights = torch.stack(attn_weights, dim=0)

    if getattr(self, "_tee_gpu_attention_returns_cache", False):
        return attn_output, attn_weights, cache if use_cache else None
    return attn_output, attn_weights


def replace_llama_attentions(
    model: nn.Module,
    *,
    key_rank: int = 4,
    query_rank: int = 4,
    prob_rank: int = 4,
    value_rank: int = 4,
    mask_scale: float = 0.02,
    trusted_device: torch.device | str | None = "cpu",
    untrusted_device: torch.device | str | None = None,
    trusted_dtype: torch.dtype | None = None,
    return_to_input_device: bool = False,
) -> PatchReport:
    """Replace LlamaAttention.forward in-place with masked attention offload."""

    targets = []
    for name, module in model.named_modules():
        if _module_is_llama_attention(module):
            targets.append(name)

    config = MaskedAttentionPatchConfig(
        key_rank=key_rank,
        query_rank=query_rank,
        prob_rank=prob_rank,
        value_rank=value_rank,
        mask_scale=mask_scale,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
        trusted_dtype=trusted_dtype,
        return_to_input_device=return_to_input_device,
    )
    for name in targets:
        parent, child_name = _get_parent(model, name)
        module = getattr(parent, child_name)
        if not hasattr(module, "_tee_gpu_original_forward"):
            module._tee_gpu_original_forward = module.forward
        signature = inspect.signature(module._tee_gpu_original_forward)
        module._tee_gpu_attention_returns_cache = (
            "past_key_value" in signature.parameters and "past_key_values" not in signature.parameters
        )
        module._tee_gpu_attention_config = config
        module._tee_gpu_attention_caches = None
        module.forward = MethodType(masked_llama_attention_forward, module)

    return PatchReport(replaced=len(targets), names=tuple(targets))
