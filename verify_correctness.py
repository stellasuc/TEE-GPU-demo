"""Correctness checks for the TEE-GPU masked matmul prototype.

This script compares the masked/offloaded path against plain PyTorch baselines
on random tensors. It does not download or load a Hugging Face model.

Examples:
    python verify_correctness.py --untrusted-device cpu
    python verify_correctness.py --trusted-device cpu --untrusted-device cuda
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from types import SimpleNamespace


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    raise SystemExit("Install dependencies first: pip install -r requirements.txt") from exc

from tee_gpu_demo.llama_patch import replace_llama_attentions, replace_llama_linears
from tee_gpu_demo.masked_ops import (
    MaskedAttentionCache,
    MaskedKVCache,
    batched_masked_attention_query,
    masked_linear,
    masked_pv,
    masked_qk,
)


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    max_abs: float
    mean_abs: float


class FakeModernAttention(nn.Module):
    """Small LlamaAttention-shaped module with the modern two-output signature."""

    def __init__(self, hidden_size: int = 12, num_heads: int = 3, num_key_value_heads: int = 1) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
        )
        self.layer_idx = 0
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.head_dim = hidden_size // num_heads
        self.attention_dropout = 0.0
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, **kwargs):
        raise NotImplementedError("This method should be monkey-patched by replace_llama_attentions().")


class FakeLegacyAttention(FakeModernAttention):
    """Small LlamaAttention-shaped module with the legacy cache-return signature."""

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        raise NotImplementedError("This method should be monkey-patched by replace_llama_attentions().")


def default_untrusted_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trusted-device", default="cpu")
    parser.add_argument("--untrusted-device", default=default_untrusted_device())
    parser.add_argument("--trusted-dtype", choices=DTYPES, default="float32")
    parser.add_argument("--untrusted-dtype", choices=DTYPES, default="float32")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mask-scale", type=float, default=0.02)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    return parser.parse_args()


def validate_device(device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA was requested for {device}, but torch.cuda.is_available() is false.")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def causal_mask(q_len: int, kv_len: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    past_len = max(kv_len - q_len, 0)
    rows = torch.arange(q_len, device=device).unsqueeze(-1)
    cols = torch.arange(kv_len, device=device).unsqueeze(0)
    blocked = cols > (past_len + rows)
    mask = torch.zeros((q_len, kv_len), dtype=dtype, device=device)
    return mask.masked_fill(blocked, -1.0e4)


def llama_attention_baseline(
    module: FakeModernAttention,
    query_hidden: torch.Tensor,
    *,
    key_value_hidden: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_value_hidden = query_hidden if key_value_hidden is None else key_value_hidden
    batch_size, q_len, _ = query_hidden.shape
    kv_len = key_value_hidden.shape[1]
    hidden_shape_q = (batch_size, q_len, module.num_heads, module.head_dim)
    hidden_shape_kv = (batch_size, kv_len, module.num_key_value_heads, module.head_dim)

    q = module.q_proj(query_hidden).view(hidden_shape_q).transpose(1, 2)
    k = module.k_proj(key_value_hidden).view(hidden_shape_kv).transpose(1, 2)
    v = module.v_proj(key_value_hidden).view(hidden_shape_kv).transpose(1, 2)
    k = repeat_kv(k, module.num_key_value_groups)
    v = repeat_kv(v, module.num_key_value_groups)

    scores = (q @ k.transpose(-1, -2)) * module.scaling
    if attention_mask is None:
        scores = scores + causal_mask(q_len, kv_len, device=scores.device, dtype=scores.dtype).view(1, 1, q_len, kv_len)
    else:
        scores = scores + attention_mask.to(device=scores.device, dtype=scores.dtype)
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    output = probs @ v
    output = output.transpose(1, 2).contiguous().reshape(batch_size, q_len, module.num_heads * module.head_dim)
    return module.o_proj(output), probs


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> CheckResult:
    actual_f = actual.detach().float().cpu()
    expected_f = expected.detach().float().cpu()
    diff = (actual_f - expected_f).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    passed = torch.allclose(actual_f, expected_f, atol=atol, rtol=rtol)
    status = "PASS" if passed else "FAIL"
    print(f"{status:4s} {name:34s} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}")
    return CheckResult(name=name, passed=passed, max_abs=max_abs, mean_abs=mean_abs)


def test_masked_linear(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype) -> CheckResult:
    x = torch.randn(2, 3, 16, device=trusted_device, dtype=trusted_dtype)
    weight = torch.randn(12, 16, device=trusted_device, dtype=trusted_dtype)
    bias = torch.randn(12, device=trusted_device, dtype=trusted_dtype)
    expected = F.linear(x, weight, bias)
    actual = masked_linear(
        x,
        weight,
        bias,
        mask_scale=args.mask_scale,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
        trusted_dtype=trusted_dtype,
        untrusted_dtype=untrusted_dtype,
    ).output
    return compare("masked_linear", actual, expected, atol=args.atol, rtol=args.rtol)


def test_masked_qk(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype) -> CheckResult:
    q = torch.randn(2, 4, 16, device=trusted_device, dtype=trusted_dtype)
    k = torch.randn(2, 6, 16, device=trusted_device, dtype=trusted_dtype)
    expected = q @ k.transpose(-1, -2)
    actual = masked_qk(
        q,
        k,
        rank=args.rank,
        mask_scale=args.mask_scale,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
        trusted_dtype=trusted_dtype,
        untrusted_dtype=untrusted_dtype,
    ).output
    return compare("masked_qk", actual, expected, atol=args.atol, rtol=args.rtol)


def test_masked_pv(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype) -> CheckResult:
    p = torch.softmax(torch.randn(2, 4, 6, device=trusted_device, dtype=trusted_dtype), dim=-1)
    v = torch.randn(2, 6, 16, device=trusted_device, dtype=trusted_dtype)
    expected = p @ v
    actual = masked_pv(
        p,
        v,
        rank_p=args.rank,
        rank_v=args.rank,
        mask_scale=args.mask_scale,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
        trusted_dtype=trusted_dtype,
        untrusted_dtype=untrusted_dtype,
    ).output
    return compare("masked_pv", actual, expected, atol=args.atol, rtol=args.rtol)


def test_masked_kv_cache(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype) -> CheckResult:
    cache = MaskedKVCache(
        dim=16,
        key_rank=args.rank,
        query_rank=args.rank,
        dtype=untrusted_dtype,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
        trusted_dtype=trusted_dtype,
        mask_scale=args.mask_scale,
    )
    cache.append(torch.randn(4, 16, device=trusted_device, dtype=trusted_dtype))
    cache.append(torch.randn(3, 16, device=trusted_device, dtype=trusted_dtype))
    q = torch.randn(5, 16, device=trusted_device, dtype=trusted_dtype)
    expected = cache.baseline_query(q)
    actual = cache.query(q).output
    return compare("masked_kv_cache", actual, expected, atol=args.atol, rtol=args.rtol)


def test_masked_attention_cache(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype) -> CheckResult:
    cache = MaskedAttentionCache(
        dim=16,
        key_rank=args.rank,
        query_rank=args.rank,
        prob_rank=args.rank,
        value_rank=args.rank,
        dtype=untrusted_dtype,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
        trusted_dtype=trusted_dtype,
        mask_scale=args.mask_scale,
    )
    keys = [
        torch.randn(4, 16, device=trusted_device, dtype=trusted_dtype),
        torch.randn(3, 16, device=trusted_device, dtype=trusted_dtype),
    ]
    values = [
        torch.randn(4, 16, device=trusted_device, dtype=trusted_dtype),
        torch.randn(3, 16, device=trusted_device, dtype=trusted_dtype),
    ]
    for key_chunk, value_chunk in zip(keys, values):
        cache.append(key_chunk, value_chunk)

    q = torch.randn(3, 16, device=trusted_device, dtype=trusted_dtype)
    key_all = torch.cat(keys, dim=0)
    value_all = torch.cat(values, dim=0)
    mask = torch.zeros(3, 7, device=trusted_device, dtype=trusted_dtype)
    mask[0, 5:] = -1.0e4
    mask[1, 6:] = -1.0e4
    scale = 1.0 / math.sqrt(16)

    expected = torch.softmax((q @ key_all.transpose(-1, -2)) * scale + mask, dim=-1) @ value_all
    actual = cache.query(q, scale=scale, attention_mask=mask).output
    return compare("masked_attention_cache", actual, expected, atol=args.atol, rtol=args.rtol)


def test_batched_masked_attention_query(
    args,
    trusted_device,
    trusted_dtype,
    untrusted_device,
    untrusted_dtype,
) -> CheckResult:
    caches = []
    queries = []
    expected_outputs = []
    for index, tokens in enumerate((5, 7, 6)):
        cache = MaskedAttentionCache(
            dim=16,
            key_rank=args.rank,
            query_rank=args.rank,
            prob_rank=args.rank,
            value_rank=args.rank,
            dtype=untrusted_dtype,
            trusted_device=trusted_device,
            untrusted_device=untrusted_device,
            trusted_dtype=trusted_dtype,
            mask_scale=args.mask_scale,
        )
        keys = torch.randn(tokens, 16, device=trusted_device, dtype=trusted_dtype)
        values = torch.randn(tokens, 16, device=trusted_device, dtype=trusted_dtype)
        query = torch.randn(1 + (index % 2), 16, device=trusted_device, dtype=trusted_dtype)
        cache.append(keys, values)
        caches.append(cache)
        queries.append(query)
        expected_outputs.append(cache.baseline_query(query))

    actual_outputs = [result.output for result in batched_masked_attention_query(caches, queries)]
    actual = torch.cat(actual_outputs, dim=0)
    expected = torch.cat(expected_outputs, dim=0)
    return compare("batched_masked_attention_query", actual, expected, atol=args.atol, rtol=args.rtol)


def test_replace_llama_linears(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype) -> CheckResult:
    module = nn.Module()
    module.q_proj = nn.Linear(16, 12).to(device=trusted_device, dtype=trusted_dtype)
    x = torch.randn(2, 3, 16, device=trusted_device, dtype=trusted_dtype)
    expected = module.q_proj(x)
    report = replace_llama_linears(
        module,
        include_names=("q_proj",),
        mask_scale=args.mask_scale,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
        trusted_dtype=trusted_dtype,
    )
    if report.replaced != 1:
        raise AssertionError(f"Expected one Linear replacement, got {report.replaced}.")
    actual = module.q_proj(x)
    return compare("replace_llama_linears", actual, expected, atol=args.atol, rtol=args.rtol)


def test_masked_llama_attention_prefill(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype) -> CheckResult:
    module = nn.Module()
    module.attn = FakeModernAttention().to(device=trusted_device, dtype=trusted_dtype)
    hidden = torch.randn(2, 5, 12, device=trusted_device, dtype=trusted_dtype)
    mask = causal_mask(5, 5, device=trusted_device, dtype=trusted_dtype).view(1, 1, 5, 5).expand(2, 1, 5, 5).clone()
    mask[1, :, :, 4] = -1.0e4
    expected, expected_weights = llama_attention_baseline(module.attn, hidden, attention_mask=mask)

    report = replace_llama_attentions(
        module,
        key_rank=args.rank,
        query_rank=args.rank,
        prob_rank=args.rank,
        value_rank=args.rank,
        mask_scale=args.mask_scale,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
        trusted_dtype=trusted_dtype,
    )
    if report.replaced != 1:
        raise AssertionError(f"Expected one attention replacement, got {report.replaced}.")
    actual, actual_weights = module.attn(hidden, attention_mask=mask, output_attentions=True)

    output_result = compare("masked_llama_attention_prefill", actual, expected, atol=args.atol, rtol=args.rtol)
    weights_result = compare(
        "masked_llama_attention_weights",
        actual_weights,
        expected_weights,
        atol=args.atol,
        rtol=args.rtol,
    )
    return CheckResult(
        name="masked_llama_attention_prefill+weights",
        passed=output_result.passed and weights_result.passed,
        max_abs=max(output_result.max_abs, weights_result.max_abs),
        mean_abs=max(output_result.mean_abs, weights_result.mean_abs),
    )


def test_masked_llama_attention_decode_cache(
    args,
    trusted_device,
    trusted_dtype,
    untrusted_device,
    untrusted_dtype,
) -> CheckResult:
    module = nn.Module()
    module.attn = FakeLegacyAttention(hidden_size=8, num_heads=2, num_key_value_heads=1).to(
        device=trusted_device,
        dtype=trusted_dtype,
    )
    prefix = torch.randn(1, 3, 8, device=trusted_device, dtype=trusted_dtype)
    next_tokens = torch.randn(1, 2, 8, device=trusted_device, dtype=trusted_dtype)
    expected_prefix, _ = llama_attention_baseline(module.attn, prefix)
    expected_next, _ = llama_attention_baseline(
        module.attn,
        next_tokens,
        key_value_hidden=torch.cat([prefix, next_tokens], dim=1),
    )

    report = replace_llama_attentions(
        module,
        key_rank=args.rank,
        query_rank=args.rank,
        prob_rank=args.rank,
        value_rank=args.rank,
        mask_scale=args.mask_scale,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
        trusted_dtype=trusted_dtype,
    )
    if report.replaced != 1:
        raise AssertionError(f"Expected one attention replacement, got {report.replaced}.")

    actual_prefix, _, cache = module.attn(prefix, output_attentions=True, use_cache=True)
    actual_next, _, cache = module.attn(next_tokens, past_key_value=cache, output_attentions=True, use_cache=True)
    if cache is None:
        raise AssertionError("Expected legacy attention patch to return a cache.")

    prefix_result = compare("masked_llama_decode_prefill", actual_prefix, expected_prefix, atol=args.atol, rtol=args.rtol)
    next_result = compare("masked_llama_decode_step", actual_next, expected_next, atol=args.atol, rtol=args.rtol)
    return CheckResult(
        name="masked_llama_decode_cache",
        passed=prefix_result.passed and next_result.passed,
        max_abs=max(prefix_result.max_abs, next_result.max_abs),
        mean_abs=max(prefix_result.mean_abs, next_result.mean_abs),
    )


def main() -> None:
    args = parse_args()
    trusted_device = torch.device(args.trusted_device)
    untrusted_device = torch.device(args.untrusted_device)
    validate_device(trusted_device)
    validate_device(untrusted_device)
    trusted_dtype = DTYPES[args.trusted_dtype]
    untrusted_dtype = DTYPES[args.untrusted_dtype]
    if args.atol is None:
        args.atol = 1.0e-4 if untrusted_dtype == torch.float32 else 5.0e-2
    if args.rtol is None:
        args.rtol = 1.0e-4 if untrusted_dtype == torch.float32 else 5.0e-2

    torch.manual_seed(args.seed)
    print(
        "devices/dtypes: "
        f"trusted={trusted_device}/{trusted_dtype} "
        f"untrusted={untrusted_device}/{untrusted_dtype} "
        f"atol={args.atol} rtol={args.rtol}"
    )

    checks = [
        test_masked_linear,
        test_masked_qk,
        test_masked_pv,
        test_masked_kv_cache,
        test_masked_attention_cache,
        test_batched_masked_attention_query,
        test_replace_llama_linears,
        test_masked_llama_attention_prefill,
        test_masked_llama_attention_decode_cache,
    ]
    results = [
        check(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype)
        for check in checks
    ]
    failures = [result for result in results if not result.passed]
    if failures:
        names = ", ".join(result.name for result in failures)
        raise SystemExit(f"Correctness verification failed: {names}")
    print(f"\nAll {len(results)} correctness checks passed.")


if __name__ == "__main__":
    main()
