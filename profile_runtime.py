"""Profile runtime hotspots in the TEE-GPU masked matmul prototype.

The script uses random tensors and does not need a Hugging Face model. It times
plain PyTorch baselines and the masked/offloaded paths, then prints latency,
throughput-oriented numbers, and optional CUDA peak memory.

Examples:
    python profile_runtime.py --target all --untrusted-device cpu
    python profile_runtime.py --target attention --trusted-device cpu --untrusted-device cuda
    python profile_runtime.py --target llama --torch-profiler --trace-file trace.json
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable


try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:
    raise SystemExit("Install dependencies first: pip install -r requirements.txt") from exc

from tee_gpu_demo.llama_patch import replace_llama_attentions
from tee_gpu_demo.continuous_batching import ContinuousBatchingEngine, ContinuousRequest
from tee_gpu_demo.masked_ops import MaskedAttentionCache, MaskedKVCache, masked_linear, masked_pv, masked_qk


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class BenchResult:
    name: str
    ms: float
    peak_mb: float | None = None


class FakeModernAttention(nn.Module):
    """Small LlamaAttention-shaped module for profiling patched forward."""

    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_heads % num_key_value_heads != 0:
            raise ValueError("num_heads must be divisible by num_key_value_heads")

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
        raise NotImplementedError("replace_llama_attentions() should patch this method.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=("all", "linear", "qk", "pv", "kv", "attention", "llama", "continuous"),
        default="all",
    )
    parser.add_argument("--trusted-device", default="cpu")
    parser.add_argument("--untrusted-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trusted-dtype", choices=DTYPES, default="float32")
    parser.add_argument("--untrusted-dtype", choices=DTYPES, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--mask-scale", type=float, default=0.02)

    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--linear-out", type=int, default=512)

    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Capture a PyTorch profiler trace for the selected target.",
    )
    parser.add_argument("--trace-file", default="profile_trace.json")
    return parser.parse_args()


def validate_device(device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA was requested for {device}, but torch.cuda.is_available() is false.")


def sync(*devices: torch.device) -> None:
    if any(device.type == "cuda" for device in devices):
        torch.cuda.synchronize()


def reset_peak_memory(*devices: torch.device) -> None:
    if any(device.type == "cuda" for device in devices):
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(*devices: torch.device) -> float | None:
    cuda_devices = [device for device in devices if device.type == "cuda"]
    if not cuda_devices:
        return None
    return max(torch.cuda.max_memory_allocated(device) for device in cuda_devices) / (1024 * 1024)


def time_ms(
    fn: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
    trusted_device: torch.device,
    untrusted_device: torch.device,
) -> BenchResult:
    for _ in range(warmup):
        fn()
    sync(trusted_device, untrusted_device)
    reset_peak_memory(trusted_device, untrusted_device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync(trusted_device, untrusted_device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats
    return BenchResult(name="", ms=elapsed_ms, peak_mb=peak_memory_mb(trusted_device, untrusted_device))


def profiler_context(enabled: bool, trace_file: str):
    if not enabled:
        return nullcontext(None)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )


def maybe_export_trace(prof, trace_file: str) -> None:
    if prof is not None:
        prof.export_chrome_trace(trace_file)
        print(f"trace_file={trace_file}")


def print_result(name: str, result: BenchResult, *, work_units: float | None = None, unit: str = "items") -> None:
    throughput = ""
    if work_units is not None and result.ms > 0:
        throughput = f" throughput={work_units / (result.ms / 1000.0):.3f} {unit}/s"
    memory = "" if result.peak_mb is None else f" peak_cuda_mb={result.peak_mb:.2f}"
    print(f"{name:28s} ms={result.ms:.3f}{throughput}{memory}")


def print_speedup(name: str, baseline: BenchResult, candidate: BenchResult) -> None:
    if candidate.ms <= 0:
        print(f"{name:28s} speedup=inf")
        return
    speedup = baseline.ms / candidate.ms
    overhead = (candidate.ms / baseline.ms - 1.0) * 100.0 if baseline.ms > 0 else float("inf")
    verdict = "faster" if speedup > 1.0 else "slower"
    print(f"{name:28s} speedup={speedup:.3f}x ({verdict}, overhead={overhead:+.1f}%)")


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


def run_with_optional_profiler(args: argparse.Namespace, fn: Callable[[], object]) -> None:
    with profiler_context(args.torch_profiler, args.trace_file) as prof:
        fn()
        if prof is not None:
            prof.step()
    maybe_export_trace(prof, args.trace_file)


def profile_linear(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype) -> None:
    x = torch.randn(args.batch, args.seq, args.hidden_size, device=trusted_device, dtype=trusted_dtype)
    weight = torch.randn(args.linear_out, args.hidden_size, device=trusted_device, dtype=trusted_dtype)
    bias = torch.randn(args.linear_out, device=trusted_device, dtype=trusted_dtype)

    def baseline():
        return torch.nn.functional.linear(x, weight, bias)

    def masked():
        return masked_linear(
            x,
            weight,
            bias,
            mask_scale=args.mask_scale,
            trusted_device=trusted_device,
            untrusted_device=untrusted_device,
            trusted_dtype=trusted_dtype,
            untrusted_dtype=untrusted_dtype,
        ).output

    print("\n[linear]")
    baseline_result = time_ms(
        baseline,
        warmup=args.warmup,
        repeats=args.repeats,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
    )
    masked_result = time_ms(
        masked,
        warmup=args.warmup,
        repeats=args.repeats,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
    )
    print_result(
        "baseline linear",
        baseline_result,
        work_units=args.batch * args.seq,
        unit="tokens",
    )
    print_result(
        "masked linear",
        masked_result,
        work_units=args.batch * args.seq,
        unit="tokens",
    )
    print_speedup("linear masked/baseline", baseline_result, masked_result)
    run_with_optional_profiler(args, masked)


def profile_qk(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype) -> None:
    q = torch.randn(args.batch, args.queries, args.hidden_size, device=trusted_device, dtype=trusted_dtype)
    k = torch.randn(args.batch, args.tokens, args.hidden_size, device=trusted_device, dtype=trusted_dtype)

    def baseline():
        return q @ k.transpose(-1, -2)

    def masked():
        return masked_qk(
            q,
            k,
            rank=args.rank,
            mask_scale=args.mask_scale,
            trusted_device=trusted_device,
            untrusted_device=untrusted_device,
            trusted_dtype=trusted_dtype,
            untrusted_dtype=untrusted_dtype,
        ).output

    print("\n[qk]")
    baseline_result = time_ms(
        baseline,
        warmup=args.warmup,
        repeats=args.repeats,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
    )
    masked_result = time_ms(
        masked,
        warmup=args.warmup,
        repeats=args.repeats,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
    )
    print_result(
        "baseline qk",
        baseline_result,
        work_units=args.batch * args.queries * args.tokens,
        unit="scores",
    )
    print_result(
        "masked qk",
        masked_result,
        work_units=args.batch * args.queries * args.tokens,
        unit="scores",
    )
    print_speedup("qk masked/baseline", baseline_result, masked_result)
    run_with_optional_profiler(args, masked)


def profile_pv(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype) -> None:
    p = torch.softmax(torch.randn(args.batch, args.queries, args.tokens, device=trusted_device, dtype=trusted_dtype), dim=-1)
    v = torch.randn(args.batch, args.tokens, args.hidden_size, device=trusted_device, dtype=trusted_dtype)

    def baseline():
        return p @ v

    def masked():
        return masked_pv(
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

    print("\n[pv]")
    baseline_result = time_ms(
        baseline,
        warmup=args.warmup,
        repeats=args.repeats,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
    )
    masked_result = time_ms(
        masked,
        warmup=args.warmup,
        repeats=args.repeats,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
    )
    print_result(
        "baseline pv",
        baseline_result,
        work_units=args.batch * args.queries,
        unit="queries",
    )
    print_result(
        "masked pv",
        masked_result,
        work_units=args.batch * args.queries,
        unit="queries",
    )
    print_speedup("pv masked/baseline", baseline_result, masked_result)
    run_with_optional_profiler(args, masked)


def build_kv_cache(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype) -> MaskedKVCache:
    cache = MaskedKVCache(
        dim=args.hidden_size,
        key_rank=args.rank,
        query_rank=args.rank,
        dtype=untrusted_dtype,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
        trusted_dtype=trusted_dtype,
        mask_scale=args.mask_scale,
    )
    remaining = args.tokens
    while remaining > 0:
        size = min(args.chunk, remaining)
        cache.append(torch.randn(size, args.hidden_size, device=trusted_device, dtype=trusted_dtype))
        remaining -= size
    return cache


def profile_kv(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype) -> None:
    def append_only():
        build_kv_cache(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype)

    cache = build_kv_cache(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype)
    q = torch.randn(args.queries, args.hidden_size, device=trusted_device, dtype=trusted_dtype)

    def baseline_query():
        return cache.baseline_query(q)

    def masked_query():
        return cache.query(q).output

    print("\n[kv cache]")
    append_result = time_ms(
        append_only,
        warmup=max(1, args.warmup // 2),
        repeats=max(1, args.repeats // 2),
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
    )
    baseline_result = time_ms(
        baseline_query,
        warmup=args.warmup,
        repeats=args.repeats,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
    )
    masked_result = time_ms(
        masked_query,
        warmup=args.warmup,
        repeats=args.repeats,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
    )
    print_result(
        "masked kv append",
        append_result,
        work_units=args.tokens,
        unit="tokens",
    )
    print_result(
        "baseline kv query",
        baseline_result,
        work_units=args.queries * args.tokens,
        unit="scores",
    )
    print_result(
        "masked kv query",
        masked_result,
        work_units=args.queries * args.tokens,
        unit="scores",
    )
    print_speedup("kv query masked/baseline", baseline_result, masked_result)
    run_with_optional_profiler(args, masked_query)


def build_attention_cache(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype) -> MaskedAttentionCache:
    cache = MaskedAttentionCache(
        dim=args.hidden_size,
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
    remaining = args.tokens
    while remaining > 0:
        size = min(args.chunk, remaining)
        keys = torch.randn(size, args.hidden_size, device=trusted_device, dtype=trusted_dtype)
        values = torch.randn(size, args.hidden_size, device=trusted_device, dtype=trusted_dtype)
        cache.append(keys, values)
        remaining -= size
    return cache


def profile_attention(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype) -> None:
    def append_only():
        build_attention_cache(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype)

    cache = build_attention_cache(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype)
    q = torch.randn(args.queries, args.hidden_size, device=trusted_device, dtype=trusted_dtype)
    mask = causal_mask(args.queries, args.tokens, device=trusted_device, dtype=trusted_dtype)

    def baseline_query():
        return cache.baseline_query(q)

    def masked_query():
        return cache.query(q, attention_mask=mask).output

    print("\n[attention cache]")
    append_result = time_ms(
        append_only,
        warmup=max(1, args.warmup // 2),
        repeats=max(1, args.repeats // 2),
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
    )
    baseline_result = time_ms(
        baseline_query,
        warmup=args.warmup,
        repeats=args.repeats,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
    )
    masked_result = time_ms(
        masked_query,
        warmup=args.warmup,
        repeats=args.repeats,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
    )
    print_result(
        "masked attn append",
        append_result,
        work_units=args.tokens,
        unit="tokens",
    )
    print_result(
        "baseline attn query",
        baseline_result,
        work_units=args.queries,
        unit="queries",
    )
    print_result(
        "masked attn query",
        masked_result,
        work_units=args.queries,
        unit="queries",
    )
    print_speedup("attn query masked/baseline", baseline_result, masked_result)
    run_with_optional_profiler(args, masked_query)


def profile_llama(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype) -> None:
    module = nn.Module()
    module.attn = FakeModernAttention(args.hidden_size, args.heads, args.kv_heads).to(
        device=trusted_device,
        dtype=trusted_dtype,
    )
    hidden = torch.randn(args.batch, args.seq, args.hidden_size, device=trusted_device, dtype=trusted_dtype)
    attention_mask = causal_mask(args.seq, args.seq, device=trusted_device, dtype=trusted_dtype).view(1, 1, args.seq, args.seq)
    attention_mask = attention_mask.expand(args.batch, 1, args.seq, args.seq).clone()

    def baseline_prefill():
        attn = module.attn
        batch_size, seq_len, _ = hidden.shape
        head_dim = attn.head_dim
        q = attn.q_proj(hidden).view(batch_size, seq_len, args.heads, head_dim).transpose(1, 2)
        k = attn.k_proj(hidden).view(batch_size, seq_len, args.kv_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(hidden).view(batch_size, seq_len, args.kv_heads, head_dim).transpose(1, 2)
        k = repeat_kv(k, attn.num_key_value_groups)
        v = repeat_kv(v, attn.num_key_value_groups)
        scores = (q @ k.transpose(-1, -2)) * attn.scaling
        scores = scores + attention_mask.to(device=scores.device, dtype=scores.dtype)
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        output = probs @ v
        output = output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, args.heads * head_dim)
        return attn.o_proj(output)

    baseline_result = time_ms(
        baseline_prefill,
        warmup=args.warmup,
        repeats=args.repeats,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
    )

    replace_llama_attentions(
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

    def masked_prefill():
        return module.attn(hidden, attention_mask=attention_mask, output_attentions=False)[0]

    print("\n[patched llama attention]")
    print_result(
        "baseline llama prefill",
        baseline_result,
        work_units=args.batch * args.seq,
        unit="tokens",
    )
    masked_result = time_ms(
        masked_prefill,
        warmup=args.warmup,
        repeats=args.repeats,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
    )
    print_result(
        "masked llama prefill",
        masked_result,
        work_units=args.batch * args.seq,
        unit="tokens",
    )
    print_speedup("llama masked/baseline", baseline_result, masked_result)
    run_with_optional_profiler(args, masked_prefill)


def build_continuous_requests(args, trusted_device, trusted_dtype) -> list[ContinuousRequest]:
    requests = []
    for index in range(args.batch):
        prompt_len = args.tokens + (index % 3) * max(1, args.chunk // 2)
        decode_len = args.queries + (index % 2)
        requests.append(
            ContinuousRequest(
                request_id=f"req-{index}",
                prompt_keys=torch.randn(prompt_len, args.hidden_size, device=trusted_device, dtype=trusted_dtype),
                prompt_values=torch.randn(prompt_len, args.hidden_size, device=trusted_device, dtype=trusted_dtype),
                decode_queries=torch.randn(decode_len, args.hidden_size, device=trusted_device, dtype=trusted_dtype),
                decode_keys=torch.randn(decode_len, args.hidden_size, device=trusted_device, dtype=trusted_dtype),
                decode_values=torch.randn(decode_len, args.hidden_size, device=trusted_device, dtype=trusted_dtype),
                arrival_step=index,
            )
        )
    return requests


def profile_continuous(args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype) -> None:
    requests = build_continuous_requests(args, trusted_device, trusted_dtype)
    decoded_tokens = sum(request.decode_len for request in requests)

    def run_engine():
        engine = ContinuousBatchingEngine(
            dim=args.hidden_size,
            max_batch_size=max(1, min(args.batch, args.heads)),
            max_active_requests=args.batch,
            prefill_chunk=args.chunk,
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
        engine.submit_many(requests)
        return engine.run()

    print("\n[continuous batching]")
    result = time_ms(
        run_engine,
        warmup=max(1, args.warmup // 2),
        repeats=max(1, args.repeats // 2),
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
    )
    print_result(
        "continuous engine",
        result,
        work_units=decoded_tokens,
        unit="decoded_tokens",
    )
    run_with_optional_profiler(args, run_engine)


def main() -> None:
    args = parse_args()
    trusted_device = torch.device(args.trusted_device)
    untrusted_device = torch.device(args.untrusted_device)
    validate_device(trusted_device)
    validate_device(untrusted_device)

    trusted_dtype = DTYPES[args.trusted_dtype]
    if args.untrusted_dtype is None:
        untrusted_dtype = torch.float16 if untrusted_device.type == "cuda" else torch.float32
    else:
        untrusted_dtype = DTYPES[args.untrusted_dtype]

    torch.manual_seed(args.seed)
    print(
        "profile config: "
        f"target={args.target} trusted={trusted_device}/{trusted_dtype} "
        f"untrusted={untrusted_device}/{untrusted_dtype} "
        f"warmup={args.warmup} repeats={args.repeats}"
    )
    print(
        "sizes: "
        f"batch={args.batch} seq={args.seq} tokens={args.tokens} queries={args.queries} "
        f"hidden={args.hidden_size} heads={args.heads} kv_heads={args.kv_heads} rank={args.rank}"
    )

    targets = {
        "linear": profile_linear,
        "qk": profile_qk,
        "pv": profile_pv,
        "kv": profile_kv,
        "attention": profile_attention,
        "llama": profile_llama,
        "continuous": profile_continuous,
    }
    selected = targets.keys() if args.target == "all" else (args.target,)
    for target in selected:
        targets[target](args, trusted_device, trusted_dtype, untrusted_device, untrusted_dtype)


if __name__ == "__main__":
    main()
