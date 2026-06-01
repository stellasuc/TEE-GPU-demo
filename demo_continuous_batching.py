"""Demo for scheduler-level continuous batching with masked attention caches."""

from __future__ import annotations

import argparse
import math
import time


def default_untrusted_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--max-batch-size", type=int, default=3)
    parser.add_argument("--max-active-requests", type=int, default=None)
    parser.add_argument("--arrival-gap", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--mask-scale", type=float, default=0.02)
    parser.add_argument("--trusted-device", default="cpu")
    parser.add_argument("--untrusted-device", default=default_untrusted_device())
    parser.add_argument("--untrusted-dtype", choices=("float32", "float16", "bfloat16"), default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeline", action="store_true")
    return parser.parse_args()


def sync(device) -> None:
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    try:
        import torch

        from tee_gpu_demo.continuous_batching import (
            ContinuousBatchingEngine,
            ContinuousRequest,
            plain_attention_reference,
        )
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from exc

    torch.manual_seed(args.seed)
    trusted_device = torch.device(args.trusted_device)
    untrusted_device = torch.device(args.untrusted_device)
    if untrusted_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")

    trusted_dtype = torch.float32
    if args.untrusted_dtype is None:
        untrusted_dtype = torch.float16 if untrusted_device.type == "cuda" else torch.float32
    else:
        untrusted_dtype = getattr(torch, args.untrusted_dtype)

    requests = []
    for index in range(args.requests):
        # Slightly vary prompt/decode sizes so the scheduler has uneven work.
        prompt_len = args.prompt_tokens + (index % 3) * max(1, args.chunk // 2)
        decode_len = args.decode_tokens + (index % 2)
        request = ContinuousRequest(
            request_id=f"req-{index}",
            prompt_keys=torch.randn(prompt_len, args.dim, device=trusted_device, dtype=trusted_dtype),
            prompt_values=torch.randn(prompt_len, args.dim, device=trusted_device, dtype=trusted_dtype),
            decode_queries=torch.randn(decode_len, args.dim, device=trusted_device, dtype=trusted_dtype),
            decode_keys=torch.randn(decode_len, args.dim, device=trusted_device, dtype=trusted_dtype),
            decode_values=torch.randn(decode_len, args.dim, device=trusted_device, dtype=trusted_dtype),
            arrival_step=index * args.arrival_gap,
        )
        requests.append(request)

    engine = ContinuousBatchingEngine(
        dim=args.dim,
        max_batch_size=args.max_batch_size,
        max_active_requests=args.max_active_requests,
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

    sync(untrusted_device)
    start = time.perf_counter()
    result = engine.run()
    sync(untrusted_device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    max_abs = 0.0
    mean_abs_values = []
    for request in requests:
        expected = plain_attention_reference(request)
        actual = result.outputs[request.request_id]
        diff = (actual.float() - expected.float()).abs()
        max_abs = max(max_abs, float(diff.max().item()) if diff.numel() else 0.0)
        mean_abs_values.append(float(diff.mean().item()) if diff.numel() else 0.0)
    mean_abs = sum(mean_abs_values) / len(mean_abs_values)

    sequential_steps = sum(math.ceil(request.prompt_len / args.chunk) + request.decode_len for request in requests)
    step_reduction = sequential_steps / result.total_steps if result.total_steps else float("inf")
    decoded_tokens = sum(request.decode_len for request in requests)
    print(
        f"trusted_device={trusted_device} untrusted_device={untrusted_device} "
        f"untrusted_dtype={untrusted_dtype} requests={args.requests} dim={args.dim}"
    )
    print(
        f"max_batch_size={args.max_batch_size} max_active_requests={args.max_active_requests or args.max_batch_size} "
        f"prefill_chunk={args.chunk} rank={args.rank}"
    )
    print(
        f"continuous_steps={result.total_steps} sequential_request_steps={sequential_steps} "
        f"step_reduction={step_reduction:.3f}x"
    )
    print(
        f"decode_batches={result.decode_batches} decoded_tokens={decoded_tokens} "
        f"max_decode_batch={result.max_decode_batch}"
    )
    print(f"elapsed_ms={elapsed_ms:.3f} decoded_tokens_per_s={decoded_tokens / (elapsed_ms / 1000.0):.3f}")
    print(f"max_abs_error={max_abs:.6g}")
    print(f"mean_abs_error={mean_abs:.6g}")

    if args.timeline:
        print("\nTimeline")
        for step in result.steps:
            print(
                f"step={step.step_index:03d} "
                f"admit={list(step.admitted_ids)} "
                f"prefill={list(step.prefilled_ids)} "
                f"decode={list(step.decoded_ids)} "
                f"finish={list(step.finished_ids)} "
                f"active={list(step.active_ids)}"
            )


if __name__ == "__main__":
    main()
