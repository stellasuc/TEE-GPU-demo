"""Demo for masked QK + TEE softmax + masked PV attention."""

from __future__ import annotations

import argparse


def default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=4)
    parser.add_argument("--key-rank", type=int, default=4)
    parser.add_argument("--query-rank", type=int, default=4)
    parser.add_argument("--prob-rank", type=int, default=4)
    parser.add_argument("--value-rank", type=int, default=4)
    parser.add_argument("--trusted-device", default="cpu")
    parser.add_argument("--untrusted-device", default=default_device())
    parser.add_argument("--device", default=None, help="Alias for --untrusted-device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch

        from tee_gpu_demo.masked_ops import MaskedAttentionCache
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from exc

    trusted_device = torch.device(args.trusted_device)
    untrusted_device = torch.device(args.device or args.untrusted_device)
    untrusted_dtype = torch.float16 if untrusted_device.type == "cuda" else torch.float32
    trusted_dtype = torch.float32

    cache = MaskedAttentionCache(
        dim=args.dim,
        key_rank=args.key_rank,
        query_rank=args.query_rank,
        prob_rank=args.prob_rank,
        value_rank=args.value_rank,
        dtype=untrusted_dtype,
        trusted_dtype=trusted_dtype,
        trusted_device=trusted_device,
        untrusted_device=untrusted_device,
    )

    remaining = args.tokens
    while remaining > 0:
        size = min(args.chunk, remaining)
        keys = torch.randn(size, args.dim, device=trusted_device, dtype=trusted_dtype)
        values = torch.randn(size, args.dim, device=trusted_device, dtype=trusted_dtype)
        cache.append(keys, values)
        remaining -= size

    q = torch.randn(args.queries, args.dim, device=trusted_device, dtype=trusted_dtype)
    baseline = cache.baseline_query(q)
    masked = cache.query(q).output
    err = (baseline.float() - masked.float()).abs()

    print(
        f"trusted_device={trusted_device} untrusted_device={untrusted_device} "
        f"untrusted_dtype={untrusted_dtype} tokens={cache.masked_keys.shape[0]} dim={args.dim}"
    )
    print(
        "ranks="
        f"key:{args.key_rank} query:{args.query_rank} prob:{args.prob_rank} value:{args.value_rank}"
    )
    print(f"max_abs_error={err.max().item():.6g}")
    print(f"mean_abs_error={err.mean().item():.6g}")


if __name__ == "__main__":
    main()
