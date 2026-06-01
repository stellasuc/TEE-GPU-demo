"""Quick masked matmul benchmark."""

from __future__ import annotations

import argparse
import time

import torch

from tee_gpu_demo.masked_ops import masked_qk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_ms(fn, repeats: int, device: torch.device) -> float:
    sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync(device)
    return (time.perf_counter() - start) * 1000.0 / repeats


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    q = torch.randn(args.m, args.d, device=device, dtype=dtype)
    k = torch.randn(args.n, args.d, device=device, dtype=dtype)

    baseline = lambda: q @ k.T
    masked = lambda: masked_qk(q, k, rank=args.rank).output

    base_out = baseline()
    masked_out = masked()
    err = (base_out.float() - masked_out.float()).abs()

    print(f"device={device} dtype={dtype} shape=({args.m},{args.n},{args.d}) rank={args.rank}")
    print(f"baseline_ms={time_ms(baseline, args.repeats, device):.3f}")
    print(f"masked_ms={time_ms(masked, args.repeats, device):.3f}")
    print(f"max_abs_error={err.max().item():.6g}")
    print(f"mean_abs_error={err.mean().item():.6g}")


if __name__ == "__main__":
    main()
