"""Pre-cache a Llama model from Hugging Face.

Example:
    python cache_llama.py --cache-dir ./model_cache
"""

from __future__ import annotations

import argparse

from tee_gpu_demo.model_cache import DEFAULT_CACHE_DIR, DEFAULT_LLAMA_MODEL, cache_hf_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_LLAMA_MODEL)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download; only resolve an already cached snapshot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = cache_hf_model(
        args.model,
        cache_dir=args.cache_dir,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    print(f"model={args.model}")
    print(f"cache_dir={args.cache_dir}")
    print(f"snapshot_path={path}")


if __name__ == "__main__":
    main()
