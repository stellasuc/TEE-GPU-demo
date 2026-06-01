"""Hugging Face model cache helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


DEFAULT_LLAMA_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_CACHE_DIR = "./model_cache"


def cache_hf_model(
    model_name: str = DEFAULT_LLAMA_MODEL,
    *,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    revision: Optional[str] = None,
    local_files_only: bool = False,
) -> Path:
    """Download a Hugging Face model snapshot into the local cache."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub first: pip install huggingface_hub") from exc

    # snapshot_download returns the resolved immutable snapshot directory.
    path = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )
    return Path(path)
