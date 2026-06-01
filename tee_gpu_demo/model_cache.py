"""Hugging Face model cache helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


HF_LLAMA_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_LOCAL_LLAMA_MODEL_DIR = Path.home() / "models" / "Llama-3.2-1B"
DEFAULT_LLAMA_MODEL = (
    str(DEFAULT_LOCAL_LLAMA_MODEL_DIR.resolve())
    if DEFAULT_LOCAL_LLAMA_MODEL_DIR.exists()
    else HF_LLAMA_MODEL
)
DEFAULT_CACHE_DIR = "./model_cache"


def resolve_model_name_or_path(model_name: str) -> str:
    """Return an absolute local path when model_name points at local files."""

    path = Path(model_name).expanduser()
    is_path_like = model_name.startswith(("~", "./", "../")) or path.is_absolute()
    if path.exists() or is_path_like:
        return str(path.resolve())
    return model_name


def cache_hf_model(
    model_name: str = DEFAULT_LLAMA_MODEL,
    *,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    revision: Optional[str] = None,
    local_files_only: bool = False,
) -> Path:
    """Download a Hugging Face model snapshot into the local cache."""

    resolved_model = resolve_model_name_or_path(model_name)
    local_path = Path(resolved_model)
    if local_path.exists():
        return local_path

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub first: pip install huggingface_hub") from exc

    # snapshot_download returns the resolved immutable snapshot directory.
    path = snapshot_download(
        repo_id=resolved_model,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )
    return Path(path)
