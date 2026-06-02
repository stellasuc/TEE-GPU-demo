"""Run Llama inference with masked Linear layers.

Example:
    python demo_llama.py --prompt "请解释什么是矩阵乘法掩码卸载"
    python demo_llama.py --interactive
"""

from __future__ import annotations

import argparse

from tee_gpu_demo.model_cache import DEFAULT_CACHE_DIR, DEFAULT_LLAMA_MODEL, cache_hf_model, resolve_model_name_or_path


DEFAULT_LLAMA_LINEAR_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_LLAMA_MODEL)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Hugging Face cache dir.")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download/cache the model snapshot and exit without loading it.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force local-only cache access. Inference already uses this by default.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow network downloads while loading for inference.",
    )
    parser.add_argument("--prompt", default=None, help="Run one prompt. If omitted, ask once.")
    parser.add_argument("--interactive", action="store_true", help="Keep reading prompts from stdin.")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--mask-scale", type=float, default=0.02)
    parser.add_argument("--trusted-device", default="cpu", help="Model and trusted TEE-side device.")
    parser.add_argument(
        "--untrusted-device",
        default=default_device(),
        help="Device for untrusted masked matmul.",
    )
    parser.add_argument(
        "--compat-return-to-model-device",
        action="store_true",
        help="Compatibility mode for returning each masked Linear output to its input device.",
    )
    parser.add_argument(
        "--layers",
        default=",".join(DEFAULT_LLAMA_LINEAR_NAMES),
        help="Comma-separated Linear names to wrap, e.g. q_proj,k_proj,v_proj,o_proj",
    )
    parser.add_argument(
        "--fp32-correction",
        action="store_true",
        help="Compute correction terms in fp32 before casting back.",
    )
    parser.add_argument(
        "--disable-masked-attention",
        action="store_true",
        help="Only patch Linear layers; leave HuggingFace LlamaAttention.forward unchanged.",
    )
    parser.add_argument("--attention-key-rank", type=int, default=4)
    parser.add_argument("--attention-query-rank", type=int, default=4)
    parser.add_argument("--attention-prob-rank", type=int, default=4)
    parser.add_argument("--attention-value-rank", type=int, default=4)
    parser.add_argument(
        "--trusted-pv",
        action="store_true",
        help="Only offload masked QK; keep softmax P @ V on the trusted side.",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run one unmasked forward pass and print logits error. One-shot mode only.",
    )
    return parser.parse_args()


def load_model(
    model_name: str,
    dtype: torch.dtype,
    device: torch.device,
    *,
    cache_dir: str | None,
    revision: str | None,
    local_files_only: bool,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = resolve_model_name_or_path(model_name)
    # Transformers receives the same cache policy for tokenizer and weights.
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )
    load_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
        "cache_dir": cache_dir,
        "revision": revision,
        "local_files_only": local_files_only,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).to(device)
    except TypeError:
        load_kwargs.pop("attn_implementation")
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).to(device)
    model.eval()
    return tokenizer, model


def encode_prompt(tokenizer, prompt: str, device: torch.device, system_prompt: str | None):
    import torch

    if getattr(tokenizer, "chat_template", None):
        # Prefer the model's chat template when it exists, as Llama Instruct expects it.
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    return tokenizer(prompt, return_tensors="pt").to(device)


def generate_once(model, tokenizer, prompt: str, args: argparse.Namespace, device: torch.device) -> str:
    import torch

    inputs = encode_prompt(tokenizer, prompt, device, args.system_prompt)
    input_len = inputs["input_ids"].shape[-1]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )

    new_tokens = generated[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    if args.interactive and args.compare_baseline:
        raise SystemExit("--compare-baseline is only supported for one-shot prompts.")

    if args.download_only:
        # Explicit cache warm-up path; inference does not need to load the model.
        path = cache_hf_model(
            args.model,
            cache_dir=args.cache_dir,
            revision=args.revision,
            local_files_only=args.local_files_only,
        )
        print(f"model={args.model}")
        print(f"snapshot_path={path}")
        return

    try:
        import torch
        import transformers  # noqa: F401
        from tee_gpu_demo.llama_patch import replace_llama_attentions, replace_llama_linears
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from exc

    trusted_device = torch.device(args.trusted_device)
    dtype = torch.float16 if trusted_device.type == "cuda" else torch.float32

    # Inference is offline by default: only --allow-download permits network access.
    local_files_only = args.local_files_only or not args.allow_download
    try:
        tokenizer, model = load_model(
            args.model,
            dtype,
            trusted_device,
            cache_dir=args.cache_dir,
            revision=args.revision,
            local_files_only=local_files_only,
        )
    except OSError as exc:
        if local_files_only:
            raise SystemExit(
                "Local model snapshot was not found. "
                f"Run `python cache_llama.py --model {args.model} --cache-dir {args.cache_dir}` first, "
                "or pass `--allow-download` if this machine may use the network."
            ) from exc
        raise

    with torch.inference_mode():
        baseline = None
        if args.compare_baseline:
            # Keep one clean forward pass before patching, only for correctness checks.
            prompt = args.prompt or input("Prompt: ").strip()
            if not prompt:
                raise SystemExit("Prompt is empty.")
            inputs = encode_prompt(tokenizer, prompt, trusted_device, args.system_prompt)
            baseline = model(**inputs).logits[:, -1, :].float().cpu()
        else:
            prompt = args.prompt

    include_names = tuple(name.strip() for name in args.layers.split(",") if name.strip())
    # Patch selected Llama Linear layers before any generated-token forward passes.
    trusted_dtype = torch.float32 if trusted_device.type == "cpu" else dtype
    report = replace_llama_linears(
        model,
        include_names=include_names,
        mask_scale=args.mask_scale,
        correction_dtype=torch.float32 if args.fp32_correction else None,
        trusted_device=trusted_device,
        untrusted_device=args.untrusted_device,
        trusted_dtype=trusted_dtype,
        return_to_input_device=args.compat_return_to_model_device,
    )
    print(f"patched_layers={report.replaced}")
    if not args.disable_masked_attention:
        attention_report = replace_llama_attentions(
            model,
            key_rank=args.attention_key_rank,
            query_rank=args.attention_query_rank,
            prob_rank=args.attention_prob_rank,
            value_rank=args.attention_value_rank,
            mask_scale=args.mask_scale,
            trusted_device=trusted_device,
            untrusted_device=args.untrusted_device,
            trusted_dtype=trusted_dtype,
            return_to_input_device=args.compat_return_to_model_device,
            offload_pv=not args.trusted_pv,
        )
        print(f"patched_attentions={attention_report.replaced}")

    if args.interactive:
        # Reuse one patched model for all prompts in this terminal session.
        print("Enter an empty prompt, Ctrl-D, or Ctrl-C to exit.")
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt:
                break
            print(generate_once(model, tokenizer, prompt, args, trusted_device))
        return

    prompt = prompt or input("Prompt: ").strip()
    if not prompt:
        raise SystemExit("Prompt is empty.")

    if baseline is not None:
        inputs = encode_prompt(tokenizer, prompt, trusted_device, args.system_prompt)
        with torch.inference_mode():
            masked = model(**inputs).logits[:, -1, :].float().cpu()
        # The masked path should match the unmasked logits up to floating-point error.
        diff = (baseline - masked).abs()
        print(f"max_abs_error={diff.max().item():.6g}")
        print(f"mean_abs_error={diff.mean().item():.6g}")

    print(generate_once(model, tokenizer, prompt, args, trusted_device))


if __name__ == "__main__":
    main()
