"""Compare baseline vs masked Llama accuracy on multiple-choice datasets.

The evaluator scores each answer choice by conditional log-likelihood:

    score(choice) = log P(choice | prompt)

Baseline and masked runs share the same loaded model; the script patches Linear
layers in-place between the two runs so the comparison is tightly controlled.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

from demo_llama import load_model
from tee_gpu_demo.model_cache import DEFAULT_CACHE_DIR, DEFAULT_LLAMA_MODEL

try:
    import torch
except ImportError:
    torch = None


DEFAULT_DATASET_CACHE_DIR = "./dataset_cache"
SUPPORTED_TASKS = ("piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "boolq")
DEFAULT_LLAMA_LINEAR_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class EvalExample:
    prompt: str
    choices: List[str]
    label: int


@dataclass
class TaskResult:
    task: str
    mode: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def default_device() -> str:
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_LLAMA_MODEL)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--trusted-device", default="cpu", help="Model and trusted TEE-side device.")
    parser.add_argument("--untrusted-device", default=default_device())
    parser.add_argument(
        "--compat-return-to-model-device",
        action="store_true",
        help="Compatibility mode for returning each masked Linear output to its input device.",
    )
    parser.add_argument("--dataset-cache-dir", default=DEFAULT_DATASET_CACHE_DIR)
    parser.add_argument(
        "--tasks",
        default="piqa,arc_easy,boolq",
        help=f"Comma-separated tasks. Supported: {', '.join(SUPPORTED_TASKS)}",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=100, help="Examples per task; <=0 means all.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--mode", choices=("both", "baseline", "masked"), default="both")
    parser.add_argument("--mask-scale", type=float, default=0.02)
    parser.add_argument(
        "--layers",
        default=",".join(DEFAULT_LLAMA_LINEAR_NAMES),
        help="Comma-separated Linear names to wrap for the masked run.",
    )
    parser.add_argument("--fp32-correction", action="store_true")
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
        "--sum-logprob",
        action="store_true",
        help="Use summed log-probability instead of length-normalized score.",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow network downloads for the model. Default is local cache only.",
    )
    parser.add_argument(
        "--allow-dataset-download",
        action="store_true",
        help="Allow network downloads for datasets. Default is local cache only.",
    )
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_hf_dataset(task: str, split: str, cache_dir: str, allow_download: bool):
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets first: pip install datasets") from exc

    download_config = DownloadConfig(local_files_only=not allow_download)
    if task == "piqa":
        return load_dataset("piqa", split=split, cache_dir=cache_dir, download_config=download_config)
    if task == "arc_easy":
        return load_dataset("ai2_arc", "ARC-Easy", split=split, cache_dir=cache_dir, download_config=download_config)
    if task == "arc_challenge":
        return load_dataset(
            "ai2_arc",
            "ARC-Challenge",
            split=split,
            cache_dir=cache_dir,
            download_config=download_config,
        )
    if task == "hellaswag":
        return load_dataset("hellaswag", split=split, cache_dir=cache_dir, download_config=download_config)
    if task == "winogrande":
        return load_dataset(
            "winogrande",
            "winogrande_xl",
            split=split,
            cache_dir=cache_dir,
            download_config=download_config,
        )
    if task == "boolq":
        return load_dataset("boolq", split=split, cache_dir=cache_dir, download_config=download_config)
    raise ValueError(f"Unsupported task: {task}")


def arc_label_index(choices: dict[str, Any], answer_key: str) -> int:
    labels = [str(label) for label in choices["label"]]
    if answer_key in labels:
        return labels.index(answer_key)

    # Some ARC rows use numeric labels; support both 0-based and 1-based forms.
    if answer_key.isdigit():
        numeric = int(answer_key)
        if 0 <= numeric < len(labels):
            return numeric
        if 1 <= numeric <= len(labels):
            return numeric - 1
    raise ValueError(f"Could not match ARC answer key {answer_key!r} in {labels!r}")


def to_eval_example(task: str, row: dict[str, Any]) -> EvalExample:
    """Normalize each dataset row into one prompt, choices, and a gold index."""

    if task == "piqa":
        return EvalExample(
            prompt=f"Question: {row['goal']}\nAnswer:",
            choices=[f" {row['sol1']}", f" {row['sol2']}"],
            label=int(row["label"]),
        )

    if task in {"arc_easy", "arc_challenge"}:
        choices = [f" {text}" for text in row["choices"]["text"]]
        return EvalExample(
            prompt=f"Question: {row['question']}\nAnswer:",
            choices=choices,
            label=arc_label_index(row["choices"], str(row["answerKey"])),
        )

    if task == "hellaswag":
        context = f"{row['ctx_a']} {row['ctx_b']}".strip()
        return EvalExample(
            prompt=f"Complete the sentence:\n{context}\nContinuation:",
            choices=[f" {ending}" for ending in row["endings"]],
            label=int(row["label"]),
        )

    if task == "winogrande":
        sentence = row["sentence"].replace("_", "____")
        return EvalExample(
            prompt=f"Fill in the blank:\n{sentence}\nAnswer:",
            choices=[f" {row['option1']}", f" {row['option2']}"],
            label=int(row["answer"]) - 1,
        )

    if task == "boolq":
        return EvalExample(
            prompt=f"Passage: {row['passage']}\nQuestion: {row['question']}\nAnswer yes or no:",
            choices=[" yes", " no"],
            label=0 if bool(row["answer"]) else 1,
        )

    raise ValueError(f"Unsupported task: {task}")


def iter_examples(task: str, dataset: Iterable[dict[str, Any]], limit: int) -> Iterable[EvalExample]:
    for index, row in enumerate(dataset):
        if limit > 0 and index >= limit:
            break
        yield to_eval_example(task, row)


def choice_score(
    model,
    tokenizer,
    example: EvalExample,
    choice: str,
    *,
    device,
    max_length: int,
    length_normalize: bool,
) -> float:
    """Score one choice by the log-probability of its continuation tokens."""

    prompt_ids = tokenizer(example.prompt, add_special_tokens=True, return_tensors="pt").input_ids[0]
    choice_ids = tokenizer(choice, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if choice_ids.numel() == 0:
        return float("-inf")

    full_ids = torch.cat([prompt_ids, choice_ids], dim=0)
    start = max(0, full_ids.numel() - max_length)
    input_ids = full_ids[start:].unsqueeze(0).to(device)
    choice_start = max(0, prompt_ids.numel() - start)

    if input_ids.shape[-1] < 2:
        return float("-inf")

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=False)

    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    # Label position j corresponds to token position j + 1 in input_ids.
    positions = torch.arange(labels.shape[-1], device=device) + 1
    choice_mask = positions >= choice_start
    if not bool(choice_mask.any()):
        return float("-inf")

    score = token_log_probs[0, choice_mask].sum()
    if length_normalize:
        score = score / choice_mask.sum()
    return float(score.item())


def predict_choice(model, tokenizer, example: EvalExample, args: argparse.Namespace, device) -> int:
    scores = [
        choice_score(
            model,
            tokenizer,
            example,
            choice,
            device=device,
            max_length=args.max_length,
            length_normalize=not args.sum_logprob,
        )
        for choice in example.choices
    ]
    return max(range(len(scores)), key=scores.__getitem__)


def evaluate_task(model, tokenizer, task: str, dataset, args: argparse.Namespace, device, mode: str) -> TaskResult:
    correct = 0
    total = 0
    for total, example in enumerate(iter_examples(task, dataset, args.limit), start=1):
        pred = predict_choice(model, tokenizer, example, args, device)
        correct += int(pred == example.label)
        if total % 20 == 0:
            print(f"{mode:8s} {task:13s} {total:5d} examples accuracy={correct / total:.4f}")
    return TaskResult(task=task, mode=mode, correct=correct, total=total)


def print_results(results: list[TaskResult]) -> None:
    print("\nResults")
    print("task           mode       correct/total   accuracy")
    print("-------------  ---------  -------------   --------")
    for result in results:
        count = f"{result.correct}/{result.total}"
        print(f"{result.task:13s}  {result.mode:9s}  {count:13s}   {result.accuracy:.4f}")

    by_task = {}
    for result in results:
        by_task.setdefault(result.task, {})[result.mode] = result
    for task, modes in by_task.items():
        if "baseline" in modes and "masked" in modes:
            delta = modes["masked"].accuracy - modes["baseline"].accuracy
            print(f"delta[{task}] masked-baseline = {delta:+.4f}")


def save_results(results: list[TaskResult], output_json: str) -> None:
    payload = [
        {
            "task": result.task,
            "mode": result.mode,
            "correct": result.correct,
            "total": result.total,
            "accuracy": result.accuracy,
        }
        for result in results
    ]
    Path(output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if torch is None:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt")

    from tee_gpu_demo.llama_patch import replace_llama_attentions, replace_llama_linears

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown = sorted(set(tasks) - set(SUPPORTED_TASKS))
    if unknown:
        raise SystemExit(f"Unsupported task(s): {', '.join(unknown)}")

    trusted_device = torch.device(args.trusted_device)
    dtype = torch.float16 if trusted_device.type == "cuda" else torch.float32

    print(f"loading model={args.model} cache_dir={args.cache_dir} local_files_only={not args.allow_model_download}")
    try:
        tokenizer, model = load_model(
            args.model,
            dtype,
            trusted_device,
            cache_dir=args.cache_dir,
            revision=args.revision,
            local_files_only=not args.allow_model_download,
        )
    except OSError as exc:
        raise SystemExit(
            "Local model snapshot was not found. "
            f"Run `python cache_llama.py --model {args.model} --cache-dir {args.cache_dir}` first, "
            "or pass `--allow-model-download`."
        ) from exc

    datasets = {}
    for task in tasks:
        try:
            datasets[task] = load_hf_dataset(
                task,
                args.split,
                args.dataset_cache_dir,
                allow_download=args.allow_dataset_download,
            )
        except Exception as exc:
            raise SystemExit(
                f"Could not load dataset {task!r} from {args.dataset_cache_dir!r}. "
                "Run again with `--allow-dataset-download` to populate the local cache."
            ) from exc

    results: list[TaskResult] = []
    if args.mode in {"both", "baseline"}:
        for task in tasks:
            results.append(evaluate_task(model, tokenizer, task, datasets[task], args, trusted_device, "baseline"))

    if args.mode in {"both", "masked"}:
        include_names = tuple(name.strip() for name in args.layers.split(",") if name.strip())
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
            )
            print(f"patched_attentions={attention_report.replaced}")
        for task in tasks:
            results.append(evaluate_task(model, tokenizer, task, datasets[task], args, trusted_device, "masked"))

    print_results(results)
    if args.output_json:
        save_results(results, args.output_json)
        print(f"saved_json={args.output_json}")


if __name__ == "__main__":
    main()
