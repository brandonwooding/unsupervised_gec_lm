import argparse
import json
from pathlib import Path

import torch

from utils import (
    compute_metrics,
    choose_device,
    incorrect_labels_for,
    labels_to_ids,
    load_eval_examples,
    login_to_huggingface,
    parse_prediction,
    safe_model_name,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = (
    SCRIPT_DIR.parent
    / "ged-syntax-probing-main"
    / "datasets"
    / "marvin_linzen"
    / "stimuli"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_PROMPT_PATH = SCRIPT_DIR / "prompts" / "prompt-SVA-GED.md"
DEFAULT_MODEL_GRID = [
    "Qwen/Qwen3.5-0.8B",
    "google/gemma-4-E2B-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "mistralai/Ministral-3-3B-Instruct-2512",
]


def build_user_prompt(tokens: list[str]) -> str:
    return (
        "Label the following tokenized sentence for subject-verb agreement errors.\n"
        f"Tokens: {json.dumps(tokens)}\n"
        'Return only JSON in this form: {"labels": ["C", "..."]}'
    )


def build_prompt(tokenizer, system_prompt: str, tokens: list[str]) -> str:
    user_prompt = build_user_prompt(tokens)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"{system_prompt}\n\n{user_prompt}\n\nJSON:"


def generate_response(
    tokenizer,
    model,
    device,
    system_prompt: str,
    tokens: list[str],
    max_new_tokens: int,
) -> str:
    
    prompt = build_prompt(tokenizer, system_prompt, tokens)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate_model(
    model_id: str,
    examples: list[dict],
    system_prompt: str,
    output_dir: Path,
    device,
    hf_token: str | None,
    local_files_only: bool,
    max_new_tokens: int,
) -> dict:
    model_output_dir = output_dir / safe_model_name(model_id)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = model_output_dir / "predictions.jsonl"
    metrics_path = model_output_dir / "metrics.json"

    print(f"loading model_id={model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        local_files_only=local_files_only,
        torch_dtype=torch.float16 if device.type in {"cuda", "mps"} else torch.float32,
    )
    model.to(device)
    model.eval()

    all_gold_labels = []
    all_pred_labels = []
    parse_failures = 0

    with open(predictions_path, "w", encoding="utf-8") as f:
        for example_idx, example in enumerate(examples, start=1):
            if example_idx % 100 == 0 or example_idx == 1:
                print(f"{model_id}: sentence {example_idx}/{len(examples)}")

            raw_response = generate_response(
                tokenizer=tokenizer,
                model=model,
                device=device,
                system_prompt=system_prompt,
                tokens=example["tokens"],
                max_new_tokens=max_new_tokens,
            )

            pred_labels, parse_ok, parse_error = parse_prediction(
                raw_response,
                expected_len=len(example["tokens"]),
            )
            if not parse_ok:
                parse_failures += 1
                pred_labels = incorrect_labels_for(example["gold_labels"])

            record = {
                **example,
                "model_id": model_id,
                "pred_labels": pred_labels,
                "raw_response": raw_response,
                "parse_ok": parse_ok,
                "parse_error": parse_error,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

            all_gold_labels.extend(example["gold_labels"])
            all_pred_labels.extend(pred_labels)

    metrics = compute_metrics(
        y_true=labels_to_ids(all_gold_labels),
        y_pred=labels_to_ids(all_pred_labels),
    )
    metrics = {
        "model_id": model_id,
        "num_sentences": len(examples),
        "num_tokens": len(all_gold_labels),
        "parse_failures": parse_failures,
        **metrics,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"saved predictions={predictions_path}")
    print(f"saved metrics={metrics_path}")

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument(
        "--model-id",
        action="append",
        default=None,
        help="Hugging Face model id. Can be passed multiple times.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    system_prompt = args.prompt_path.read_text(encoding="utf-8")
    examples = load_eval_examples(args.data_dir)
    hf_token = login_to_huggingface(SCRIPT_DIR.parent / ".env")
    device = choose_device(args.device)
    model_grid = args.model_id or DEFAULT_MODEL_GRID

    print(f"device={device}")
    print(f"data_dir={args.data_dir}")
    print(f"examples={len(examples)}")
    print(f"models={len(model_grid)}")

    all_metrics = []
    for model_id in model_grid:
        metrics = evaluate_model(
            model_id=model_id,
            examples=examples,
            system_prompt=system_prompt,
            output_dir=args.output_dir,
            device=device,
            hf_token=hf_token,
            local_files_only=args.local_files_only,
            max_new_tokens=args.max_new_tokens,
        )
        all_metrics.append(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"saved summary={summary_path}")


if __name__ == "__main__":
    main()
