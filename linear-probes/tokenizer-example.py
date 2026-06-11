import argparse
from pathlib import Path
import random

import torch

from utils import (
    choose_device,
    conll_to_dataset,
    load_tokenizer_and_model,
    login_to_huggingface,
    offsets_to_word_ids,
    pool_sub_tokens_to_words,
    tokens_to_text_and_spans,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_DATA_PATH = (
    REPO_ROOT
    / "ged-syntax-probing-main"
    / "datasets"
    / "wibea"
    / "wibea_conll"
    / "wibea.dev.gold.rverbsva.conll"
)
DEFAULT_EXPERIMENT_DIR = (
    SCRIPT_DIR / "experiments" / "linear-probes-20260528-184248"
)
DEFAULT_LAYER = 22
DEFAULT_MODEL_ID = "answerdotai/ModernBERT-base"
DEFAULT_NUM_EXAMPLES = 5
DEFAULT_SEED = 13


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Show ModernBERT tokens, token word_ids, gold word labels, "
            "and layer-22 probe predictions for random W&I+BEA dev examples."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to wibea.dev CoNLL file.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Directory containing the saved linear probe.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help="ModernBERT hidden-state/probe layer to use.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=DEFAULT_NUM_EXAMPLES,
        help="Number of random sentences to show.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for selecting examples.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the ModernBERT tokenizer/model from local cache only.",
    )
    return parser.parse_args()


def build_probe(hidden_size, probe_path, device):
    probe = torch.nn.Sequential(
        torch.nn.Dropout(0.1),
        torch.nn.Linear(hidden_size, 2),
    ).to(device)
    probe.load_state_dict(torch.load(probe_path, map_location=device))
    probe.eval()
    return probe


def infer_word_predictions(example, tokenizer, model, probe, layer, device):
    sentence_text, word_spans = tokens_to_text_and_spans(example["tokens"])
    encoding = tokenizer(
        sentence_text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
    )

    offset_mapping = encoding.pop("offset_mapping")[0]
    word_ids = offsets_to_word_ids(offset_mapping, word_spans)
    input_ids = encoding["input_ids"][0].tolist()
    model_inputs = encoding.to(device)

    with torch.inference_mode():
        output = model(**model_inputs, output_hidden_states=True)
        hidden_state = output.hidden_states[layer][0]
        word_vectors = pool_sub_tokens_to_words(hidden_state, word_ids)
        logits = probe(word_vectors.float())
        pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    return sentence_text, tokens, word_ids, pred_ids


def print_example(
    example_number,
    dataset_index,
    example,
    sentence,
    tokens,
    word_ids,
    preds,
    id2label,
    layer,
):
    pred_labels = [id2label[pred] for pred in preds]

    print("=" * 100)
    print(f"Example {example_number} (dataset index {dataset_index})")
    print(f"Sentence: {sentence}")
    print()
    print(f"ModernBERT tokens ({len(tokens)}):")
    print(tokens)
    print()
    print("Token word_ids:")
    print(word_ids)
    print()
    print("Word labels:")
    print(example["labels"])
    print()
    print(f"Layer-{layer} probe predictions:")
    print(pred_labels)
    print()
    print("Word-level view:")
    print(f"{'idx':>3}  {'word':<24}  {'gold':<12}  {'probe_pred'}")
    print("-" * 62)
    for idx, (word, gold_label, pred_label) in enumerate(
        zip(example["tokens"], example["labels"], pred_labels)
    ):
        print(f"{idx:>3}  {word:<24.24}  {gold_label:<12}  {pred_label}")
    print()


def main():
    args = parse_args()

    data_path = args.data_path
    experiment_dir = args.experiment_dir
    probe_path = experiment_dir / f"probe_layer_{args.layer:02d}.pt"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")
    if not probe_path.exists():
        raise FileNotFoundError(f"Missing probe file: {probe_path}")

    device = choose_device(args.device)
    hf_token = None
    if not args.local_files_only:
        hf_token = login_to_huggingface(REPO_ROOT / ".env")

    dataset = conll_to_dataset(data_path)
    if args.num_examples > len(dataset):
        raise ValueError(
            f"Requested {args.num_examples} examples, but dataset has {len(dataset)}."
        )

    tokenizer, model = load_tokenizer_and_model(
        model_id=args.model_id,
        local_files_only=args.local_files_only,
        token=hf_token,
    )
    model = model.to(device)
    model.eval()

    probe = build_probe(
        hidden_size=model.config.hidden_size,
        probe_path=probe_path,
        device=device,
    )

    label2id = {"C": 0, "R:VERB:SVA": 1}
    id2label = {idx: label for label, idx in label2id.items()}

    rng = random.Random(args.seed)
    selected_indices = rng.sample(range(len(dataset)), args.num_examples)

    print(f"data_path={data_path}")
    print(f"model_id={args.model_id}")
    print(f"probe_path={probe_path}")
    print(f"device={device}")
    print(f"random_seed={args.seed}")
    print(f"selected_indices={selected_indices}")
    print()

    for example_number, dataset_index in enumerate(selected_indices, start=1):
        example = dataset[dataset_index]
        sentence, tokens, word_ids, preds = infer_word_predictions(
            example=example,
            tokenizer=tokenizer,
            model=model,
            probe=probe,
            layer=args.layer,
            device=device,
        )

        if len(preds) != len(example["labels"]):
            raise ValueError(
                f"Prediction/label length mismatch for dataset index {dataset_index}: "
                f"{len(preds)} predictions vs {len(example['labels'])} labels."
            )

        print_example(
            example_number=example_number,
            dataset_index=dataset_index,
            example=example,
            sentence=sentence,
            tokens=tokens,
            word_ids=word_ids,
            preds=preds,
            id2label=id2label,
            layer=args.layer,
        )


if __name__ == "__main__":
    main()
