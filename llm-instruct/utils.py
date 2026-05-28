import json
import os
from dotenv import load_dotenv
from pathlib import Path

import torch
from huggingface_hub import login
from sklearn.metrics import accuracy_score, fbeta_score, precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer


LABEL2ID = {"C": 0, "R:VERB:SVA": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
VALID_LABELS = set(LABEL2ID)


def conll_to_examples(file_path: Path) -> list[dict]:
    """
    Read a token-label CoNLL file into sentence-level examples.

    Blank lines separate sentences. The first column is the token and the last
    column is the label.
    """
    file_path = Path(file_path)
    examples = []
    tokens = []
    labels = []
    sentence_idx = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if tokens:
                    examples.append(
                        {
                            "source_file": file_path.name,
                            "sentence_idx": sentence_idx,
                            "tokens": tokens,
                            "gold_labels": labels,
                        }
                    )
                    sentence_idx += 1
                    tokens = []
                    labels = []
                continue

            columns = line.split()
            tokens.append(columns[0])
            labels.append(columns[-1].upper())

    if tokens:
        examples.append(
            {
                "source_file": file_path.name,
                "sentence_idx": sentence_idx,
                "tokens": tokens,
                "gold_labels": labels,
            }
        )

    return examples


def load_eval_examples(data_dir: Path) -> list[dict]:
    data_dir = Path(data_dir)
    conll_paths = sorted(data_dir.glob("*.conll"))
    if not conll_paths:
        raise FileNotFoundError(f"No .conll files found in {data_dir}")

    examples = []
    for conll_path in conll_paths:
        examples.extend(conll_to_examples(conll_path))
    return examples


def extract_json_object(text: str) -> str:
    """Return the first JSON object embedded in text."""
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    decoder = json.JSONDecoder()
    obj, end = decoder.raw_decode(text[start:])
    return json.dumps(obj)


def parse_prediction(text: str, expected_len: int) -> tuple[list[str] | None, bool, str | None]:
    try:
        json_text = extract_json_object(text)
        data = json.loads(json_text)
    except Exception as exc:
        return None, False, f"invalid_json: {exc}"

    if not isinstance(data, dict):
        return None, False, "json_not_dict"

    labels = data.get("labels")
    if not isinstance(labels, list):
        return None, False, "labels_missing_or_not_list"

    labels = [str(label).strip().upper() for label in labels]
    if len(labels) != expected_len:
        return None, False, f"wrong_label_count: expected={expected_len} got={len(labels)}"

    unknown_labels = sorted(set(labels) - VALID_LABELS)
    if unknown_labels:
        return None, False, f"unknown_labels: {unknown_labels}"

    return labels, True, None


def labels_to_ids(labels: list[str]) -> torch.Tensor:
    return torch.tensor([LABEL2ID[label] for label in labels], dtype=torch.long)


def incorrect_labels_for(gold_labels: list[str]) -> list[str]:
    return ["R:VERB:SVA" if label == "C" else "C" for label in gold_labels]


def safe_model_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")

def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device


def login_to_huggingface(env_file: Path) -> str | None:
    if env_file.exists():
        load_dotenv(env_file)
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("huggingface_login=skipped token_env_missing")
        return None
    login(token=token, add_to_git_credential=False)
    print("huggingface_login=ok")
    return token

def load_tokenizer_and_model(
    model_id: str, local_files_only: bool, token: str | None
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, local_files_only=local_files_only, token=token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=local_files_only, token=token
    )
    return tokenizer, model


def compute_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1],
        average="binary",
        zero_division=0,
    )

    f05 = fbeta_score(
        y_true,
        y_pred,
        beta=0.5,
        pos_label=1,
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f0.5": float(f05),
    }
