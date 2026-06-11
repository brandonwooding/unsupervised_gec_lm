#!/usr/bin/env python3
"""Align MultiGED TSV token labels to paragraph chunks from FCE JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_JSONL = Path("ged-syntax-probing-main/datasets/fce/json/fce.dev.json")
DEFAULT_TSV = Path("linear-probes/multiGED-2023-english/en_fce_dev.tsv")
DEFAULT_OUTPUT = Path("llm-probability-thresh/fce-dev-paragraph-labels.json")


class AlignmentError(RuntimeError):
    pass


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise AlignmentError(f"{path}:{line_no}: invalid JSONL row") from exc
    return records


def load_tsv_tokens(path: Path) -> list[dict]:
    tokens = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t", quotechar='"', escapechar="\\")
        for line_no, row in enumerate(reader, start=1):
            if not row or not row[0].strip():
                continue
            if len(row) < 2:
                raise AlignmentError(f"{path}:{line_no}: expected TOKEN<TAB>LABEL")
            tokens.append({"token": row[0], "label": row[1]})
    return tokens


def paragraph_spans(text: str) -> list[tuple[int, int, str]]:
    spans = []
    cursor = 0
    for paragraph in text.split("\n\n"):
        start = cursor
        end = start + len(paragraph)
        spans.append((start, end, paragraph))
        cursor = end + 2
    return spans


def skip_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def normalize_tsv_token(token: str) -> str:
    # Some readers leave escaped quotes as \" rather than unescaping them.
    if token == r"\"":
        return '"'
    return token


def context(text: str, start: int, width: int = 50) -> str:
    left = max(0, start - width)
    right = min(len(text), start + width)
    return text[left:right].replace("\n", "\\n")


def align_record(
    record: dict,
    tsv_tokens: list[dict],
    token_index: int,
    essay_index: int,
) -> tuple[list[dict], int]:
    text = record["text"]
    position = 0
    aligned_tokens = []

    while skip_whitespace(text, position) < len(text):
        position = skip_whitespace(text, position)

        if token_index >= len(tsv_tokens):
            raise AlignmentError(
                f"ran out of TSV tokens while aligning essay {essay_index}"
            )

        item = tsv_tokens[token_index]
        token = normalize_tsv_token(item["token"])
        if not text.startswith(token, position):
            raise AlignmentError(
                "token mismatch while aligning "
                f"essay={essay_index} id={record.get('id')} q={record.get('q')} "
                f"token_index={token_index} expected={token!r} "
                f"found={text[position:position + len(token) + 20]!r} "
                f"context={context(text, position)!r}"
            )

        end = position + len(token)
        aligned_tokens.append(
            {
                "token": token,
                "label": item["label"],
                "start": position,
                "end": end,
            }
        )
        position = end
        token_index += 1

    paragraphs = []
    aligned_index = 0
    for paragraph_index, (start, end, paragraph_text) in enumerate(paragraph_spans(text)):
        paragraph_tokens = []
        paragraph_labels = []
        paragraph_spans_out = []

        while (
            aligned_index < len(aligned_tokens)
            and aligned_tokens[aligned_index]["start"] < end
        ):
            item = aligned_tokens[aligned_index]
            if item["start"] < start or item["end"] > end:
                raise AlignmentError(
                    "token crossed paragraph boundary "
                    f"essay={essay_index} paragraph={paragraph_index} "
                    f"token={item['token']!r}"
                )
            paragraph_tokens.append(item["token"])
            paragraph_labels.append(item["label"])
            paragraph_spans_out.append([item["start"] - start, item["end"] - start])
            aligned_index += 1

        paragraphs.append(
            {
                "text": paragraph_text,
                "tokens": paragraph_tokens,
                "labels": paragraph_labels,
                "spans": paragraph_spans_out,
            }
        )

    return paragraphs, token_index


def align(jsonl_path: Path, tsv_path: Path) -> tuple[list[dict], dict]:
    records = load_jsonl(jsonl_path)
    tsv_tokens = load_tsv_tokens(tsv_path)

    aligned = []
    token_index = 0
    for essay_index, record in enumerate(records):
        paragraphs, token_index = align_record(
            record=record,
            tsv_tokens=tsv_tokens,
            token_index=token_index,
            essay_index=essay_index,
        )
        aligned.append(paragraphs)

    if token_index != len(tsv_tokens):
        raise AlignmentError(
            f"unused TSV tokens: consumed {token_index} of {len(tsv_tokens)}"
        )

    paragraph_counts = [len(paragraphs) for paragraphs in aligned]
    label_counts = Counter(token["label"] for token in tsv_tokens)
    summary = {
        "essays": len(records),
        "paragraphs": sum(paragraph_counts),
        "tokens": len(tsv_tokens),
        "labels": dict(sorted(label_counts.items())),
        "min_paragraphs_per_essay": min(paragraph_counts) if paragraph_counts else 0,
        "max_paragraphs_per_essay": max(paragraph_counts) if paragraph_counts else 0,
    }
    return aligned, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align token-level c/i labels from the MultiGED FCE TSV back onto "
            "paragraph chunks from the FCE JSONL file."
        )
    )
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate alignment and print the summary without writing output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aligned, summary = align(args.jsonl, args.tsv)

    print(json.dumps(summary, indent=2))

    if args.check:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(aligned, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
