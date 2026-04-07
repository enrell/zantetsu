#!/usr/bin/env python3
"""Generate a structurally tokenized hybrid BIO dataset from RAD JSONL samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from ..common.jsonl import iter_jsonl
from .hybrid_tokenizer import (
    AUDIO_TERMS,
    CODEC_TERMS,
    CONTAINER_TERMS,
    LANGUAGE_TERMS,
    QUALITY_TERMS,
    RESOLUTION_TERMS,
    SOURCE_TERMS,
    SUBTITLE_TERMS,
    HybridToken,
    build_context_features,
    tokenize_filename,
)


YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
EPISODE_RANGE_RE = re.compile(r"^\d{1,4}-\d{1,4}$")
SEASON_RE = re.compile(r"(?i)^s\d{1,2}$")
EPISODE_RE = re.compile(r"(?i)^(?:e|ep)\d{1,4}$")
VERSION_RE = re.compile(r"(?i)^v\d+$")
CRC32_RE = re.compile(r"^[0-9a-f]{8}$", re.IGNORECASE)
def token_indices_for_span(
    tokens: list[HybridToken], start: int, end: int
) -> list[int]:
    return [
        token.index
        for token in tokens
        if token.end > start and token.start < end and token.text.strip()
    ]


def mark_indices(
    tags: list[str],
    indices: list[int],
    label: str,
    *,
    allow_overwrite: bool = False,
) -> bool:
    if not indices:
        return False
    if not allow_overwrite and any(tags[index] != "O" for index in indices):
        return False

    tags[indices[0]] = f"B-{label}"
    for index in indices[1:]:
        tags[index] = f"I-{label}"
    return True


def find_casefold_spans(text: str, candidate: str) -> list[tuple[int, int]]:
    if not candidate:
        return []

    spans: list[tuple[int, int]] = []
    haystack = text.casefold()
    needle = candidate.casefold()
    cursor = 0
    while True:
        position = haystack.find(needle, cursor)
        if position == -1:
            break
        spans.append((position, position + len(candidate)))
        cursor = position + 1
    return spans


def best_span_match(
    text: str,
    tokens: list[HybridToken],
    candidates: list[str],
    *,
    prefer_bracketed: bool = False,
) -> list[int]:
    scored_matches: list[tuple[int, int, list[int]]] = []

    for candidate in candidates:
        for start, end in find_casefold_spans(text, candidate):
            indices = token_indices_for_span(tokens, start, end)
            if not indices:
                continue
            bracket_score = sum(1 for index in indices if tokens[index].inside_brackets)
            score = bracket_score if prefer_bracketed else 0
            scored_matches.append((score, len(indices), indices))

    if not scored_matches:
        return []

    scored_matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored_matches[0][2]


def title_variants(title: str) -> list[str]:
    variants = {title}
    variants.add(title.replace(" ", "."))
    variants.add(title.replace(" ", "_"))
    variants.add(title.replace(" - ", "-"))
    variants.add(title.replace(":", " -"))
    return [variant for variant in variants if variant]


def group_variants(group: str) -> list[str]:
    variants = {group, f"[{group}]", f"({group})", f"{{{group}}}"}
    return [variant for variant in variants if variant]


def episode_context(tokens: list[HybridToken], index: int) -> bool:
    previous = tokens[index - 1].normalized if index > 0 else ""
    next_token = tokens[index + 1].normalized if index + 1 < len(tokens) else ""
    return previous in {"-", ".", "_", "episode", "ep", "e"} or next_token in {
        ")",
        "]",
        ".",
        "-",
    }


def build_tags(text: str, tokens: list[HybridToken], metadata: dict[str, Any]) -> list[str]:
    tags = ["O"] * len(tokens)

    group = str(metadata.get("group") or "").strip()
    if group:
        mark_indices(
            tags,
            best_span_match(text, tokens, group_variants(group), prefer_bracketed=True),
            "GROUP",
        )

    title = str(metadata.get("title") or "").strip()
    if title:
        mark_indices(tags, best_span_match(text, tokens, title_variants(title)), "TITLE")

    episode = str(metadata.get("episode") or "").strip()

    for index, token in enumerate(tokens):
        normalized = token.normalized

        if normalized in {"[", "]", "(", ")", "{", "}", ".", "-", "_", "+"}:
            continue

        if EPISODE_RANGE_RE.fullmatch(normalized):
            mark_indices(tags, [index], "EP_RANGE")
            continue

        if YEAR_RE.fullmatch(normalized):
            mark_indices(tags, [index], "YEAR")
            continue

        if CRC32_RE.fullmatch(normalized) and token.inside_brackets:
            mark_indices(tags, [index], "CRC32")
            continue

        if normalized in RESOLUTION_TERMS:
            mark_indices(tags, [index], "RESOLUTION")
            continue

        if normalized in SOURCE_TERMS:
            mark_indices(tags, [index], "SOURCE")
            continue

        if normalized in CODEC_TERMS:
            mark_indices(tags, [index], "CODEC")
            continue

        if normalized in AUDIO_TERMS:
            mark_indices(tags, [index], "AUDIO")
            continue

        if normalized in LANGUAGE_TERMS:
            mark_indices(tags, [index], "LANG")
            continue

        if normalized in SUBTITLE_TERMS:
            mark_indices(tags, [index], "SUB")
            continue

        if normalized in QUALITY_TERMS:
            mark_indices(tags, [index], "QUALITY")
            continue

        if normalized in CONTAINER_TERMS and (
            index == len(tokens) - 1 or (index > 0 and tokens[index - 1].normalized == ".")
        ):
            mark_indices(tags, [index], "CONTAINER")
            continue

        if SEASON_RE.fullmatch(normalized):
            mark_indices(tags, [index], "SEASON")
            continue

        if normalized == "season" and index + 1 < len(tokens) and tokens[index + 1].normalized.isdigit():
            mark_indices(tags, [index, index + 1], "SEASON")
            continue

        if EPISODE_RE.fullmatch(normalized):
            mark_indices(tags, [index], "EPISODE")
            continue

        if normalized in {"episode", "ep"} and index + 1 < len(tokens) and tokens[index + 1].normalized.isdigit():
            mark_indices(tags, [index, index + 1], "EPISODE")
            continue

        if VERSION_RE.fullmatch(normalized):
            mark_indices(tags, [index], "VERSION")
            continue

        if episode and normalized == episode and episode_context(tokens, index):
            mark_indices(tags, [index], "EPISODE")

    return tags


def build_example(record: dict[str, Any], include_features: bool) -> dict[str, Any] | None:
    text = str(record.get("filename") or record.get("text") or "").strip()
    if not text:
        return None

    tokens = tokenize_filename(text)
    if not tokens:
        return None

    metadata = record.get("metadata") or {}
    tags = build_tags(text, tokens, metadata)

    if "B-TITLE" not in tags:
        return None

    result: dict[str, Any] = {
        "text": text,
        "tokens": [token.text for token in tokens],
        "tags": tags,
        "token_spans": [[token.start, token.end] for token in tokens],
        "metadata": metadata,
        "augmentation": record.get("augmentation", "none"),
    }

    if include_features:
        result["features"] = build_context_features(tokens)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a structurally tokenized hybrid BIO dataset"
    )
    parser.add_argument(
        "--input",
        default="data/training/rad_dataset_50k.jsonl",
        help="Input RAD dataset JSONL",
    )
    parser.add_argument(
        "--output",
        default="data/training/hybrid_dataset.jsonl",
        help="Output hybrid dataset JSONL",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional maximum number of samples to emit",
    )
    parser.add_argument(
        "--include-features",
        action="store_true",
        help="Persist derived token feature dictionaries into the output JSONL",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    emitted = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(input_path):
            example = build_example(record, include_features=args.include_features)
            if example is None:
                skipped += 1
                continue

            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            emitted += 1
            if args.max_samples and emitted >= args.max_samples:
                break

    print(f"Wrote {emitted} hybrid samples to {output_path}")
    if skipped:
        print(f"Skipped {skipped} records without a recoverable title span")


if __name__ == "__main__":
    main()
