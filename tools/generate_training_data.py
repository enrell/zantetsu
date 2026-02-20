#!/usr/bin/env python3
"""Generate BIO-tagged training data from matched Nyaa titles - improved tokenization."""

import re
import os


def tokenize_aggressive(filename: str) -> list[tuple[str, int, int]]:
    """Split on common delimiters while preserving them as tokens."""
    # Split on: brackets, parentheses, spaces, dots, dashes, underscores
    pattern = r"(\[\]|\[\[|\]\]|\(\)|\(\(|\)\)|[ \[\](){}.,;:_/\-]|\s+)"
    parts = re.split(pattern, filename)

    tokens = []
    pos = 0
    for part in parts:
        if part:
            tokens.append((part, pos, pos + len(part)))
            pos += len(part)
    return tokens


def extract_title_positions(filename: str, title: str) -> list[tuple[int, int]]:
    """Find all positions of title in filename (case-insensitive)."""
    positions = []
    filename_lower = filename.lower()
    title_lower = title.lower()
    start = 0
    while True:
        idx = filename_lower.find(title_lower, start)
        if idx == -1:
            break
        positions.append((idx, idx + len(title)))
        start = idx + 1
    return positions


def label_token(
    token: str, token_start: int, token_end: int, title_positions: list[tuple[int, int]]
) -> str:
    """Assign BIO label to a token based on title positions."""
    # Skip whitespace tokens
    if token.strip() == "":
        return "O"

    for start, end in title_positions:
        # Check overlap with title span
        if token_start >= start and token_end <= end:
            if token_start == start:
                return "B-TITLE"
            else:
                return "I-TITLE"
    return "O"


def generate_training_example(filename: str, title: str) -> str:
    """Generate BIO-tagged training example."""
    tokens = tokenize_aggressive(filename)

    if not tokens:
        return None

    title_positions = extract_title_positions(filename, title)
    if not title_positions:
        return None

    # Check if any token overlaps with title
    has_title = False
    for token, start, end in tokens:
        label = label_token(token.strip(), start, end, title_positions)
        if label != "O":
            has_title = True
            break

    if not has_title:
        return None

    lines = []
    for token, start, end in tokens:
        label = label_token(token.strip(), start, end, title_positions)
        if token.strip():  # Skip whitespace-only tokens
            lines.append(f"{token}\t{label}")

    return "\n".join(lines)


# Load matched data
matched = {}
with open("data/training/nyaa_titles_matched.txt") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        match = re.match(r"\((\d+)\)\s+(.+)", line)
        if match:
            num = int(match.group(1))
            title = match.group(2).strip()
            matched[num] = title

# Load input filenames
inputs = {}
with open("data/training/nyaa_titles_input.txt") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        match = re.match(r"\((\d+)\)\s+(.+)", line)
        if match:
            num = int(match.group(1))
            inputs[num] = match.group(2).strip()

# Generate training data
output = []
skipped_no_match = 0
skipped_no_title = 0

for num in sorted(inputs.keys()):
    filename = inputs[num]
    title = matched.get(num, "")

    if title and title != "Unknown":
        example = generate_training_example(filename, title)
        if example:
            output.append(f"# {num}: {filename} → {title}")
            output.append(example)
            output.append("")
        else:
            skipped_no_title += 1
    else:
        skipped_no_match += 1

with open("data/training/bio_train.txt", "w") as f:
    f.write("\n".join(output))

print(f"Generated {len([o for o in output if o.startswith('#')])} training examples")
print(f"Skipped (no match): {skipped_no_match}")
print(f"Skipped (title not found in filename): {skipped_no_title}")
print(f"Saved to data/training/bio_train.txt")

# Show some stats
labels = []
for line in output:
    if "\t" in line and not line.startswith("#"):
        labels.append(line.split("\t")[1])

if labels:
    from collections import Counter

    c = Counter(labels)
    print(f"\nLabel distribution: {dict(c)}")
