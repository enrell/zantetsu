#!/usr/bin/env python3
"""Visualize improved CRF model predictions vs ground truth - using trained weights."""

import json
import random

# Load model - trained weights from Rust
with open("models/crf_model_v2.json") as f:
    model_data = json.load(f)

transition = model_data["transition"]
emission_weights = model_data["emission_weights"]
NUM_LABELS = 3

print("Loaded model weights:")
print(f"  emission_weights: {emission_weights}")
print(f"  transition: {transition}")

# Labels map
LABELS = {0: "O", 1: "B-TITLE", 2: "I-TITLE"}


def extract_features(token, prev_token=None, next_token=None):
    """Extract features for a token."""
    features = []

    lower = token.lower()

    # Basic features (match Rust model)
    is_all_caps = len(token) > 1 and all(not c.isalpha() or c.isupper() for c in token)
    features.append(1.0 if is_all_caps else 0.0)  # is_all_caps
    features.append(
        1.0 if token.startswith("[") or token.startswith("(") else 0.0
    )  # has_bracket_start
    features.append(
        1.0 if token.endswith("]") or token.endswith(")") else 0.0
    )  # has_bracket_end
    features.append(
        1.0
        if "e0" in lower or "s0" in lower or (token.isdigit() and len(token) <= 2)
        else 0.0
    )  # is_episode
    features.append(
        1.0
        if "720p" in lower
        or "1080p" in lower
        or "480p" in lower
        or lower == "bd"
        or lower == "web"
        else 0.0
    )  # is_quality
    features.append(1.0 if any(c.isdigit() for c in token) else 0.0)  # has_digit
    features.append(1.0 if len(token) > 3 else 0.0)  # long_token

    # Context features
    features.append(
        1.0
        if prev_token and (prev_token.startswith("[") or prev_token.startswith("("))
        else 0.0
    )
    features.append(
        1.0
        if next_token and (next_token.startswith("[") or next_token.startswith("("))
        else 0.0
    )

    return features


def compute_emission(token, prev_token, next_token, label):
    """Compute emission score using trained weights."""
    features = extract_features(token, prev_token, next_token)

    bias = emission_weights[label]
    score = bias

    # Use same weights as Rust model
    if label == 0:  # O
        score += features[2] * 2.0  # bracket end
        score += features[3] * 2.0  # episode
        score += features[4] * 2.0  # quality
        score -= features[0] * 1.0  # all_caps
    elif label == 1:  # B-TITLE
        score += features[0] * 2.0  # all_caps
        score -= features[2] * 2.0  # bracket
        score -= features[3] * 2.0  # episode
        score -= features[4] * 2.0  # quality
        score += features[5] * 0.5  # digit
    elif label == 2:  # I-TITLE
        score += features[0] * 1.5  # all_caps
        score -= features[2] * 2.0  # bracket
        score -= features[3] * 2.0  # episode
        score -= features[4] * 2.0  # quality

    return score


def forward(tokens):
    """Compute emission scores."""
    emissions = []

    for i, token in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else None
        next = tokens[i + 1] if i < len(tokens) - 1 else None

        for label in range(NUM_LABELS):
            emissions.append(compute_emission(token, prev, next, label))

    return emissions


def viterbi_decode(emissions, transitions, num_labels):
    if not emissions or num_labels == 0:
        return []

    seq_len = len(emissions) // num_labels
    if seq_len == 0:
        return []

    viterbi = [[float("-inf")] * num_labels for _ in range(seq_len)]
    backpointers = [[0] * num_labels for _ in range(max(1, seq_len - 1))]

    # Initialize
    for j in range(num_labels):
        if j < len(emissions):
            viterbi[0][j] = emissions[j]

    # Forward pass
    for t in range(1, seq_len):
        for j in range(num_labels):
            best_score = float("-inf")
            best_prev = 0

            for i in range(num_labels):
                score = viterbi[t - 1][i] + transitions[j * num_labels + i]
                if score > best_score:
                    best_score = score
                    best_prev = i

            emission_idx = t * num_labels + j
            if emission_idx < len(emissions):
                viterbi[t][j] = best_score + emissions[emission_idx]
            if t - 1 < len(backpointers):
                backpointers[t - 1][j] = best_prev

    # Backtrack
    path = [0] * seq_len
    if seq_len > 0:
        last_row = viterbi[seq_len - 1]
        path[seq_len - 1] = max(range(num_labels), key=lambda i: last_row[i])

        for t in range(seq_len - 2, -1, -1):
            if t + 1 < len(backpointers):
                path[t] = backpointers[t + 1][path[t + 1]]

    return path


# Load training data
def load_bio_dataset(path):
    examples = []
    with open(path) as f:
        tokens = []
        labels = []

        for line in f:
            line = line.strip()

            if not line:
                if tokens:
                    examples.append({"tokens": tokens, "labels": labels})
                    tokens = []
                    labels = []
                continue

            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) == 2:
                token = parts[0]
                label_str = parts[1]

                label_idx = {"O": 0, "B-TITLE": 1, "I-TITLE": 2}.get(label_str)
                if label_idx is not None:
                    tokens.append(token)
                    labels.append(label_idx)

        if tokens:
            examples.append({"tokens": tokens, "labels": labels})

    return examples


print("Loading dataset...")
examples = load_bio_dataset("data/training/bio_train_50k.txt")

# Sample 10 random examples
random.seed(42)
samples = random.sample(examples, min(10, len(examples)))

print("\n" + "=" * 80)
print(f"{'TOKEN':<25} | {'PREDICTED':<12} | {'EXPECTED':<12} | {'MATCH'}")
print("=" * 80)

total_correct = 0
total_tokens = 0

for i, ex in enumerate(samples):
    if not ex["tokens"]:
        continue

    tokens = ex["tokens"]
    true_labels = ex["labels"]

    # Get predictions
    emissions = forward(tokens)
    preds = viterbi_decode(emissions, transition, NUM_LABELS)

    for j, token in enumerate(tokens):
        pred = LABELS.get(preds[j] if j < len(preds) else 0, "O")
        expected = LABELS.get(true_labels[j] if j < len(true_labels) else 0, "O")

        match = "✓" if pred == expected else "✗"
        if pred == expected:
            total_correct += 1
        total_tokens += 1

        token_display = token[:23].ljust(25)

        if j < len(preds):
            print(f"{token_display} | {pred:<12} | {expected:<12} | {match}")

    if i < len(samples) - 1:
        print("-" * 80)

print("=" * 80)
accuracy = (total_correct / total_tokens * 100) if total_tokens > 0 else 0
print(f"\nOverall Accuracy: {total_correct}/{total_tokens} ({accuracy:.1f}%)")
