#!/usr/bin/env python3
"""Train a BiLSTM-CRF with char embeddings on the hybrid BIO dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from ..common.jsonl import iter_jsonl
from ..common.torch_runtime import configure_torch_runtime
from ..data.hybrid_tokenizer import (
    LEXICON_CATEGORIES,
    LEXICON_LOOKUP,
    build_context_features_from_texts,
)


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_CHAR = "<PAD>"
UNK_CHAR = "<UNK>"

BRACKET_TO_ID = {None: 0, "[": 1, "(": 2, "{": 3}
POSITION_TO_ID = {"singleton": 0, "start": 1, "middle": 2, "end": 3}
def load_dataset(path: Path, max_samples: int = 0) -> list[dict[str, Any]]:
    records = list(iter_jsonl(path))
    return records[:max_samples] if max_samples else records


def build_vocab(records: list[dict[str, Any]], min_freq: int) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    token_counter: Counter[str] = Counter()
    char_counter: Counter[str] = Counter()
    labels: set[str] = {"O"}

    for record in records:
        for token in record["tokens"]:
            token_counter[token.lower()] += 1
            char_counter.update(token)
        labels.update(record["tags"])

    token_vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in token_counter.items():
        if count >= min_freq:
            token_vocab[token] = len(token_vocab)

    char_vocab = {PAD_CHAR: 0, UNK_CHAR: 1}
    for char, _ in char_counter.most_common():
        char_vocab[char] = len(char_vocab)

    label_vocab = {label: index for index, label in enumerate(sorted(labels))}
    return token_vocab, char_vocab, label_vocab


def split_records(records: list[dict[str, Any]], validation_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    validation_size = max(1, int(len(shuffled) * validation_ratio)) if len(shuffled) > 1 else 0
    validation_records = shuffled[:validation_size]
    training_records = shuffled[validation_size:] if validation_size else shuffled
    return training_records, validation_records


@dataclass
class EncodedSample:
    token_ids: list[int]
    char_ids: list[list[int]]
    bracket_ids: list[int]
    position_ids: list[int]
    lexicon_ids: list[int]
    label_ids: list[int]


class HybridDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        token_vocab: dict[str, int],
        char_vocab: dict[str, int],
        label_vocab: dict[str, int],
        max_char_len: int,
    ):
        self.samples: list[EncodedSample] = []
        self.token_vocab = token_vocab
        self.char_vocab = char_vocab
        self.label_vocab = label_vocab
        self.max_char_len = max_char_len

        for record in records:
            self.samples.append(self.encode_record(record))

    def encode_record(self, record: dict[str, Any]) -> EncodedSample:
        tokens = record["tokens"]
        tags = record["tags"]

        features = record.get("features")
        if features is None:
            features = build_context_features_from_texts(tokens)

        token_ids = [self.token_vocab.get(token.lower(), self.token_vocab[UNK_TOKEN]) for token in tokens]
        char_ids = [
            [self.char_vocab.get(char, self.char_vocab[UNK_CHAR]) for char in token[: self.max_char_len]]
            for token in tokens
        ]
        bracket_ids = [
            BRACKET_TO_ID[feature["bracket_kind"] if feature["inside_brackets"] else None]
            for feature in features
        ]
        position_ids = [POSITION_TO_ID[feature["position_bucket"]] for feature in features]
        lexicon_ids = [LEXICON_LOOKUP.get(token.lower(), 0) for token in tokens]
        label_ids = [self.label_vocab[tag] for tag in tags]

        return EncodedSample(token_ids, char_ids, bracket_ids, position_ids, lexicon_ids, label_ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> EncodedSample:
        return self.samples[index]


def collate_batch(batch: list[EncodedSample]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_seq_len = max(len(sample.token_ids) for sample in batch)
    max_char_len = max(
        max((len(chars) for chars in sample.char_ids), default=1) for sample in batch
    )

    token_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    char_ids = torch.zeros((batch_size, max_seq_len, max_char_len), dtype=torch.long)
    bracket_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    position_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    lexicon_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    label_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

    for row, sample in enumerate(batch):
        seq_len = len(sample.token_ids)
        token_ids[row, :seq_len] = torch.tensor(sample.token_ids, dtype=torch.long)
        bracket_ids[row, :seq_len] = torch.tensor(sample.bracket_ids, dtype=torch.long)
        position_ids[row, :seq_len] = torch.tensor(sample.position_ids, dtype=torch.long)
        lexicon_ids[row, :seq_len] = torch.tensor(sample.lexicon_ids, dtype=torch.long)
        label_ids[row, :seq_len] = torch.tensor(sample.label_ids, dtype=torch.long)
        mask[row, :seq_len] = True

        for index, token_chars in enumerate(sample.char_ids):
            char_ids[row, index, : len(token_chars)] = torch.tensor(token_chars, dtype=torch.long)

    return {
        "token_ids": token_ids,
        "char_ids": char_ids,
        "bracket_ids": bracket_ids,
        "position_ids": position_ids,
        "lexicon_ids": lexicon_ids,
        "labels": label_ids,
        "mask": mask,
    }


class LinearChainCrf(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        nn.init.xavier_uniform_(self.transitions)
        nn.init.zeros_(self.start_transitions)
        nn.init.zeros_(self.end_transitions)
        # Non-learnable masks for BIO-invalid transitions (set via apply_bio_constraints)
        self.register_buffer("invalid_trans_mask", torch.zeros(num_tags, num_tags, dtype=torch.bool))
        self.register_buffer("invalid_start_mask", torch.zeros(num_tags, dtype=torch.bool))

    def apply_bio_constraints(self, label_vocab: dict[str, int]) -> None:
        """Mask out BIO-invalid transitions with -1e9 during forward/decode."""
        for from_label, from_idx in label_vocab.items():
            for to_label, to_idx in label_vocab.items():
                if to_label.startswith("I-"):
                    entity_type = to_label[2:]
                    if from_label == "O":
                        self.invalid_trans_mask[from_idx, to_idx] = True
                    elif from_label[2:] != entity_type:
                        # B-X → I-Y or I-X → I-Y where X ≠ Y
                        self.invalid_trans_mask[from_idx, to_idx] = True
        for label, idx in label_vocab.items():
            if label.startswith("I-"):
                self.invalid_start_mask[idx] = True

    def _masked_transitions(self) -> torch.Tensor:
        return self.transitions.masked_fill(self.invalid_trans_mask, -1e9)

    def _masked_start_transitions(self) -> torch.Tensor:
        return self.start_transitions.masked_fill(self.invalid_start_mask, -1e9)

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gold_score = self.compute_gold_score(emissions, tags, mask)
        partition = self.compute_normalizer(emissions, mask)
        return (partition - gold_score).mean()

    def compute_gold_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        trans = self._masked_transitions()
        score = self._masked_start_transitions()[tags[:, 0]]
        score = score + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)

        for timestep in range(1, emissions.size(1)):
            transition = trans[tags[:, timestep - 1], tags[:, timestep]]
            emission = emissions[:, timestep].gather(1, tags[:, timestep : timestep + 1]).squeeze(1)
            score = score + (transition + emission) * mask[:, timestep]

        last_index = mask.long().sum(dim=1) - 1
        last_tag = tags.gather(1, last_index.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tag]
        return score

    def compute_normalizer(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self._masked_start_transitions() + emissions[:, 0]

        for timestep in range(1, emissions.size(1)):
            broadcast_score = score.unsqueeze(2)
            broadcast_transitions = self._masked_transitions().unsqueeze(0)
            broadcast_emission = emissions[:, timestep].unsqueeze(1)
            next_score = broadcast_score + broadcast_transitions + broadcast_emission
            next_score = torch.logsumexp(next_score, dim=1)
            timestep_mask = mask[:, timestep].unsqueeze(1)
            score = next_score * timestep_mask + score * (~timestep_mask)

        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        batch_size, seq_len, _ = emissions.shape
        score = self._masked_start_transitions() + emissions[:, 0]
        history: list[torch.Tensor] = []

        for timestep in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            next_score = broadcast_score + self._masked_transitions().unsqueeze(0)
            next_score, indices = next_score.max(dim=1)
            next_score = next_score + emissions[:, timestep]
            timestep_mask = mask[:, timestep].unsqueeze(1)
            score = next_score * timestep_mask + score * (~timestep_mask)
            history.append(indices)

        score = score + self.end_transitions
        best_last_tags = score.argmax(dim=1)

        best_tags = [best_last_tags.tolist()]
        for hist in reversed(history):
            best_last_tags = hist.gather(1, best_last_tags.unsqueeze(1)).squeeze(1)
            best_tags.insert(0, best_last_tags.tolist())

        sequences = [list(sequence) for sequence in zip(*best_tags)]
        lengths = mask.long().sum(dim=1).tolist()
        return [sequence[:length] for sequence, length in zip(sequences, lengths)]


class HybridBiLstmCrf(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        char_vocab_size: int,
        num_tags: int,
        token_dim: int = 128,
        char_dim: int = 32,
        char_filters: int = 64,
        bracket_dim: int = 8,
        position_dim: int = 8,
        lexicon_dim: int = 16,
        hidden_dim: int = 160,
        num_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, token_dim, padding_idx=0)
        self.char_embedding = nn.Embedding(char_vocab_size, char_dim, padding_idx=0)
        self.char_conv = nn.Conv1d(char_dim, char_filters, kernel_size=3, padding=1)
        self.bracket_embedding = nn.Embedding(len(BRACKET_TO_ID), bracket_dim)
        self.position_embedding = nn.Embedding(len(POSITION_TO_ID), position_dim)
        self.lexicon_embedding = nn.Embedding(len(LEXICON_CATEGORIES), lexicon_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_dim = token_dim + char_filters + bracket_dim + position_dim + lexicon_dim
        self.encoder = nn.LSTM(
            encoder_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_tags)
        self.crf = LinearChainCrf(num_tags)

    def encode_characters(self, char_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, char_len = char_ids.shape
        embedded = self.char_embedding(char_ids)
        embedded = embedded.view(batch_size * seq_len, char_len, -1).transpose(1, 2)
        encoded = F.relu(self.char_conv(embedded))
        encoded = encoded.max(dim=2).values
        return encoded.view(batch_size, seq_len, -1)

    def forward(
        self,
        token_ids: torch.Tensor,
        char_ids: torch.Tensor,
        bracket_ids: torch.Tensor,
        position_ids: torch.Tensor,
        lexicon_ids: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_embedding = self.token_embedding(token_ids)
        char_embedding = self.encode_characters(char_ids)
        bracket_embedding = self.bracket_embedding(bracket_ids)
        position_embedding = self.position_embedding(position_ids)
        lexicon_embedding = self.lexicon_embedding(lexicon_ids)

        encoded_input = torch.cat(
            [token_embedding, char_embedding, bracket_embedding, position_embedding, lexicon_embedding], dim=-1
        )
        encoded_input = self.dropout(encoded_input)
        encoded_output, _ = self.encoder(encoded_input)
        encoded_output = self.dropout(encoded_output)
        emissions = self.classifier(encoded_output)
        emissions = emissions.masked_fill(~mask.unsqueeze(-1), -1e4)

        if labels is None:
            return emissions
        return self.crf(emissions, labels, mask)

    def decode(
        self,
        token_ids: torch.Tensor,
        char_ids: torch.Tensor,
        bracket_ids: torch.Tensor,
        position_ids: torch.Tensor,
        lexicon_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[list[int]]:
        emissions = self.forward(token_ids, char_ids, bracket_ids, position_ids, lexicon_ids, mask)
        return self.crf.decode(emissions, mask)


def decode_labels(ids: list[int], id_to_label: dict[int, str]) -> list[str]:
    return [id_to_label[index] for index in ids]


def extract_spans(labels: list[str]) -> set[tuple[int, int, str]]:
    spans: set[tuple[int, int, str]] = set()
    start = None
    current_type = None

    for index, label in enumerate(labels):
        if label == "O":
            if current_type is not None and start is not None:
                spans.add((start, index, current_type))
            start = None
            current_type = None
            continue

        prefix, entity_type = label.split("-", 1)
        if prefix == "B" or entity_type != current_type:
            if current_type is not None and start is not None:
                spans.add((start, index, current_type))
            start = index
            current_type = entity_type

    if current_type is not None and start is not None:
        spans.add((start, len(labels), current_type))
    return spans


def compute_metrics(
    predictions: list[list[int]],
    labels: list[list[int]],
    masks: list[list[bool]],
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    total_tokens = 0
    correct_tokens = 0
    predicted_spans = 0
    gold_spans = 0
    correct_spans = 0
    per_label_counts: dict[str, dict[str, int]] = {}

    entity_labels = sorted(
        {
            label.split("-", 1)[1]
            for label in id_to_label.values()
            if label != "O" and "-" in label
        }
    )
    for entity_label in entity_labels:
        per_label_counts[entity_label] = {
            "predicted_spans": 0,
            "gold_spans": 0,
            "correct_spans": 0,
        }

    for prediction, gold, mask in zip(predictions, labels, masks):
        active_length = sum(mask)
        trimmed_prediction = prediction[:active_length]
        trimmed_gold = gold[:active_length]

        for predicted_id, gold_id in zip(trimmed_prediction, trimmed_gold):
            total_tokens += 1
            if predicted_id == gold_id:
                correct_tokens += 1

        predicted_labels = decode_labels(trimmed_prediction, id_to_label)
        gold_labels = decode_labels(trimmed_gold, id_to_label)
        predicted_entities = extract_spans(predicted_labels)
        gold_entities = extract_spans(gold_labels)
        predicted_spans += len(predicted_entities)
        gold_spans += len(gold_entities)
        correct_spans += len(predicted_entities & gold_entities)

        for _, _, entity_label in predicted_entities:
            per_label_counts.setdefault(
                entity_label,
                {"predicted_spans": 0, "gold_spans": 0, "correct_spans": 0},
            )["predicted_spans"] += 1

        for _, _, entity_label in gold_entities:
            per_label_counts.setdefault(
                entity_label,
                {"predicted_spans": 0, "gold_spans": 0, "correct_spans": 0},
            )["gold_spans"] += 1

        for _, _, entity_label in predicted_entities & gold_entities:
            per_label_counts.setdefault(
                entity_label,
                {"predicted_spans": 0, "gold_spans": 0, "correct_spans": 0},
            )["correct_spans"] += 1

    precision = correct_spans / predicted_spans if predicted_spans else 0.0
    recall = correct_spans / gold_spans if gold_spans else 0.0
    span_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    per_label_metrics: dict[str, dict[str, float | int]] = {}
    for entity_label, counts in sorted(per_label_counts.items()):
        label_precision = (
            counts["correct_spans"] / counts["predicted_spans"]
            if counts["predicted_spans"]
            else 0.0
        )
        label_recall = (
            counts["correct_spans"] / counts["gold_spans"]
            if counts["gold_spans"]
            else 0.0
        )
        label_f1 = (
            2 * label_precision * label_recall / (label_precision + label_recall)
            if label_precision + label_recall > 0
            else 0.0
        )
        per_label_metrics[entity_label] = {
            **counts,
            "precision": label_precision,
            "recall": label_recall,
            "span_f1": label_f1,
        }

    return {
        "token_accuracy": correct_tokens / total_tokens if total_tokens else 0.0,

        "span_precision": precision,
        "span_recall": recall,
        "span_f1": span_f1,
        "per_label": per_label_metrics,
    }


def run_epoch(
    model: HybridBiLstmCrf,
    dataloader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    aux_ce_weight: float = 0.0,
    class_weights: torch.Tensor | None = None,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        batch = {
            key: value.to(device, non_blocking=device.type == "cuda")
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)

        # Get emissions, then compute CRF loss
        emissions = model(
            batch["token_ids"],
            batch["char_ids"],
            batch["bracket_ids"],
            batch["position_ids"],
            batch["lexicon_ids"],
            batch["mask"],
        )
        loss = model.crf(emissions, batch["labels"], batch["mask"])

        if aux_ce_weight > 0.0:
            # Auxiliary per-token CE loss on emission scores (bypasses CRF)
            # This gives direct gradient to the emission classifier for rare labels
            active = batch["mask"].reshape(-1)
            active_em = emissions.reshape(-1, emissions.size(-1))[active]
            active_labels = batch["labels"].reshape(-1)[active]
            w = class_weights.to(device) if class_weights is not None else None
            ce_loss = F.cross_entropy(active_em, active_labels, weight=w)
            loss = loss + aux_ce_weight * ce_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def evaluate(
    model: HybridBiLstmCrf,
    dataloader: DataLoader,
    device: torch.device,
    id_to_label: dict[int, str],
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    predictions: list[list[int]] = []
    labels: list[list[int]] = []
    masks: list[list[bool]] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in batch.items()
            }
            loss = model(
                batch["token_ids"],
                batch["char_ids"],
                batch["bracket_ids"],
                batch["position_ids"],
                batch["lexicon_ids"],
                batch["mask"],
                batch["labels"],
            )
            total_loss += loss.item()

            decoded = model.decode(
                batch["token_ids"],
                batch["char_ids"],
                batch["bracket_ids"],
                batch["position_ids"],
                batch["lexicon_ids"],
                batch["mask"],
            )
            predictions.extend(decoded)
            labels.extend(batch["labels"].cpu().tolist())
            masks.extend(batch["mask"].cpu().tolist())

    metrics = compute_metrics(predictions, labels, masks, id_to_label)
    return total_loss / max(1, len(dataloader)), metrics


def save_artifacts(
    output_dir: Path,
    model: HybridBiLstmCrf,
    token_vocab: dict[str, int],
    char_vocab: dict[str, int],
    label_vocab: dict[str, int],
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")

    (output_dir / "token_vocab.json").write_text(
        json.dumps(token_vocab, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "char_vocab.json").write_text(
        json.dumps(char_vocab, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "label_vocab.json").write_text(
        json.dumps(label_vocab, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "data": str(args.data),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "min_token_freq": args.min_token_freq,
                "max_char_len": args.max_char_len,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hybrid BiLSTM-CRF parser")
    parser.add_argument(
        "--data",
        default="data/training/hybrid_dataset.jsonl",
        type=Path,
        help="Hybrid dataset JSONL generated by tools/data/generate_hybrid_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("models/hybrid_bilstm_crf"),
        type=Path,
        help="Directory for model and vocab artifacts",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--min-token-freq", type=int, default=2)
    parser.add_argument("--max-char-len", type=int, default=24)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--aux-ce-weight",
        type=float,
        default=0.5,
        help="Weight for auxiliary per-token cross-entropy loss alongside CRF loss (0=disabled)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading records from {args.data}...", flush=True)
    records = load_dataset(args.data, args.max_samples)
    if not records:
        raise SystemExit(f"No training records found in {args.data}")
    print(f"Loaded {len(records):,} records", flush=True)

    print("Splitting train/validation sets...", flush=True)
    train_records, validation_records = split_records(records, args.validation_ratio, args.seed)
    print(
        f"Training samples: {len(train_records):,} | Validation samples: {len(validation_records):,}",
        flush=True,
    )

    print("Building vocabularies...", flush=True)
    token_vocab, char_vocab, label_vocab = build_vocab(train_records, args.min_token_freq)
    id_to_label = {index: label for label, index in label_vocab.items()}
    print(
        f"Token vocab: {len(token_vocab):,} | Char vocab: {len(char_vocab):,} | Labels: {len(label_vocab):,}",
        flush=True,
    )

    # Inverse-frequency class weights (capped at 50x to prevent extreme values)
    label_token_counts: Counter[str] = Counter()
    for rec in train_records:
        label_token_counts.update(rec["tags"])
    total_label_tokens = sum(label_token_counts.values())
    num_classes = len(label_vocab)
    class_weights = torch.ones(num_classes)
    for label, idx in label_vocab.items():
        count = label_token_counts.get(label, 0)
        if count > 0:
            class_weights[idx] = min(50.0, total_label_tokens / (num_classes * count))
    print(
        f"Class weights range: {class_weights.min().item():.2f} – {class_weights.max().item():.2f}",
        flush=True,
    )

    print("Encoding training dataset...", flush=True)
    train_dataset = HybridDataset(
        train_records, token_vocab, char_vocab, label_vocab, args.max_char_len
    )
    print("Encoding validation dataset...", flush=True)
    validation_dataset = HybridDataset(
        validation_records, token_vocab, char_vocab, label_vocab, args.max_char_len
    )

    device = torch.device(args.device)
    configure_torch_runtime(device)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        **loader_kwargs,
    )

    print(f"Using device: {device}", flush=True)
    model = HybridBiLstmCrf(
        vocab_size=len(token_vocab),
        char_vocab_size=len(char_vocab),
        num_tags=len(label_vocab),
    ).to(device)
    model.crf.apply_bio_constraints(label_vocab)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    print("Starting training...", flush=True)

    best_metric = -1.0
    best_metrics: dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, device,
            aux_ce_weight=args.aux_ce_weight,
            class_weights=class_weights,
        )
        validation_loss, metrics = evaluate(model, validation_loader, device, id_to_label)

        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} token_acc={metrics['token_accuracy']:.4f} "
            f"span_f1={metrics['span_f1']:.4f}"
        )

        if metrics["span_f1"] >= best_metric:
            best_metric = metrics["span_f1"]
            best_metrics = metrics
            save_artifacts(
                args.output_dir,
                model,
                token_vocab,
                char_vocab,
                label_vocab,
                metrics,
                args,
            )

    print("Best validation metrics:")
    print(json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()
