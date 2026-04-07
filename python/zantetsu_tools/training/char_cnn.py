#!/usr/bin/env python3
"""
Character-Level CNN for Anime Filename NER
==========================================
Train a lightweight character-level CNN for sequence labeling.
Uses RAD (Reinforcement Learning with Augmented Data) augmentations.

Architecture:
- Character embedding (128 dims)
- Multiple Conv1d layers with different kernel sizes (3, 5, 7)
- Highway layers for gating
- Bidirectional LSTM for sequence modeling
- Linear projection to BIO tags

Outputs ONNX model for Rust inference.
"""

import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

from ..common.torch_runtime import configure_torch_runtime


# BIO Tag vocabulary
TAG_TO_IDX = {
    "O": 0,
    "B-TITLE": 1,
    "I-TITLE": 2,
    "B-GROUP": 3,
    "I-GROUP": 4,
    "B-EPISODE": 5,
    "I-EPISODE": 6,
    "B-SEASON": 7,
    "I-SEASON": 8,
    "RESOLUTION": 9,
    "VCODEC": 10,
    "ACODEC": 11,
    "SOURCE": 12,
    "YEAR": 13,
    "CRC32": 14,
    "EXTENSION": 15,
    "VERSION": 16,
}
IDX_TO_TAG = {v: k for k, v in TAG_TO_IDX.items()}
NUM_TAGS = len(TAG_TO_IDX)

# Character vocabulary (95 printable ASCII chars)
# Indices: 0=PAD, 1-95=printable chars (ASCII 32-126), 96=UNK
CHAR_VOCAB = {chr(i): i - 30 for i in range(32, 127)}  # ASCII 32->1, ..., 126->95
CHAR_VOCAB["<PAD>"] = 0
CHAR_VOCAB["<UNK>"] = 96
NUM_CHARS = 97

MAX_LEN = 256


def iter_tensor_batches(
    chars: torch.Tensor,
    tags: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
):
    """Yield batches directly from already-stacked tensors."""
    num_samples = chars.size(0)
    if num_samples == 0:
        return

    if shuffle:
        indices = torch.randperm(num_samples, device=chars.device)
    else:
        indices = torch.arange(num_samples, device=chars.device)

    for start in range(0, num_samples, batch_size):
        batch_indices = indices[start : start + batch_size]
        yield chars[batch_indices], tags[batch_indices]


def pad_seq(seq: List[int], max_len: int) -> List[int]:
    """Pad sequence to max_len."""
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [CHAR_VOCAB["<PAD>"]] * (max_len - len(seq))


def char_encode(text: str) -> List[int]:
    """Encode text as character indices."""
    indices = [CHAR_VOCAB.get(c, CHAR_VOCAB["<UNK>"]) for c in text[: MAX_LEN - 2]]
    return pad_seq(indices, MAX_LEN)


def parse_char_tags(char_tags: List[str], filename: str) -> List[int]:
    """Convert char_tags strings to tag indices, handling padding."""
    result = []
    for tag in char_tags:
        result.append(TAG_TO_IDX.get(tag, TAG_TO_IDX["O"]))
    return pad_seq(result, MAX_LEN)


@dataclass
class TrainingSample:
    filename: str
    char_indices: List[int]
    tags: List[int]


class AnimeFilenameDataset(Dataset):
    """Dataset for anime filename NER."""

    def __init__(self, filepath: str, max_samples: Optional[int] = None):
        self.samples: List[TrainingSample] = []
        self._load_data(filepath, max_samples)

    def _load_data(self, filepath: str, max_samples: Optional[int]):
        print(f"Loading data from {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                if i % 50000 == 0 and i > 0:
                    print(f"  Loaded {i} samples...")
                data = json.loads(line)
                filename = data["filename"]
                char_indices = char_encode(filename)
                tags = parse_char_tags(data["char_tags"], filename)
                self.samples.append(
                    TrainingSample(
                        filename=filename, char_indices=char_indices, tags=tags
                    )
                )
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        return (
            torch.tensor(sample.char_indices, dtype=torch.long),
            torch.tensor(sample.tags, dtype=torch.long),
        )


def build_training_tensors(
    base_dataset: AnimeFilenameDataset,
    augmenter: "RADAugmenter",
    aug_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare stacked CPU tensors once, with optional augmentation progress logs."""
    total_samples = len(base_dataset.samples)
    started_at = time.perf_counter()

    print("Stacking base tensors...")
    all_chars = np.asarray(
        [sample.char_indices for sample in base_dataset.samples],
        dtype=np.int64,
    )
    all_tags = np.asarray(
        [sample.tags for sample in base_dataset.samples],
        dtype=np.int64,
    )
    print(f"Base tensors ready in {time.perf_counter() - started_at:.1f}s")

    if aug_ratio > 0:
        print(f"Applying RAD augmentation (ratio={aug_ratio:.2f})...")
        augmented_count = 0
        augment_started_at = time.perf_counter()

        for idx, sample in enumerate(base_dataset.samples, start=1):
            if sample.filename and random.random() < aug_ratio:
                all_chars[idx - 1] = np.asarray(
                    char_encode(augmenter.augment(sample.filename)),
                    dtype=np.int64,
                )
                augmented_count += 1

            if idx % 10000 == 0 or idx == total_samples:
                elapsed = time.perf_counter() - augment_started_at
                print(
                    f"  Augmented {idx}/{total_samples} samples in {elapsed:.1f}s "
                    f"({augmented_count} modified)"
                )

    return torch.from_numpy(all_chars), torch.from_numpy(all_tags)


class CharacterCNN(nn.Module):
    """
    Character-level CNN with highway layers and BiLSTM.

    Args:
        num_chars: Size of character vocabulary
        char_dim: Character embedding dimension
        cnn_filters: List of filter sizes for conv layers
        num_filters: Number of filters per size
        hidden_dim: BiLSTM hidden dimension
        num_layers: Number of BiLSTM layers
        num_tags: Number of BIO tags
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_chars: int = NUM_CHARS,
        char_dim: int = 128,
        cnn_filters: List[int] = None,
        num_filters: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_tags: int = NUM_TAGS,
        dropout: float = 0.3,
    ):
        super().__init__()

        if cnn_filters is None:
            cnn_filters = [3, 5, 7]

        self.char_dim = char_dim
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(num_chars, char_dim, padding_idx=0)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(char_dim, num_filters, kernel_size=k, padding=k // 2)
                for k in cnn_filters
            ]
        )

        num_conv_outputs = num_filters * len(cnn_filters)

        self.highway_input = nn.Linear(num_conv_outputs, num_conv_outputs)
        self.highway_gate = nn.Linear(num_conv_outputs, num_conv_outputs)

        lstm_input_dim = num_conv_outputs
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len] character indices
        Returns:
            emissions: [batch_size, seq_len, num_tags]
        """
        char_mask = (x != 0).float()

        embedded = self.embedding(x)
        embedded = embedded.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            h = F.relu(conv(embedded))
            conv_outputs.append(h)

        concatenated = torch.cat(conv_outputs, dim=1)
        concatenated = concatenated.transpose(1, 2)

        highway_out = self.highway(concatenated)

        lstm_out, _ = self.lstm(highway_out)
        lstm_out = self.dropout(lstm_out)

        emissions = self.classifier(lstm_out)

        mask_expanded = char_mask.unsqueeze(-1)
        emissions = emissions * mask_expanded + (1 - mask_expanded) * (-1e9)

        return emissions

    def highway(self, x: torch.Tensor) -> torch.Tensor:
        """Highway layer for information flow."""
        t = torch.sigmoid(self.highway_gate(x))
        return t * F.relu(self.highway_input(x)) + (1 - t) * x


class CrfLayer(nn.Module):
    """CRF layer for sequence labeling."""

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        nn.init.xavier_uniform_(self.transitions)
        nn.init.zeros_(self.start_transitions)
        nn.init.zeros_(self.end_transitions)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute negative log likelihood.

        Args:
            emissions: [batch_size, seq_len, num_tags]
            tags: [batch_size, seq_len]
            mask: [batch_size, seq_len]
        """
        score = self._compute_score(emissions, tags, mask)
        log_normalizer = self._compute_normalizer(emissions, mask)
        log_likelihood = score - log_normalizer

        return -log_likelihood.mean(), log_likelihood.mean()

    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute unnormalized score for the gold path."""
        batch_size, seq_len = tags.shape

        # Start transition
        score = self.start_transitions[tags[:, 0]]
        # First emission
        score = score + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)

        for i in range(1, seq_len):
            # Transition from tag[i-1] to tag[i]
            transition = self.transitions[tags[:, i - 1], tags[:, i]]
            # Emission at position i
            emission = emissions[:, i].gather(1, tags[:, i : i + 1]).squeeze(1)
            # Only add for non-PAD positions
            score = score + (transition + emission) * mask[:, i]

        # End transition for last real position
        last_real = mask.sum(dim=1).long() - 1
        last_tag = tags.gather(1, last_real.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tag] * (last_real >= 0).float()

        return score

    def _compute_normalizer(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log-sum-exp over all possible tag sequences."""
        batch_size, seq_len, num_tags = emissions.shape

        # Initialize with start transitions + first emission
        score = self.start_transitions + emissions[:, 0]  # [batch, num_tags]

        for i in range(1, seq_len):
            # score: [batch, num_tags_prev]
            # expand: [batch, num_tags_prev, 1]
            # transitions: [num_tags_prev, num_tags] -> [1, num_tags_prev, num_tags]
            # emissions[:, i]: [batch, num_tags] -> [batch, 1, num_tags]
            broadcast_score = score.unsqueeze(2)  # [batch, tags, 1]
            broadcast_transitions = self.transitions.unsqueeze(0)  # [1, tags, tags]
            broadcast_emissions = emissions[:, i].unsqueeze(1)  # [batch, 1, tags]

            next_score = broadcast_score + broadcast_transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)  # [batch, tags]

            # Mask padding positions
            mask_i = mask[:, i].unsqueeze(1)  # [batch, 1]
            score = next_score * mask_i + score * (1 - mask_i)

        # Add end transitions
        score = score + self.end_transitions  # [batch, tags]

        return torch.logsumexp(score, dim=1)  # [batch]


class CharCnnCrf(nn.Module):
    """Character CNN + CRF for sequence labeling."""

    def __init__(self, use_crf: bool = True):
        super().__init__()
        self.cnn = CharacterCNN()
        self.use_crf = use_crf
        if use_crf:
            self.crf = CrfLayer(NUM_TAGS)

    def forward(
        self,
        x: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len] character indices
            tags: [batch_size, seq_len] (required for training)
            mask: [batch_size, seq_len] (required for training)
        """
        emissions = self.cnn(x)

        if tags is not None and mask is not None:
            if self.use_crf:
                loss, _ = self.crf(emissions, tags, mask)
                return loss
            else:
                loss = F.cross_entropy(
                    emissions.view(-1, emissions.size(-1)),
                    tags.view(-1),
                    ignore_index=0,
                )
                return loss

        return emissions

    def decode(self, x: torch.Tensor) -> List[List[int]]:
        """Viterbi or greedy decoding."""
        emissions = self.cnn(x)
        if self.use_crf:
            return viterbi_decode(
                emissions,
                self.crf.transitions,
                self.crf.start_transitions,
                self.crf.end_transitions,
            )
        else:
            return emissions.argmax(dim=-1).tolist()


def viterbi_decode(
    emissions: torch.Tensor,
    transitions: torch.Tensor,
    start_transitions: torch.Tensor,
    end_transitions: torch.Tensor,
) -> List[List[int]]:
    """Viterbi decoding for CRF."""
    batch_size, seq_len, num_tags = emissions.shape

    score = start_transitions + emissions[:, 0, :]
    history = []

    for i in range(1, seq_len):
        broadcast_score = score.unsqueeze(2)
        broadcast_transitions = transitions.unsqueeze(0)

        next_score = broadcast_score + broadcast_transitions
        next_score, indices = next_score.max(dim=1)

        emission_score = emissions[:, i, :]
        score = next_score + emission_score
        history.append(indices)

    score = score + end_transitions
    _, best_last_tags = score.max(dim=1)

    best_tags = [best_last_tags.tolist()]
    for hist in reversed(history):
        best_last_tags = hist.gather(1, best_last_tags.unsqueeze(1)).squeeze(1)
        best_tags.insert(0, best_last_tags.tolist())

    return [list(sequence) for sequence in zip(*best_tags)]


class RADAugmenter:
    """
    RAD (Reinforcement Learning with Augmented Data) augmentations.
    Applies random augmentations to input text for data diversity.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def augment(self, text: str) -> str:
        """Apply random augmentation to filename."""
        augs = [
            self.random_char_swap,
            self.random_char_mask,
            self.random_char_substitute,
            self.random_char_insert,
            self.random_char_delete,
            self.random_case_flip,
            self.random_spacing_variation,
        ]

        num_augs = random.randint(1, len(augs))
        selected_augs = random.sample(augs, num_augs)

        result = text
        for aug in selected_augs:
            result = aug(result)

        return result

    def random_char_swap(self, text: str) -> str:
        """Randomly swap adjacent characters."""
        if random.random() > self.p:
            return text

        chars = list(text)
        if len(chars) < 2:
            return text

        i = random.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)

    def random_char_mask(self, text: str) -> str:
        """Replace random character with space."""
        if random.random() > self.p:
            return text

        chars = list(text)
        if not chars:
            return text

        i = random.randint(0, len(chars) - 1)
        chars[i] = " "
        return "".join(chars)

    def random_char_substitute(self, text: str) -> str:
        """Substitute random character with similar-looking char."""
        if random.random() > self.p:
            return text

        similar = {
            "0": "O",
            "1": "l",
            "2": "Z",
            "5": "S",
            "a": "@",
            "e": "3",
            "i": "!",
            "o": "0",
            "s": "$",
            "l": "1",
            "b": "8",
            "g": "9",
        }

        chars = list(text)
        if not chars:
            return text

        i = random.randint(0, len(chars) - 1)
        if chars[i] in similar:
            chars[i] = similar[chars[i]]
        return "".join(chars)

    def random_char_insert(self, text: str) -> str:
        """Insert random character."""
        if random.random() > self.p:
            return text

        noise_chars = "_- .[]"
        chars = list(text)
        if not chars:
            return text

        i = random.randint(0, len(chars))
        c = random.choice(noise_chars)
        chars.insert(i, c)
        return "".join(chars)

    def random_char_delete(self, text: str) -> str:
        """Delete random character."""
        if random.random() > self.p:
            return text

        chars = list(text)
        if len(chars) <= 3:
            return text

        i = random.randint(0, len(chars) - 1)
        del chars[i]
        return "".join(chars)

    def random_case_flip(self, text: str) -> str:
        """Randomly flip character case."""
        if random.random() > self.p:
            return text

        chars = list(text)
        if not chars:
            return text

        i = random.randint(0, len(chars) - 1)
        chars[i] = chars[i].swapcase()
        return "".join(chars)

    def random_spacing_variation(self, text: str) -> str:
        """Add/remove spaces around delimiters."""
        if random.random() > self.p:
            return text

        result = text
        for delim in [" ", "-", "."]:
            if delim == " ":
                pattern = r"(\s+)" if delim == " " else r"(\s*\.\s*)"
            else:
                pattern = rf"(\s*\{delim}\s*)"

            if random.random() < 0.5:
                result = re.sub(pattern, delim, result)
            else:
                result = re.sub(pattern, f" {delim} ", result)

        return result.strip()


class AugmentedDataset(Dataset):
    """Dataset with RAD augmentations pre-applied."""

    def __init__(
        self, base_dataset: Dataset, augmenter: RADAugmenter, aug_ratio: float = 0.3
    ):
        self.base_dataset = base_dataset
        self.augmenter = augmenter
        self.aug_ratio = aug_ratio

        # Pre-augment once during construction (GPU won't starve)
        print("Pre-augmenting dataset...")
        self.samples = []
        for i in range(len(base_dataset)):
            char_indices, tags = base_dataset[i]
            if random.random() < aug_ratio:
                first_token = char_indices[0].item()
                if first_token != CHAR_VOCAB["<PAD>"]:
                    chars = [
                        chr(i - 1 + 32) if 1 <= i <= 95 else ""
                        for i in char_indices.tolist()
                    ]
                    text = "".join(chars).rstrip("\x00 ")
                    if text:
                        aug_text = augmenter.augment(text)
                        aug_indices = char_encode(aug_text)
                        aug_indices = pad_seq(aug_indices, MAX_LEN)
                        char_indices = torch.tensor(aug_indices, dtype=torch.long)
            self.samples.append((char_indices, tags))
        print(f"Pre-augmented {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def train_epoch(
    model: nn.Module,
    all_chars: torch.Tensor,
    all_tags: torch.Tensor,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    use_crf: bool = True,
    accum_steps: int = 1,
    epoch_index: int = 0,
) -> float:
    """Train for one epoch."""
    model.train()
    del use_crf

    total_loss = torch.zeros((), device=all_chars.device)
    num_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (char_indices, tags) in enumerate(
        iter_tensor_batches(all_chars, all_tags, batch_size, shuffle=True)
    ):
        if batch_idx == 0 and epoch_index == 0 and getattr(torch.version, "hip", None):
            print("Starting first ROCm batch (kernel compilation may take a while)...")

        mask = (char_indices != 0).float()

        loss = model(char_indices, tags, mask)
        loss = loss / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()

        total_loss = total_loss + (loss.detach() * accum_steps)
        num_batches += 1

    # Handle remaining gradients
    if num_batches % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return (total_loss / num_batches).item()


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            char_indices, tags = batch
            char_indices = char_indices.to(device)
            tags = tags.to(device)
            mask = char_indices != 0

            loss = model(char_indices, tags, mask)
            total_loss += loss.item()
            num_batches += 1

            preds = model.decode(char_indices)
            all_preds.extend(preds)
            all_labels.extend(tags.tolist())

    avg_loss = total_loss / num_batches

    accuracy = compute_accuracy(all_preds, all_labels, (char_indices != 0).tolist())

    return avg_loss, accuracy


def compute_accuracy(
    preds: List[List[int]], labels: List[List[int]], masks: List[List[bool]]
) -> Dict[str, float]:
    """Compute token-level and entity-level accuracy."""
    total_tokens = 0
    correct_tokens = 0
    total_entities = 0
    correct_entities = 0

    for pred_seq, label_seq, mask_seq in zip(preds, labels, masks):
        for pred, label, mask in zip(pred_seq, label_seq, mask_seq):
            if mask:
                total_tokens += 1
                if pred == label:
                    correct_tokens += 1

    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0

    return {
        "token_accuracy": token_acc,
    }


def export_to_onnx(model: nn.Module, output_path: str):
    """Export CNN to ONNX and save CRF parameters separately."""
    model.eval()

    # Save full model state dict (primary format)
    pt_path = output_path.replace(".onnx", ".pt")
    torch.save(model.state_dict(), pt_path)
    print(f"Full model saved to {pt_path}")

    # Save CRF parameters separately for Viterbi decoding
    if model.use_crf:
        crf_path = output_path.replace(".onnx", "_crf.pt")
        torch.save(
            {
                "transitions": model.crf.transitions.data,
                "start_transitions": model.crf.start_transitions.data,
                "end_transitions": model.crf.end_transitions.data,
            },
            crf_path,
        )
        print(f"CRF parameters saved to {crf_path}")

    # Try ONNX export (may fail with LSTM + dynamic shapes)
    try:
        import onnxscript  # noqa: F401

        cnn_module = model.cnn.to("cpu").eval()
        dummy_input = torch.randint(0, NUM_CHARS, (1, MAX_LEN), dtype=torch.long)
        onnx_path = output_path

        torch.onnx.export(
            cnn_module,
            dummy_input,
            onnx_path,
            input_names=["char_indices"],
            output_names=["emissions"],
            opset_version=18,
        )
        print(f"CNN exported to ONNX: {onnx_path}")
    except Exception as e:
        print(f"ONNX export skipped: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Character CNN for anime filename NER"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/training/rad_dataset_50k.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/char_cnn_crf.onnx",
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_char_cnn_crf.pt",
        help="Checkpoint path for saving/loading the best model",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--aug_ratio",
        type=float,
        default=0.4,
        help="Probability of applying RAD augmentation to a sample during tensor prep",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to load (for debugging)",
    )
    parser.add_argument(
        "--no_crf",
        action="store_true",
        help="Disable CRF layer (use cross-entropy only)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--export_only",
        action="store_true",
        help="Skip training and export the checkpoint specified by --checkpoint",
    )
    args = parser.parse_args()

    device = args.device
    checkpoint_path = Path(args.checkpoint)

    if args.export_only:
        print(f"Using device: {device}")
        print(f"Loading checkpoint from {checkpoint_path}...")
        model = CharCnnCrf(use_crf=not args.no_crf)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Exporting to ONNX: {output_path}")
        export_to_onnx(model, str(output_path))
        print("Done!")
        return

    print(f"Using device: {device}")
    print(f"Loading dataset from {args.data}...")
    load_started_at = time.perf_counter()

    base_dataset = AnimeFilenameDataset(args.data, max_samples=args.max_samples)
    print(f"Dataset load completed in {time.perf_counter() - load_started_at:.1f}s")

    augmenter = RADAugmenter(p=0.5)
    prep_started_at = time.perf_counter()
    all_chars, all_tags = build_training_tensors(
        base_dataset,
        augmenter,
        args.aug_ratio,
    )
    print(f"CPU tensor prep completed in {time.perf_counter() - prep_started_at:.1f}s")

    # Move to GPU once
    device_tensor = torch.device(device)
    configure_torch_runtime(device_tensor)
    transfer_started_at = time.perf_counter()
    all_chars = all_chars.to(device_tensor)
    all_tags = all_tags.to(device_tensor)
    print(f"GPU transfer completed in {time.perf_counter() - transfer_started_at:.1f}s")
    print(f"Tensors on GPU: chars={all_chars.shape}, tags={all_tags.shape}")

    print("Initializing model...")
    model = CharCnnCrf(use_crf=not args.no_crf).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(all_chars.size(0) / args.batch_size)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
    )

    # Gradient accumulation for effective larger batch
    accum_steps = max(1, 256 // args.batch_size)
    if accum_steps > 1:
        print(
            f"Gradient accumulation: {accum_steps} steps (effective batch: {args.batch_size * accum_steps})"
        )

    print(f"Training for {args.epochs} epochs...")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model,
            all_chars,
            all_tags,
            args.batch_size,
            optimizer,
            scheduler,
            use_crf=not args.no_crf,
            accum_steps=accum_steps,
            epoch_index=epoch,
        )

        print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best model (loss={best_loss:.4f})")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to ONNX: {output_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    export_to_onnx(model, str(output_path))

    print("Done!")


if __name__ == "__main__":
    main()
