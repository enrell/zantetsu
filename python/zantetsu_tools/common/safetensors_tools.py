from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file

from .paths import MODEL_DIR


DEFAULT_MODEL_PATH = MODEL_DIR / "ner_model" / "model.safetensors"
DEFAULT_CONFIG_PATH = MODEL_DIR / "ner_model" / "config.json"
TRAIN_TAGS = [
    "O",
    "B-TITLE",
    "I-TITLE",
    "B-GROUP",
    "I-GROUP",
    "B-EPISODE",
    "I-EPISODE",
    "B-SEASON",
    "I-SEASON",
    "RESOLUTION",
    "VCODEC",
    "ACODEC",
    "SOURCE",
    "YEAR",
    "CRC32",
    "EXTENSION",
    "VERSION",
]


def iter_keys(model_path: Path = DEFAULT_MODEL_PATH) -> list[str]:
    with safe_open(model_path, framework="pt") as tensor_file:
        return list(tensor_file.keys())


def convert_layer_norm_keys(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    temp_path = model_path.with_suffix(".converted.safetensors")
    tensors: dict[str, Any] = {}

    with safe_open(model_path, framework="pt") as tensor_file:
        for key in tensor_file.keys():
            converted_key = key.replace(".gamma", ".weight").replace(".beta", ".bias")
            tensors[converted_key] = tensor_file.get_tensor(key)

    save_file(tensors, str(temp_path))
    temp_path.replace(model_path)
    return model_path


def inspect_ner_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    with safe_open(model_path, framework="pt") as tensor_file:
        classifier_weight = tensor_file.get_tensor("classifier.weight")
        classifier_bias = tensor_file.get_tensor("classifier.bias")
        keys = list(tensor_file.keys())

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    mismatches = []
    for index, train_tag in enumerate(TRAIN_TAGS):
        config_tag = config["id2label"].get(str(index), "MISSING")
        if train_tag != config_tag:
            mismatches.append(
                {
                    "index": index,
                    "expected": train_tag,
                    "actual": config_tag,
                }
            )

    return {
        "classifier_weight_shape": list(classifier_weight.shape),
        "classifier_bias_shape": list(classifier_bias.shape),
        "num_labels": classifier_weight.shape[0],
        "id2label": config["id2label"],
        "train_tags": TRAIN_TAGS,
        "mismatches": mismatches,
        "keys": keys,
    }
