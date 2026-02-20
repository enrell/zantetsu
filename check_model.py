#!/usr/bin/env python3
"""Check the NER model weights and verify the label mapping matches Rust BioTag ordering."""

import json
from safetensors import safe_open

# Check classifier shape
t = safe_open("models/ner_model/model.safetensors", framework="pt")
w = t.get_tensor("classifier.weight")
b = t.get_tensor("classifier.bias")
print(f"classifier.weight shape: {w.shape}")
print(f"classifier.bias shape: {b.shape}")
print(f"Number of labels in model: {w.shape[0]}")
print()

# Check config.json (what the trained model has)
with open("models/ner_model/config.json") as f:
    cfg = json.load(f)

print("config.json id2label (ordered by index):")
for k, v in sorted(cfg["id2label"].items(), key=lambda x: int(x[0])):
    print(f"  {k}: {v}")
print()

# Train.py TAGS order (this is what the model was TRAINED with)
TRAIN_TAGS = [
    "O", "B-TITLE", "I-TITLE", "B-GROUP", "I-GROUP", "B-EPISODE", "I-EPISODE",
    "B-SEASON", "I-SEASON", "RESOLUTION", "VCODEC", "ACODEC", "SOURCE",
    "YEAR", "CRC32", "EXTENSION", "VERSION"
]
print("train.py TAGS order (what labels correspond to during training):")
for i, t in enumerate(TRAIN_TAGS):
    print(f"  {i}: {t}")
print()

# Check what Rust BioTag expects (from bio_tags.rs)
print("Cross-checking config.json vs train.py for mismatches:")
for i, train_tag in enumerate(TRAIN_TAGS):
    config_tag = cfg["id2label"].get(str(i), "MISSING")
    match = "✓" if train_tag == config_tag else "✗ MISMATCH"
    if train_tag != config_tag:
        print(f"  Index {i}: train.py='{train_tag}' vs config.json='{config_tag}' {match}")

print()
# Check if the model keys match what Rust candle expects
print("All safetensor keys:")
st = safe_open("models/ner_model/model.safetensors", framework="pt")
for k in st.keys():
    print(f"  {k}")
