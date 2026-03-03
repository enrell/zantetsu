import os
import json
import random
from pathlib import Path
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)

MODEL_NAME = "distilbert-base-uncased"
# Save directly to models/ner_model so Rust can load it immediately
OUTPUT_DIR = "models/ner_model"
RANDOM_SEED = 42
MIN_TRAINING_EXAMPLES = 4000

# Define TAG set matching Rust bio_tags.rs BioTag::index() ordering EXACTLY:
#   0: B-TITLE, 1: I-TITLE, 2: B-GROUP, 3: I-GROUP,
#   4: B-EPISODE, 5: I-EPISODE, 6: B-SEASON, 7: I-SEASON,
#   8: RESOLUTION, 9: VCODEC, 10: ACODEC, 11: SOURCE,
#   12: YEAR, 13: CRC32, 14: EXTENSION, 15: VERSION, 16: O
# WARNING: Do NOT move O to index 0. Rust bio_tags.rs has O=16.
TAGS = [
    "B-TITLE", "I-TITLE", "B-GROUP", "I-GROUP", "B-EPISODE", "I-EPISODE",
    "B-SEASON", "I-SEASON", "RESOLUTION", "VCODEC", "ACODEC", "SOURCE",
    "YEAR", "CRC32", "EXTENSION", "VERSION", "O"
]

tag2id = {t: i for i, t in enumerate(TAGS)}
id2tag = {i: t for i, t in enumerate(TAGS)}

def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            # Drop empty tokens and preserve alignment
            filtered = [
                (tok, tag)
                for tok, tag in zip(obj.get("tokens", []), obj.get("ner_tags", []))
                if str(tok).strip()
            ]
            if not filtered:
                continue
            obj["tokens"] = [t for t, _ in filtered]
            obj["ner_tags"] = [tag for _, tag in filtered]
            data.append(obj)
    return data

def augment_example(example: Dict) -> Dict:
    tokens = list(example["tokens"])
    tags = list(example["ner_tags"])

    if len(tokens) != len(tags) or not tokens:
        return example

    # Resolution replacements keep the same tag semantics
    for idx, (tok, tag) in enumerate(zip(tokens, tags)):
        if tag == "RESOLUTION":
            tokens[idx] = random.choice(["480p", "720p", "1080p", "2160p"])

    # Add harmless noise tokens as Outside class
    noise_candidates = ["batch", "v2", "multi", "dual", "uncensored", "remux"]
    if random.random() < 0.7:
        insert_at = random.randint(0, len(tokens))
        tokens.insert(insert_at, random.choice(noise_candidates))
        tags.insert(insert_at, "O")

    if random.random() < 0.4:
        insert_at = random.randint(0, len(tokens))
        tokens.insert(insert_at, str(random.randint(2000, 2026)))
        tags.insert(insert_at, "YEAR")

    return {"tokens": tokens, "ner_tags": tags}


def expand_corpus(data: List[Dict], target_size: int) -> List[Dict]:
    if not data:
        return data

    expanded = list(data)
    while len(expanded) < target_size:
        base = random.choice(data)
        expanded.append(augment_example(base))

    random.shuffle(expanded)
    return expanded

def prepare_dataset(data: List[Dict]):
    random.seed(RANDOM_SEED)

    all_data = expand_corpus(data, max(MIN_TRAINING_EXAMPLES, len(data) * 6))
    random.shuffle(all_data)
    
    for ex in all_data:
        ex["ner_tags"] = [tag2id.get(t, tag2id["O"]) for t in ex["ner_tags"]]
        
    return Dataset.from_list(all_data)


def resolve_data_path() -> Path | None:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]

    candidates = [
        Path("data/training/silver_dataset.jsonl"),
        repo_root / "data" / "training" / "silver_dataset.jsonl",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n  - ".join(str(p) for p in candidates)
    print("Data file not found: data/training/silver_dataset.jsonl")
    print("Checked these locations:\n  - " + checked)
    print("\nGenerate the dataset, then rerun training:")
    print("  1) Ensure input exists at data/training/nyaa_titles_5000_raw.txt")
    print("  2) Run: cargo run -p zantetsu-trainer --bin bootstrap_dataset")
    print("  3) Run: uv run .\\tools\\train_ner\\train.py")
    return None

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    data_path = resolve_data_path()
    if data_path is None:
        return
        
    data = load_jsonl(str(data_path))
    print(f"Loaded raw examples: {len(data)}")
    train_dataset = prepare_dataset(data)
    print(f"Expanded training examples: {len(train_dataset)}")
    
    split = train_dataset.train_test_split(test_size=0.1, seed=RANDOM_SEED)
    train_dataset = split["train"]
    val_dataset = split["test"]
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True, max_length=128
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    l = label[word_idx]
                    if id2tag[l].startswith("B-"):
                        l = tag2id["I-" + id2tag[l][2:]]
                    label_ids.append(l)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
        
    train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)
    
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(TAGS), 
        id2label=id2tag, 
        label2id=tag2id
    )
    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        num_train_epochs=12,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=5,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=RANDOM_SEED,
    )
    
    collator = DataCollatorForTokenClassification(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    print("Starting training...")
    trainer.train()
    
    # Export to safetensors for candle
    print("Exporting model to safetensors...")
    os.makedirs("models/ner_model", exist_ok=True)
    model.save_pretrained("models/ner_model", safe_serialization=True)
    tokenizer.save_pretrained("models/ner_model")
    print("Export complete.")

if __name__ == "__main__":
    main()
