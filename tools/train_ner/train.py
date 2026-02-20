import os
import json
import random
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

MODEL_NAME = "distilbert-base-uncased"
# Save directly to models/ner_model so Rust can load it immediately
OUTPUT_DIR = "models/ner_model"

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
            data.append(obj)
    return data

def hard_negative_augmentation(example: Dict) -> Dict:
    # Introduce "Session", "Part", "Version" noise into episode numbers
    tokens = example["tokens"]
    tags = example["ner_tags"]
    
    new_tokens = []
    new_tags = []
    
    changed = False
    for t, tag in zip(tokens, tags):
        if tag == "B-EPISODE" and random.random() < 0.3:
            confusing_prefix = random.choice(["Session", "Part", "Ver", "Volume", "Vol"])
            new_tokens.append(confusing_prefix)
            new_tags.append("O")
            changed = True
        
        if tag == "O" and t.lower() in ["movie", "special"] and random.random() < 0.3:
            new_tokens.append(t)
            new_tags.append("O")
            new_tokens.append(random.choice(["2.0", "3.0", "1.11"]))
            new_tags.append("O")
            changed = True
            continue

        new_tokens.append(t)
        new_tags.append(tag)
        
    return {"tokens": new_tokens, "ner_tags": new_tags} if changed else None

def prepare_dataset(data: List[Dict]):
    augmented = []
    for ex in data:
        aug = hard_negative_augmentation(ex)
        if aug:
            augmented.append(aug)
            
    all_data = data + augmented
    random.shuffle(all_data)
    
    for ex in all_data:
        ex["ner_tags"] = [tag2id.get(t, tag2id["O"]) for t in ex["ner_tags"]]
        
    return Dataset.from_list(all_data)

def main():
    data_path = "data/training/silver_dataset.jsonl"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
        
    data = load_jsonl(data_path)
    train_dataset = prepare_dataset(data)
    
    split = train_dataset.train_test_split(test_size=0.1)
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
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=45,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=5,
    )
    
    collator = DataCollatorForTokenClassification(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
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
