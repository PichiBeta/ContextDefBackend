import os
import argparse
import logging
import json
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from datasets import load_dataset

log = logging.getLogger(__name__)

LABEL_MAP = {"A1":0, "A2":1, "B1":2, "B2":3, "C1":4, "C2":5}

HF_TOKEN = os.getenv("HF_TOKEN")
REPO = "pichibeta/difficulty_prediction_SPA"

tokenizer = AutoTokenizer.from_pretrained(REPO, token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(REPO, token=HF_TOKEN)

def load_tabular(path, text_col="text", label_col="label"):
    path = os.fspath(path)
    if path.endswith(".jsonl") or path.endswith(".json"):
        # try jsonlines or single json list
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        try:
            # one json per line
            rows = [json.loads(l) for l in lines]
        except Exception:
            # maybe a single JSON list
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Columns {text_col} and {label_col} must exist in {path}")
    texts = df[text_col].astype(str).tolist()
    labels_raw = df[label_col].tolist()
    labels = []
    for l in labels_raw:
        if isinstance(l, str):
            l = l.strip()
            if l in LABEL_MAP:
                labels.append(LABEL_MAP[l])
            else:
                try:
                    labels.append(int(l))
                except Exception:
                    raise ValueError(f"Unknown label value: {l}")
        else:
            labels.append(int(l))
    from collections import Counter
    dist = Counter(labels)
    print({k: dist[k] for k in sorted(dist)})
    return texts, labels


def tokenize_texts(tokenizer, texts, labels, max_length=512):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    encodings["labels"] = labels
    return encodings


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        return item


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


def text_to_score(model, tokenizer, text, device=None, max_length=512):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if device is None:
        device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze(0)
    print(probs)
    score_map = torch.tensor([0, 20, 40, 60, 80, 100], dtype=probs.dtype, device=probs.device)
    difficulty_score = (probs * score_map).sum().item()
    return difficulty_score



def main():
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # -------------------------------
    # Config
    # -------------------------------
    MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
    OUTPUT_DIR = "./best_model/beto"
    EPOCHS = 4
    BATCH_SIZE = 8
    MAX_LENGTH = 512
    LEARNING_RATE = 2e-5
    SEED = 42

    # -------------------------------
    # Load UniversalCEFR Spanish dataset
    # -------------------------------
    log.info("Loading UniversalCEFR Spanish dataset...")
    ds = load_dataset("UniversalCEFR/hablacultura_es")
    full = ds["train"]

    examples = [full[i] for i in range(len(full))]


    texts = [ex["text"] for ex in examples]
    labels_cefr = [ex["cefr_level"] for ex in examples]

    # Map CEFR levels to integers for classification
    CEFR_TO_INT = {"A1":0, "A2":1, "B": 2, "B1":2, "B2":3, "C": 4, "C1":4, "C2":5}
    labels = [CEFR_TO_INT[l] for l in labels_cefr]

    # -------------------------------
    # Tokenize and create PyTorch datasets
    # -------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    encodings = tokenize_texts(tokenizer, texts, labels, max_length=MAX_LENGTH)
    full_dataset = TextDataset(encodings)

    # Stratified train/validation split
    train_idx, val_idx = train_test_split(
        list(range(len(labels))), test_size=0.15, stratify=labels, random_state=SEED
    )
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    # -------------------------------
    # Load BETO model
    # -------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6)

    # -------------------------------
    # Training arguments
    # -------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # -------------------------------
    # Train
    # -------------------------------
    trainer.train()

    # Save model & tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # -------------------------------
    # Example inference
    # -------------------------------
    sample = "Este es un texto de ejemplo para evaluar la dificultad."
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    score = text_to_score(model, tokenizer, sample, device=device, max_length=MAX_LENGTH)
    log.info("Sample difficulty score: %.2f", score)

if __name__ == "__main__":
    ml_model = {}
    REPO = "pichibeta/difficulty_prediction_SPA"
    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(REPO, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(REPO, token=token)
    print(text_to_score(model, tokenizer, "La fenomenología de Heidegger postula una ontología fundamental."))
