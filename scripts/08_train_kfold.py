"""
Stratified 5-fold fine-tuning of MuRIL or mBERT on the final dataset.

Usage:
    python 08_train_kfold.py --model muril --data ../data/hinglish_veracity_568.csv --out ../results
    python 08_train_kfold.py --model mbert ...

Configuration matches the paper: class-weighted cross-entropy (weights
n / (2 * class_count) per training fold), AdamW, lr 3e-5, batch size 16,
12 epochs, max length 256, 10% linear warmup then linear decay, gradient
clipping at 1.0, StratifiedKFold(n_splits=5, shuffle=True, random_state=42).

Note: only the fold split is seeded. Model initialisation and batch order are
not, so re-running reproduces the protocol but not the exact per-fold numbers.
The predictions behind the paper's results are in results/*_kfold_results.json.
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

MODELS = {"muril": "google/muril-base-cased", "mbert": "bert-base-multilingual-cased"}
LABEL2ID = {"False": 0, "Not-Clearly-False": 1}
N_FOLDS, EPOCHS, BATCH_SIZE, LR, MAX_LENGTH = 5, 12, 16, 3e-5, 256


class HinglishDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts, self.labels = list(texts), list(labels)
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length",
                             max_length=self.max_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--data", default="../data/hinglish_veracity_568.csv")
    ap.add_argument("--out", default="../results")
    args = ap.parse_args()

    name = MODELS[args.model]
    df = pd.read_csv(args.data)
    texts = df["text"].values
    labels = df["binary_label"].map(LABEL2ID).values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    results = []
    for fold, (tr, va) in enumerate(skf.split(texts, labels), 1):
        print(f"\n=== {args.model} fold {fold}/{N_FOLDS} ===")
        train_ds = HinglishDataset(texts[tr], labels[tr], tokenizer)
        val_ds = HinglishDataset(texts[va], labels[va], tokenizer)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        counts = np.bincount(labels[tr])
        weights = torch.tensor([len(labels[tr]) / (2 * c) for c in counts], dtype=torch.float).to(device)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        total = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total), total)
        loss_fn = nn.CrossEntropyLoss(weight=weights)

        model.train()
        for epoch in range(EPOCHS):
            running = 0.0
            for b in train_loader:
                optimizer.zero_grad()
                out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
                loss = loss_fn(out.logits, b["labels"].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                running += loss.item()
            if (epoch + 1) % 3 == 0 or epoch == 0:
                print(f"  epoch {epoch + 1}/{EPOCHS} avg loss {running / len(train_loader):.4f}")

        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for b in val_loader:
                out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
                preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
                true.extend(b["labels"].numpy())
        f1 = f1_score(true, preds, average="macro")
        print(f"fold {fold} macro-F1 {f1:.4f}")
        print(classification_report(true, preds, target_names=list(LABEL2ID), zero_division=0))
        results.append({"fold": fold, "macro_f1": float(f1),
                        "preds": [int(p) for p in preds], "true": [int(t) for t in true]})
        del model
        torch.cuda.empty_cache()

    f1s = [r["macro_f1"] for r in results]
    print(f"\nmean macro-F1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    with open(f"{args.out}/{args.model}_kfold_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
