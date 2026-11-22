"""DistilBERT cross-encoder: both sentences go into the transformer together
("[CLS] sentence 1 [SEP] sentence 2 [SEP]"), so attention can compare every
word of one with every word of the other. This is what lets it notice that the
same words have swapped roles.
"""
from __future__ import annotations

import time

import numpy as np

from .data import Pair, labels
from .metrics import classification

BASE_MODEL = "distilbert-base-uncased"


def batches(pairs: list[Pair], size: int, rng: np.random.Generator | None = None):
    order = np.arange(len(pairs)) if rng is None else rng.permutation(len(pairs))
    for i in range(0, len(order), size):
        yield [pairs[j] for j in order[i:i + size]]


def run(train: list[Pair], test: list[Pair], epochs: int = 2, batch_size: int = 16, lr: float = 3e-5,
        max_length: int = 128, seed: int = 42) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    def encode(batch):
        enc = tokenizer([p.s1 for p in batch], [p.s2 for p in batch], truncation=True,
                        max_length=max_length, padding=True, return_tensors="pt")
        return enc, torch.tensor([p.label for p in batch])

    steps = epochs * ((len(train) + batch_size - 1) // batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        began, total, count = time.time(), 0.0, 0
        for batch in batches(train, batch_size, rng):
            enc, y = encode(batch)
            loss = model(**enc, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            total, count = total + loss.item() * len(batch), count + len(batch)
        test_scores = evaluate(model, encode, test)
        history.append({"epoch": epoch, "train_loss": total / count, "test_accuracy": test_scores["accuracy"],
                        "minutes": (time.time() - began) / 60})
        print(f"  epoch {epoch}: train loss {total / count:.4f}, test accuracy {test_scores['accuracy']:.4f} "
              f"({history[-1]['minutes']:.1f} min)", flush=True)
    return {**evaluate(model, encode, test), "history": history}


def evaluate(model, encode, pairs: list[Pair]) -> dict:
    import torch

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in batches(pairs, 64):
            enc, _ = encode(batch)
            predictions += model(**enc).logits.argmax(-1).tolist()
    return classification(labels(pairs), predictions)
