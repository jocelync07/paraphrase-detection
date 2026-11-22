"""Classification and retrieval metrics."""
from __future__ import annotations

import numpy as np


def classification(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def ranks(scores: np.ndarray, correct: list[int]) -> list[int]:
    """1-based rank of the correct candidate in each row of a query x candidate score matrix."""
    return [int((row > row[c]).sum()) + 1 for row, c in zip(scores, correct)]


def retrieval(scores: np.ndarray, correct: list[int], ks=(1, 5)) -> dict:
    r = np.array(ranks(scores, correct))
    out = {f"recall@{k}": float((r <= k).mean()) for k in ks}
    out["mrr"] = float((1.0 / r).mean())
    return out


def best_threshold(similarities, y_true) -> float:
    """The cosine-similarity cut-off that maximises accuracy (chosen on training data)."""
    sims = np.asarray(similarities)
    y = np.asarray(y_true)
    candidates = np.unique(sims)
    accuracy = [((sims >= t) == y).mean() for t in candidates]
    return float(candidates[int(np.argmax(accuracy))])
