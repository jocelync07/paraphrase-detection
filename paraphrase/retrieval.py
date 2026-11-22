"""Siamese Sentence-BERT: one encoder turns each sentence into a vector, and
cosine similarity between the vectors scores a pair.

Used two ways:
- Retrieval: for each paraphrase pair in the test set, search with sentence 1
  among every test sentence 2 and see where the true partner ranks.
- Classification: call a pair a paraphrase when its similarity is above a
  threshold chosen on the training data.

The encoder is fine-tuned with a contrastive loss that pulls paraphrases
together and pushes non-paraphrases apart.
"""
from __future__ import annotations

import numpy as np

from .data import Pair, labels
from .metrics import best_threshold, classification, retrieval

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def encode(model, sentences: list[str]) -> np.ndarray:
    return model.encode(sentences, batch_size=64, convert_to_numpy=True, normalize_embeddings=True,
                        show_progress_bar=False)


def pair_similarity(model, pairs: list[Pair]) -> np.ndarray:
    a = encode(model, [p.s1 for p in pairs])
    b = encode(model, [p.s2 for p in pairs])
    return (a * b).sum(axis=1)


def retrieval_scores(model, test: list[Pair]) -> dict:
    """Queries: sentence 1 of every paraphrase pair. Candidates: all test sentence 2s."""
    candidates = [p.s2 for p in test]
    queries = [(i, p.s1) for i, p in enumerate(test) if p.label == 1]
    q = encode(model, [s for _, s in queries])
    c = encode(model, candidates)
    return retrieval(q @ c.T, [i for i, _ in queries])


def evaluate(model, train: list[Pair], test: list[Pair]) -> dict:
    threshold = best_threshold(pair_similarity(model, train), labels(train))
    predictions = (pair_similarity(model, test) >= threshold).astype(int)
    sims = pair_similarity(model, test)
    y = np.array(labels(test))
    return {
        **retrieval_scores(model, test),
        **classification(y, predictions),
        "threshold": threshold,
        "similarity_gap": float(sims[y == 1].mean() - sims[y == 0].mean()),
    }


def fine_tune(train: list[Pair], epochs: int = 1, batch_size: int = 16, lr: float = 2e-5, seed: int = 42):
    """Contrastive fine-tuning with a plain PyTorch loop (no Trainer needed)."""
    import torch
    from sentence_transformers import SentenceTransformer, losses

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SentenceTransformer(BASE_MODEL, device="cpu")
    loss_fn = losses.OnlineContrastiveLoss(model)  # loss only on the hardest pairs in each batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        order, total = rng.permutation(len(train)), 0.0
        for i in range(0, len(order), batch_size):
            batch = [train[j] for j in order[i:i + batch_size]]
            features = [model.tokenize([p.s1 for p in batch]), model.tokenize([p.s2 for p in batch])]
            y = torch.tensor([float(p.label) for p in batch])
            loss = loss_fn(features, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()
        print(f"  Sentence-BERT epoch {epoch}: mean batch loss {total / max(1, len(order) // batch_size):.4f}", flush=True)
    model.eval()
    return model


def run(train: list[Pair], test: list[Pair], epochs: int = 1) -> dict:
    from sentence_transformers import SentenceTransformer

    before = evaluate(SentenceTransformer(BASE_MODEL, device="cpu"), train, test)
    after = evaluate(fine_tune(train, epochs), train, test)
    return {"before": before, "after": after}
