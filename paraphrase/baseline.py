"""Baselines built on word-overlap features.

Eight features per pair: how many words, bigrams and trigrams the two sentences
share, how much their lengths differ, and how similar their word ORDER is. The
order features are the interesting ones on PAWS, where many non-paraphrases use
the same words in a different order.
"""
from __future__ import annotations

import re

import numpy as np

from .data import Pair

FEATURE_NAMES = [
    "unigram_jaccard", "bigram_jaccard", "trigram_jaccard", "shared_words",
    "length_difference", "lcs_ratio", "order_agreement", "same_first_word",
]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(words: list[str], n: int) -> set[tuple]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence: shared words kept in the same order."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def order_agreement(a: list[str], b: list[str]) -> float:
    """Of the pairs of shared words, the share that appear in the same order in both sentences.

    1.0 = same order, 0.0 = completely reversed. This is a Kendall-tau-style score.
    """
    pos_b = {}
    for i, w in enumerate(b):
        pos_b.setdefault(w, i)
    shared = [pos_b[w] for w in dict.fromkeys(a) if w in pos_b]
    pairs = [(i, j) for i in range(len(shared)) for j in range(i + 1, len(shared))]
    if not pairs:
        return 1.0
    return sum(shared[i] < shared[j] for i, j in pairs) / len(pairs)


def features(pair: Pair) -> list[float]:
    a, b = tokens(pair.s1), tokens(pair.s2)
    longest = max(len(a), len(b), 1)
    return [
        jaccard(set(a), set(b)),
        jaccard(ngrams(a, 2), ngrams(b, 2)),
        jaccard(ngrams(a, 3), ngrams(b, 3)),
        len(set(a) & set(b)) / longest,
        abs(len(a) - len(b)) / longest,
        lcs_length(a, b) / longest,
        order_agreement(a, b),
        float(bool(a) and bool(b) and a[0] == b[0]),
    ]


def feature_matrix(pairs: list[Pair]) -> np.ndarray:
    return np.array([features(p) for p in pairs])


def models():
    """The two baselines: Naive Bayes, and logistic regression for comparison."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "Naive Bayes": GaussianNB(),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }
