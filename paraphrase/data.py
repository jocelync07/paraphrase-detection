"""PAWS: sentence pairs that look alike but may or may not mean the same thing.

PAWS (Paraphrase Adversaries from Word Scrambling, Zhang et al. 2019) was built
by swapping words and phrases, so a non-paraphrase often uses almost exactly
the same words as its partner ("Flights from New York to Florida" vs "Flights
from Florida to New York"). Word overlap alone cannot solve it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Pair:
    s1: str
    s2: str
    label: int  # 1 = paraphrase, 0 = not


def load_paws(n_train: int = 5000, n_test: int = 1000, seed: int = 42) -> tuple[list[Pair], list[Pair]]:
    """A random, repeatable subset of the PAWS labeled_final train and test splits."""
    from datasets import load_dataset

    data = load_dataset("google-research-datasets/paws", "labeled_final")
    return (sample(data["train"], n_train, seed), sample(data["test"], n_test, seed))


def sample(split, n: int, seed: int) -> list[Pair]:
    rows = list(range(len(split)))
    random.Random(seed).shuffle(rows)
    picked = split.select(sorted(rows[:n])) if n < len(rows) else split
    return [Pair(r["sentence1"], r["sentence2"], int(r["label"])) for r in picked]


def labels(pairs: list[Pair]) -> list[int]:
    return [p.label for p in pairs]
