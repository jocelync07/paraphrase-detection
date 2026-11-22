"""Fast tests: no downloads, no model training."""
import numpy as np
import pytest

from paraphrase.baseline import FEATURE_NAMES, feature_matrix, features, jaccard, lcs_length, ngrams, order_agreement, tokens
from paraphrase.data import Pair
from paraphrase.metrics import best_threshold, classification, ranks, retrieval


def test_tokens_lowercase_and_drop_punctuation():
    assert tokens("Flights from New-York, to Florida!") == ["flights", "from", "new", "york", "to", "florida"]


def test_jaccard_and_ngrams():
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == 1.0
    assert ngrams(["a", "b", "c"], 2) == {("a", "b"), ("b", "c")}


def test_lcs_length():
    assert lcs_length(list("abcde"), list("ace")) == 3
    assert lcs_length(["x"], ["y"]) == 0


def test_order_agreement_detects_swaps():
    a = tokens("flights from new york to florida")
    assert order_agreement(a, a) == 1.0
    assert order_agreement(a, list(reversed(a))) == 0.0
    swapped = tokens("flights from florida to new york")
    assert 0.0 < order_agreement(a, swapped) < 1.0


def test_swapped_sentences_share_words_but_not_order():
    """The PAWS trap: identical word sets, different meaning."""
    same = features(Pair("Flights from New York to Florida", "Flights from New York to Florida", 1))
    swap = features(Pair("Flights from New York to Florida", "Flights from Florida to New York", 0))
    names = dict(zip(FEATURE_NAMES, zip(same, swap)))
    assert names["unigram_jaccard"] == (1.0, 1.0)      # bag of words cannot tell them apart
    assert names["order_agreement"][1] < 1.0           # order features can
    assert names["bigram_jaccard"][1] < 1.0


def test_feature_matrix_shape():
    pairs = [Pair("a b c", "a b d", 1), Pair("x", "y z", 0)]
    assert feature_matrix(pairs).shape == (2, len(FEATURE_NAMES))


def test_classification_metrics():
    m = classification([1, 1, 0, 0], [1, 0, 0, 0])
    assert m["accuracy"] == 0.75
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5


def test_ranks_and_retrieval():
    scores = np.array([[0.9, 0.1, 0.2],   # correct (0) ranks 1st
                       [0.8, 0.3, 0.9],   # correct (1) ranks 3rd
                       [0.1, 0.7, 0.6]])  # correct (2) ranks 2nd
    assert ranks(scores, [0, 1, 2]) == [1, 3, 2]
    r = retrieval(scores, [0, 1, 2], ks=(1, 2))
    assert r["recall@1"] == pytest.approx(1 / 3)
    assert r["recall@2"] == pytest.approx(2 / 3)
    assert r["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 2) / 3)


def test_best_threshold_separates_classes():
    sims = [0.1, 0.2, 0.8, 0.9]
    t = best_threshold(sims, [0, 0, 1, 1])
    assert 0.2 < t <= 0.8
