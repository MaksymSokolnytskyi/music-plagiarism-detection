"""Evaluation statistics and manifest handling."""

from __future__ import annotations

import numpy as np
import pytest

from musicplag import data, metrics


def test_auc_matches_sklearn_including_ties() -> None:
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    s = np.round(rng.random(60) + 0.3 * y, 1)  # rounding creates ties
    assert metrics.auc(s, y) == pytest.approx(sk.roc_auc_score(y, s))


def test_summary_of_a_perfect_and_a_null_score() -> None:
    y = np.r_[np.ones(15), np.zeros(15)]
    perfect = metrics.summarize(np.r_[np.ones(15) + 1, np.zeros(15)] + y * 0, y)
    assert perfect.auc == 1.0 and perfect.p_value < 0.001
    null = metrics.summarize(np.random.default_rng(1).random(30), y)
    assert null.ci_low < 0.5 < null.ci_high


def test_cohens_d_sign() -> None:
    labels = [1, 1, 1, 0, 0, 0]
    assert metrics.cohens_d([2.0, 3.0, 4.0, 0.0, 1.0, 2.0], labels) == pytest.approx(2.0)
    assert metrics.cohens_d([0.0, 1.0, 2.0, 0.0, 1.0, 2.0], labels) == 0.0


def test_confusion_counts() -> None:
    assert metrics.confusion_at([0.9, 0.2, 0.8, 0.1], [1, 1, 0, 0], 0.5) == {
        "tp": 1,
        "fp": 1,
        "tn": 1,
        "fn": 1,
    }


def test_title_normalisation_and_leakage_check() -> None:
    raw = "audio/acdc_acdc_-_thunderstruck_(z3.fm).wav"
    assert data.normalize_title(raw) == "acdc acdc thunderstruck"
    train = [data.Pair(original="a", suspect="b", positive=True)]
    test = [data.Pair(original="b", suspect="c", positive=False)]
    assert data.shared_tracks(train, test) == {"b"}
