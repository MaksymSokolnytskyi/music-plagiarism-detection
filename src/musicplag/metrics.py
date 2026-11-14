r"""Evaluation statistics for small, imbalanced pair sets.

All headline numbers in this repository are AUCs with uncertainty, because with 14-66
pairs a single accuracy at a tuned threshold is mostly noise.

AUC as a rank statistic (Hanley & McNeil 1982): with midranks :math:`R` of the positive
scores among all :math:`n_1 + n_0` scores,

.. math::
    \mathrm{AUC} = \frac{\sum R_{pos} - n_1(n_1+1)/2}{n_1 n_0}
                 = \Pr(s^+ > s^-) + \tfrac12 \Pr(s^+ = s^-).

Uncertainty: a stratified percentile bootstrap (resample positives and negatives
separately, so every replicate has both classes) and a one-sided permutation test of
:math:`H_0: \mathrm{AUC} = 0.5` (labels exchangeable). Effect size: Cohen's :math:`d`
with pooled standard deviation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import rankdata

FloatArray = NDArray[np.float64]
N_BOOTSTRAP = 5000  # percentile CI resamples
N_PERMUTATIONS = 5000  # permutation-test label shuffles


def _split(scores: ArrayLike, labels: ArrayLike) -> tuple[FloatArray, FloatArray]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    assert s.shape == y.shape and y.any() and (~y).any(), "need both classes"
    assert np.all(np.isfinite(s)), "non-finite score"
    return s[y], s[~y]


def auc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties count 1/2)."""
    pos, neg = _split(scores, labels)
    ranks = rankdata(np.concatenate([pos, neg]))
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def cohens_d(scores: ArrayLike, labels: ArrayLike) -> float:
    """Standardised mean difference, positives minus negatives (pooled SD)."""
    pos, neg = _split(scores, labels)
    n1, n0 = len(pos), len(neg)
    pooled = np.sqrt(
        ((n1 - 1) * pos.var(ddof=1) + (n0 - 1) * neg.var(ddof=1)) / (n1 + n0 - 2)
    )
    return float((pos.mean() - neg.mean()) / pooled) if pooled > 0 else 0.0


class AucSummary(BaseModel):
    """AUC with a bootstrap interval and a permutation p-value."""

    model_config = ConfigDict(frozen=True)

    n_pos: int = Field(ge=1)
    n_neg: int = Field(ge=1)
    auc: float = Field(ge=0, le=1)
    ci_low: float = Field(ge=0, le=1, description="2.5th bootstrap percentile.")
    ci_high: float = Field(ge=0, le=1, description="97.5th bootstrap percentile.")
    p_value: float = Field(ge=0, le=1, description="One-sided, H0: AUC = 0.5.")
    cohens_d: float

    def line(self) -> str:
        """One human-readable line, e.g. for a report table."""
        return (
            f"AUC {self.auc:.3f} [{self.ci_low:.3f}, {self.ci_high:.3f}], "
            f"p={self.p_value:.3f}, d={self.cohens_d:+.2f} "
            f"({self.n_pos}+/{self.n_neg}-)"
        )


def summarize(scores: ArrayLike, labels: ArrayLike, seed: int = 0) -> AucSummary:
    """AUC, stratified bootstrap 95% CI, permutation p-value and Cohen's d."""
    pos, neg = _split(scores, labels)
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    point = auc(s, y)
    boot = np.array(
        [
            auc(
                np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))],
                y,
            )
            for _ in range(N_BOOTSTRAP)
        ]
    )
    perm = np.array([auc(s, rng.permutation(y)) for _ in range(N_PERMUTATIONS)])
    p_value = (1 + np.sum(perm >= point)) / (1 + N_PERMUTATIONS)
    return AucSummary(
        n_pos=len(pos),
        n_neg=len(neg),
        auc=point,
        ci_low=float(np.percentile(boot, 2.5)),
        ci_high=float(np.percentile(boot, 97.5)),
        p_value=float(p_value),
        cohens_d=cohens_d(s, y),
    )


def confusion_at(
    scores: ArrayLike, labels: ArrayLike, threshold: float
) -> dict[str, int]:
    """TP/FP/TN/FN when predicting positive for ``score >= threshold``."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    pred = s >= threshold
    return {
        "tp": int(np.sum(pred & y)),
        "fp": int(np.sum(pred & ~y)),
        "tn": int(np.sum(~pred & ~y)),
        "fn": int(np.sum(~pred & y)),
    }
