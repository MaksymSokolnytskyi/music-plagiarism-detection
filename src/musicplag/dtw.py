r"""Dynamic time warping: global alignment and subsequence search.

Given a local cost matrix :math:`C \in \mathbb{R}^{N\times M}` (here, cosine distance
between feature frames), the accumulated cost obeys (Sakoe & Chiba 1978; Müller,
*Fundamentals of Music Processing*, §3.2):

.. math::
    D[i, j] = C[i, j] + \min\{D[i-1, j],\; D[i, j-1],\; D[i-1, j-1]\},
    \qquad D[0, 0] = C[0, 0].

The DTW distance is :math:`D[N-1, M-1]`; dividing by the warping-path length gives a
per-step cost in the range of :math:`C`, which for non-negative features and cosine
distance is :math:`[0, 1]`.

Band constraint. The original v4 code limited cells to :math:`|i - j| \le W`. When the
two recordings have different lengths (a 1.3× tempo change is common) and
:math:`|N - M| > W`, the end cell falls outside the band and the distance is
:math:`+\infty`. Here the band follows the diagonal of the rectangle instead:
:math:`|i/(N-1) - j/(M-1)| \le w`.

Subsequence DTW (Müller §7.2.3) lets the query start and end anywhere in the
reference: :math:`D[0, j] = C[0, j]` for all :math:`j`, answer :math:`\min_j D[N-1, j]`.
That is the right model for a sample or a quoted hook inside a longer track.

Implementation: cells on one anti-diagonal :math:`i + j = d` depend only on earlier
anti-diagonals, so each diagonal is filled in one vectorised step —
:math:`O(N + M)` Python iterations instead of :math:`O(NM)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_EPS = 1e-12


@dataclass(frozen=True)
class Alignment:
    """Result of a DTW alignment.

    ``path`` has shape ``(2, L)``: row 0 indexes the query, row 1 the reference.
    """

    distance: float
    path: NDArray[np.int64]

    @property
    def normalized_distance(self) -> float:
        """Accumulated cost divided by path length."""
        return self.distance / self.path.shape[1]

    @property
    def similarity(self) -> float:
        """``1 - normalized_distance``; in [0, 1] for cosine cost on non-negative data."""
        return float(np.clip(1.0 - self.normalized_distance, 0.0, 1.0))

    @property
    def diagonal_fraction(self) -> float:
        """Share of steps that advance both sequences — 1.0 for a pure time shift."""
        steps = np.diff(self.path, axis=1)
        return float(np.mean((steps[0] == 1) & (steps[1] == 1))) if steps.size else 1.0


def cosine_cost(a: FloatArray, b: FloatArray) -> FloatArray:
    """Cosine distance between the columns of ``a`` ``(F, N)`` and ``b`` ``(F, M)``."""
    an = a / (np.linalg.norm(a, axis=0, keepdims=True) + _EPS)
    bn = b / (np.linalg.norm(b, axis=0, keepdims=True) + _EPS)
    return np.clip(1.0 - an.T @ bn, 0.0, 2.0)


def _band_mask(n: int, m: int, band: float) -> NDArray[np.bool_]:
    i = np.arange(n)[:, None] / max(n - 1, 1)
    j = np.arange(m)[None, :] / max(m - 1, 1)
    return np.abs(i - j) <= band


def _accumulate(cost: FloatArray, subsequence: bool) -> FloatArray:
    """Padded accumulated-cost matrix ``Dp`` of shape ``(N+1, M+1)``."""
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    if subsequence:
        acc[0, :] = 0.0
    for d in range(n + m - 1):
        i = np.arange(max(0, d - m + 1), min(n - 1, d) + 1)
        j = d - i
        best = np.minimum(np.minimum(acc[i, j + 1], acc[i + 1, j]), acc[i, j])
        acc[i + 1, j + 1] = cost[i, j] + best
    return acc


def _backtrack(acc: FloatArray, end_j: int, subsequence: bool) -> NDArray[np.int64]:
    i, j = acc.shape[0] - 1, end_j + 1
    path = [(i - 1, j - 1)]
    while i > 1 or (not subsequence and j > 1):
        candidates = (acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
        step = int(np.argmin(candidates))  # ties prefer the diagonal
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
        path.append((i - 1, j - 1))
    return np.asarray(path[::-1], dtype=np.int64).T


def dtw(cost: FloatArray, band: float | None = None) -> Alignment:
    """Global DTW over a cost matrix, optionally with a slope-following band."""
    cost = np.asarray(cost, dtype=np.float64)
    if band is not None:
        cost = np.where(_band_mask(*cost.shape, band), cost, np.inf)
    acc = _accumulate(cost, subsequence=False)
    distance = float(acc[-1, -1])
    assert np.isfinite(distance), "band too narrow: end cell unreachable"
    return Alignment(distance, _backtrack(acc, cost.shape[1] - 1, subsequence=False))


def subsequence_dtw(cost: FloatArray) -> Alignment:
    """Best match of the whole query (rows) inside any span of the reference (columns)."""
    cost = np.asarray(cost, dtype=np.float64)
    acc = _accumulate(cost, subsequence=True)
    end_j = int(np.argmin(acc[-1, 1:]))
    distance = float(acc[-1, end_j + 1])
    return Alignment(distance, _backtrack(acc, end_j, subsequence=True))
