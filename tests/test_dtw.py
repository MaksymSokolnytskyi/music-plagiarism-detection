"""DTW against a brute-force recursion and known alignments."""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from musicplag import dtw


def _reference(cost: np.ndarray) -> float:
    @cache
    def d(i: int, j: int) -> float:
        if i == 0 and j == 0:
            return float(cost[0, 0])
        best = min(
            d(i - 1, j) if i > 0 else np.inf,
            d(i, j - 1) if j > 0 else np.inf,
            d(i - 1, j - 1) if i > 0 and j > 0 else np.inf,
        )
        return float(cost[i, j]) + best

    return d(cost.shape[0] - 1, cost.shape[1] - 1)


@pytest.mark.parametrize("shape", [(1, 1), (5, 5), (7, 12), (13, 4)])
def test_matches_textbook_recursion(shape: tuple[int, int]) -> None:
    cost = np.random.default_rng(sum(shape)).random(shape)
    result = dtw.dtw(cost)
    assert result.distance == pytest.approx(_reference(cost))
    # the path is monotone, starts at (0,0), ends at (N-1,M-1) and sums to the distance
    assert tuple(result.path[:, 0]) == (0, 0)
    assert tuple(result.path[:, -1]) == (shape[0] - 1, shape[1] - 1)
    assert np.all(np.diff(result.path, axis=1) >= 0)
    assert cost[result.path[0], result.path[1]].sum() == pytest.approx(result.distance)


def test_time_stretch_costs_nothing() -> None:
    rng = np.random.default_rng(1)
    a = rng.random((12, 40))
    b = np.repeat(a, 2, axis=1)  # same content at half tempo
    result = dtw.dtw(dtw.cosine_cost(a, b), band=0.1)
    assert result.distance == pytest.approx(0.0, abs=1e-9)
    assert result.similarity == pytest.approx(1.0)


def test_band_follows_the_diagonal_for_unequal_lengths() -> None:
    # |N - M| = 40 > the old |i - j| <= 0.1 * N band; the slope-following band works
    cost = np.random.default_rng(2).random((60, 100))
    banded = dtw.dtw(cost, band=0.1)
    assert np.isfinite(banded.distance)
    assert banded.distance >= dtw.dtw(cost).distance - 1e-12


def test_subsequence_finds_an_embedded_quote() -> None:
    rng = np.random.default_rng(3)
    reference = rng.random((12, 200))
    query = reference[:, 120:150]
    result = dtw.subsequence_dtw(dtw.cosine_cost(query, reference))
    assert result.distance == pytest.approx(0.0, abs=1e-9)
    assert (result.path[1, 0], result.path[1, -1]) == (120, 149)
