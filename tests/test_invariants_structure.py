"""Fourier-Mellin invariances, phase correlation, and SSM form comparison."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import zoom

from musicplag import invariants, structure

RATE = 43.07  # frames per second at 22050 / 512
BINS = 32


def _pattern(seed: int, frames: int = 600) -> np.ndarray:
    """Sparse 'notes' on an 84 x frames grid, with musical periodicity."""
    rng = np.random.default_rng(seed)
    x = np.zeros((84, frames))
    for _ in range(60):
        f, t, d = rng.integers(20, 64), rng.integers(0, frames - 30), rng.integers(5, 30)
        x[f, t : t + d] += rng.uniform(0.5, 1.0)
    return x


def test_descriptor_ignores_circular_transposition_and_time_shift() -> None:
    x = _pattern(0)
    moved = np.roll(np.roll(x, 5, axis=0), 120, axis=1)
    d0 = invariants.fourier_mellin(x, RATE, BINS)
    d1 = invariants.fourier_mellin(moved, RATE, BINS)
    np.testing.assert_allclose(d0, d1, atol=1e-10)


@pytest.mark.parametrize("stretch", [0.8, 1.25])
def test_descriptor_is_close_under_tempo_change(stretch: float) -> None:
    x = _pattern(1)
    stretched = zoom(x, (1.0, stretch), order=1)
    same = invariants.descriptor_similarity(
        invariants.fourier_mellin(x, RATE, BINS),
        invariants.fourier_mellin(stretched, RATE, BINS),
    )
    other = invariants.descriptor_similarity(
        invariants.fourier_mellin(x, RATE, BINS),
        invariants.fourier_mellin(_pattern(2), RATE, BINS),
    )
    assert same > other


@pytest.mark.parametrize("shift", [-9, -1, 0, 4, 11])
def test_phase_correlation_recovers_circular_shift(shift: int) -> None:
    a = np.random.default_rng(3).random(84)
    assert invariants.estimate_shift(np.roll(a, shift), a) == shift


def test_ssm_form_survives_tempo_change_but_not_rearrangement() -> None:
    rng = np.random.default_rng(4)
    verse, chorus = rng.random((12, 20)), rng.random((12, 20))
    song = np.hstack([verse, chorus, verse, chorus, chorus])
    slower = np.repeat(song, 2, axis=1)
    other = np.hstack([verse, verse, chorus, verse, verse])
    assert structure.ssm_correlation(song, slower, 48) > 0.9
    assert structure.ssm_correlation(song, other, 48) < structure.ssm_correlation(
        song, slower, 48
    )
