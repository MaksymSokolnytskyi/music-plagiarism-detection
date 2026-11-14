"""CQT, chroma, Tonnetz and transposition against analytic expectations."""

from __future__ import annotations

import numpy as np
import pytest

from musicplag import spectral
from musicplag.config import Config


def _tone(freq: float, cfg: Config, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * cfg.sample_rate)) / cfg.sample_rate
    return np.sin(2 * np.pi * freq * t)


def test_pure_tone_peaks_at_its_semitone(fast_cfg: Config) -> None:
    # A4 = MIDI 69, C1 = MIDI 24 -> bin 45
    mag = spectral.cqt(_tone(440.0, fast_cfg), fast_cfg)
    assert int(np.argmax(mag.mean(axis=1))) == 45


@pytest.mark.parametrize("semitones", [-7, -2, 3, 5])
def test_transposition_is_a_bin_shift(fast_cfg: Config, semitones: int) -> None:
    base = spectral.cqt(_tone(440.0, fast_cfg), fast_cfg).mean(axis=1)
    moved = spectral.cqt(_tone(440.0 * 2 ** (semitones / 12), fast_cfg), fast_cfg)
    assert int(np.argmax(moved.mean(axis=1))) - int(np.argmax(base)) == semitones


def test_chroma_folds_octaves_and_is_unit_norm(fast_cfg: Config) -> None:
    y = _tone(220.0, fast_cfg) + _tone(880.0, fast_cfg)  # two A's
    c = spectral.chroma(spectral.cqt(y, fast_cfg))
    assert int(np.argmax(c.mean(axis=1))) == 9  # A
    np.testing.assert_allclose(np.linalg.norm(c[:, 5:-5], axis=0), 1.0, atol=1e-9)


def test_tonnetz_matches_librosa() -> None:
    librosa = pytest.importorskip("librosa")
    rng = np.random.default_rng(0)
    c = rng.random((12, 20))
    ours = spectral.tonnetz(c)
    ref = librosa.feature.tonnetz(chroma=c)
    np.testing.assert_allclose(ours, ref, atol=1e-6)


@pytest.mark.parametrize("shift", range(12))
def test_optimal_transposition_inverts_a_roll(shift: int) -> None:
    rng = np.random.default_rng(shift)
    a = rng.random((12, 30)) ** 4  # peaky profile
    assert spectral.optimal_transposition(a, np.roll(a, shift, axis=0)) == (-shift) % 12


def test_transpose_cqt_round_trips_inside_range() -> None:
    x = np.zeros((84, 3))
    x[40] = 1.0
    np.testing.assert_array_equal(
        spectral.transpose_cqt(spectral.transpose_cqt(x, 4), -4), x
    )


def test_multirate_matches_direct_convolution(fast_cfg: Config) -> None:
    rng = np.random.default_rng(7)
    t = np.arange(int(3.0 * fast_cfg.sample_rate)) / fast_cfg.sample_rate
    freqs = 55.0 * 2 ** (rng.integers(0, 72, 12) / 12)  # tones across six octaves
    y = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in freqs)
    direct = spectral.cqt(y, fast_cfg, multirate=False)
    fast = spectral.cqt(y, fast_cfg)
    assert np.linalg.norm(fast - direct) / np.linalg.norm(direct) < 0.1
    expected = set(np.round(12 * np.log2(freqs / fast_cfg.fmin)).astype(int))
    top = set(np.argsort(fast.mean(axis=1))[-len(expected) :])
    assert top == expected  # every tone in its own semitone bin


def test_silence_matches_silence() -> None:
    from musicplag import dtw

    mag = np.zeros((84, 40))
    mag[45, 10:30] = 1.0  # A4 between two stretches of silence
    c = spectral.chroma(mag)
    assert dtw.dtw(dtw.cosine_cost(c, c)).distance == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(c[:, 0], np.full(12, 1 / np.sqrt(12)))
