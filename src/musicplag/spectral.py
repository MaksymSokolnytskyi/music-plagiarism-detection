r"""Pitch-aware time-frequency representations: CQT, chroma, Tonnetz, transposition.

Why a constant-Q front end: in a CQT with one bin per semitone, transposing a melody by
:math:`s` semitones is a *vertical shift by s bins* (exactly, up to the edges). Every
transposition-invariant feature in this package rests on that one fact.

Constant-Q transform (Brown 1991). Bin :math:`k` has centre frequency
:math:`f_k = f_{\min} 2^{k/b}` and a window long enough to keep the quality factor
:math:`Q = f_k / \Delta f_k = (2^{1/b} - 1)^{-1}` fixed:

.. math::
    N_k = \left\lceil Q f_s / f_k \right\rceil,\qquad
    X[k, n] = \frac{1}{N_k} \sum_{m=0}^{N_k-1}
              x[nH + m - \lfloor N_k/2 \rfloor]\, w_{N_k}[m]\,
              e^{-j 2\pi Q m / N_k}.

Chroma folds octaves: :math:`c[p, n] = \sum_{o} |X[p + bo, n]|^2`, :math:`p = 0..11`.

Tonnetz (Harte, Sandler & Gasser 2006) projects L1-normalised chroma onto three circles
(fifths, minor thirds, major thirds): :math:`\zeta = \Phi c / \lVert c\rVert_1`.

Optimal transposition index (Serrà, Gómez & Herrera 2008):
:math:`k^* = \arg\max_k \langle \bar c_A, \mathrm{roll}(\bar c_B, k)\rangle`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve, resample_poly

from musicplag.config import CONFIG, Config

FloatArray = NDArray[np.float64]

_EPS = 1e-12  # guards divisions by zero energy (silence)
_SILENCE_DB = -60.0  # frames this far below the peak count as silence
_ANTIALIAS_FRACTION = 0.4  # octave must sit below 40% of a decimated rate
TONNETZ_RADII = (1.0, 1.0, 0.5)  # fifths, minor thirds, major thirds — Harte 2006
TONNETZ_ANGLES = (7 * np.pi / 6, 3 * np.pi / 2, 2 * np.pi / 3)


def bin_frequencies(cfg: Config = CONFIG) -> FloatArray:
    """Centre frequencies :math:`f_k = f_{min} 2^{k/b}` of all CQT bins, Hz."""
    k = np.arange(cfg.n_bins, dtype=np.float64)
    return cfg.fmin * 2.0 ** (k / cfg.bins_per_octave)


def _decimation(f_top: float, sr: int, hop: int) -> int:
    """Largest power-of-two factor that keeps ``f_top`` below 40% of the reduced rate."""
    factor = 1
    while hop % (2 * factor) == 0 and f_top < _ANTIALIAS_FRACTION * sr / (2 * factor):
        factor *= 2
    return factor


def cqt(y: FloatArray, cfg: Config = CONFIG, multirate: bool = True) -> FloatArray:
    """Magnitude constant-Q transform, shape ``(n_bins, n_frames)``.

    Direct per-bin convolution (Brown 1991) with a Hann window. With ``multirate``
    (default) each octave is computed on the signal decimated by the largest power of
    two that keeps the octave below 40% of the reduced sample rate (an anti-aliased
    polyphase resampler precedes each halving). The cost drops from ``n_bins`` to
    about ``2 * bins_per_octave`` full-length convolutions; ``tests/test_spectral.py``
    checks the result against ``multirate=False``. Bins above Nyquist stay zero;
    frames are centred on ``n * hop_length``.
    """
    y = np.asarray(y, dtype=np.float64)
    assert y.ndim == 1, "cqt expects a mono signal"
    q = 1.0 / (2.0 ** (1.0 / cfg.bins_per_octave) - 1.0)
    n_frames = 1 + len(y) // cfg.hop_length
    out = np.zeros((cfg.n_bins, n_frames))
    freqs = bin_frequencies(cfg)
    signals: dict[int, FloatArray] = {1: y}
    for k, f_k in enumerate(freqs):
        if f_k >= cfg.sample_rate / 2:
            break
        octave_top = freqs[
            min(len(freqs) - 1, (k // cfg.bins_per_octave + 1) * cfg.bins_per_octave - 1)
        ]
        factor = (
            _decimation(octave_top, cfg.sample_rate, cfg.hop_length) if multirate else 1
        )
        while factor not in signals:  # halve the previous level once per new factor
            prev = max(signals)
            signals[prev * 2] = resample_poly(signals[prev], 1, 2)
        x, sr = signals[factor], cfg.sample_rate / factor
        n_k = int(np.ceil(q * sr / f_k))
        m = np.arange(n_k)
        kernel = np.hanning(n_k) * np.exp(-2j * np.pi * q * m / n_k) / n_k
        full = fftconvolve(x, kernel[::-1])  # correlation == convolution with reversal
        idx = np.arange(n_frames) * (cfg.hop_length // factor) + n_k - 1 - n_k // 2
        valid = idx < len(full)
        out[k, valid] = np.abs(full[idx[valid]])
    return out


def pool(features: FloatArray, factor: int) -> FloatArray:
    """Average-pool the time axis by ``factor`` (drops the ragged tail)."""
    if factor <= 1:
        return features
    n = features.shape[1] // factor
    assert n > 0, "signal shorter than one pooled frame"
    return features[:, : n * factor].reshape(features.shape[0], n, factor).mean(axis=2)


def chroma(cqt_mag: FloatArray, bins_per_octave: int = 12) -> FloatArray:
    """Fold CQT energy into pitch classes; each frame L2-normalised, shape ``(12, T)``.

    Assumes ``fmin`` is a C, so row 0 is C. Requires one bin per semitone. Frames more
    than 60 dB below the loudest frame carry no pitch information and become uniform;
    otherwise two silent frames would have cosine distance 1, not 0.
    """
    assert bins_per_octave == 12, "chroma folding assumes one bin per semitone"
    energy = cqt_mag**2
    c = np.zeros((12, cqt_mag.shape[1]))
    for p in range(12):
        c[p] = energy[p::12].sum(axis=0)
    total = c.sum(axis=0)
    silent = total <= total.max() * 10 ** (_SILENCE_DB / 10) if total.size else total > 0
    c[:, silent] = 1.0  # no pitch information: uniform, so silence matches silence
    return c / (np.linalg.norm(c, axis=0, keepdims=True) + _EPS)


def tonnetz(chroma_frames: FloatArray) -> FloatArray:
    """Harte 2006 six-dimensional tonal centroid, shape ``(6, T)``."""
    p = np.arange(12)
    rows = []
    for r, a in zip(TONNETZ_RADII, TONNETZ_ANGLES, strict=True):
        rows += [r * np.sin(p * a), r * np.cos(p * a)]
    phi = np.vstack(rows)  # (6, 12)
    l1 = np.abs(chroma_frames).sum(axis=0, keepdims=True) + _EPS
    return phi @ (chroma_frames / l1)


def optimal_transposition(chroma_a: FloatArray, chroma_b: FloatArray) -> int:
    """Semitone shift ``k`` such that ``roll(chroma_b, k)`` best matches ``chroma_a``.

    Global profiles are compared, so the answer is defined modulo 12 and is robust
    to tempo and structure differences (Serrà 2008).
    """
    ga = chroma_a.mean(axis=1)
    gb = chroma_b.mean(axis=1)
    scores = [float(ga @ np.roll(gb, k)) for k in range(12)]
    return int(np.argmax(scores))


def transpose_cqt(
    cqt_mag: FloatArray, semitones: int, bins_per_octave: int = 12
) -> FloatArray:
    """Shift a CQT up by ``semitones`` (zero-filled at the edge, not wrapped)."""
    shift = semitones * bins_per_octave // 12
    out = np.zeros_like(cqt_mag)
    if shift > 0:
        out[shift:] = cqt_mag[:-shift]
    elif shift < 0:
        out[:shift] = cqt_mag[-shift:]
    else:
        out[:] = cqt_mag
    return out
