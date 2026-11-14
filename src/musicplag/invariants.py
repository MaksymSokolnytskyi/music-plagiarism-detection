r"""Transposition- and tempo-invariant descriptors (Fourier-Mellin) and shift estimation.

Two editing operations dominate musical borrowing: transposition and tempo change. In a
log-frequency (CQT) representation :math:`X(f, t)` they act as

* transposition by :math:`s` semitones — a shift :math:`X(f - s, t)`;
* tempo change by :math:`a` — a scaling of time :math:`X(f, t/a)`.

**Shifts** vanish in the magnitude of the 2-D Fourier transform (shift theorem):
:math:`|\mathcal F\{X(f-s, t-\tau)\}| = |\mathcal F\{X\}|`.

**Time scaling** becomes a scaling of the modulation-frequency axis,
:math:`\omega_t \mapsto a\,\omega_t`, which a logarithmic resampling
:math:`u = \ln \omega_t` turns into a shift :math:`u \mapsto u + \ln a`; a second Fourier
magnitude along :math:`u` removes it. This is the Mellin / scale transform (Cohen 1993)
applied to one axis only.

Why one axis. v1 applied an image log-polar warp (``cv2.logPolar``) around the centre of
a spectrogram patch. Log-polar turns *isotropic* scaling and rotation into shifts, but a
tempo change scales time alone and a transposition is a shift, not a scale, in a CQT —
so the warp did not encode either invariance it was meant to. The construction below
encodes exactly the two operations above. The invariance to tempo is approximate (finite
frequency range, interpolation); the tests bound the error.

The modulation axis is measured in hertz, not in DFT bins, so descriptors of patches of
different lengths are comparable.

Phase correlation (Kuglin & Hines 1975) recovers the shift itself:
:math:`\arg\max_n \mathcal F^{-1}\{\hat A \hat B^* / |\hat A \hat B^*|\}[n]` — this is
the "transposed by +3 semitones" line in an evidence report.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_EPS = 1e-12
MODULATION_RANGE_HZ = (0.25, 16.0)  # ~1 cycle per 4 s (phrases) up to fast rhythm


def fourier_mellin(patch: FloatArray, frame_rate: float, n_log_bins: int) -> FloatArray:
    """Unit-norm descriptor of a ``(pitch, time)`` patch, invariant to both edits above.

    ``frame_rate`` is in frames per second; the patch must span at least ~4 s so the
    lowest modulation frequency is resolved.
    """
    x = np.log1p(np.asarray(patch, dtype=np.float64))
    x = x - x.mean()
    # half-plane of the 2-D spectrum: real input makes the other half redundant
    spectrum = np.abs(np.fft.fft(np.fft.rfft(x, axis=1), axis=0))  # (F, T//2+1)
    hz = np.fft.rfftfreq(x.shape[1], d=1.0 / frame_rate)
    lo, hi = MODULATION_RANGE_HZ
    hi = min(hi, hz[-1])
    assert hz[1] <= lo, "patch shorter than 1/lowest modulation frequency (~4 s)"
    log_grid = np.exp(np.linspace(np.log(lo), np.log(hi), n_log_bins))
    warped = np.vstack([np.interp(log_grid, hz, row) for row in spectrum])
    descriptor = np.abs(np.fft.rfft(warped, axis=1))
    flat = descriptor.ravel()
    return flat / (np.linalg.norm(flat) + _EPS)


def descriptor_similarity(d_a: FloatArray, d_b: FloatArray) -> float:
    """Cosine similarity of two unit-norm descriptors."""
    return float(np.clip(d_a @ d_b, -1.0, 1.0))


def estimate_shift(a: FloatArray, b: FloatArray) -> int:
    """Integer ``n`` such that ``roll(b, n)`` best aligns with ``a`` (phase correlation).

    Exact for circular shifts; for zero-filled shifts it is the correlation peak.
    Result is in ``[-len//2, len//2)``.
    """
    fa = np.fft.fft(a - a.mean())
    fb = np.fft.fft(b - b.mean())
    cross = fa * np.conj(fb)
    peak = int(np.argmax(np.real(np.fft.ifft(cross / (np.abs(cross) + _EPS)))))
    n = len(a)
    return peak - n if peak >= n // 2 else peak
