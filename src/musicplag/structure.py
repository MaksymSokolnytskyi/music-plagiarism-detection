r"""Self-similarity matrices: song form (verse/chorus repetition) as a picture.

:math:`S_{ij} = \cos(f_i, f_j)` over chroma frames (Foote 1999). Repeated sections show up
as stripes parallel to the main diagonal, homogeneous sections as blocks. Because the
SSM of a uniformly time-stretched track is the same picture at a different resolution,
both matrices are resampled to one size before comparison (as in v4).

The comparison is the Pearson correlation of the strictly-upper triangle beyond a small
band around the diagonal — the band is similar in every recording (a frame resembles its
neighbours) and only inflates the score.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom

FloatArray = NDArray[np.float64]
_EPS = 1e-12
_DIAGONAL_BAND = 2  # resampled cells excluded around the main diagonal


def self_similarity(frames: FloatArray) -> FloatArray:
    """Cosine self-similarity of the columns of ``frames`` ``(F, T)`` → ``(T, T)``."""
    unit = frames / (np.linalg.norm(frames, axis=0, keepdims=True) + _EPS)
    return unit.T @ unit


def resample(ssm: FloatArray, size: int) -> FloatArray:
    """Bilinear resampling of a square matrix to ``size × size``."""
    return zoom(ssm, size / ssm.shape[0], order=1)[:size, :size]


def ssm_correlation(frames_a: FloatArray, frames_b: FloatArray, size: int) -> float:
    """Pearson correlation of two recordings' resampled SSMs (off-diagonal part)."""
    a = resample(self_similarity(frames_a), size)
    b = resample(self_similarity(frames_b), size)
    iu = np.triu_indices(size, k=_DIAGONAL_BAND)
    x, y = a[iu], b[iu]
    if x.std() < _EPS or y.std() < _EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
