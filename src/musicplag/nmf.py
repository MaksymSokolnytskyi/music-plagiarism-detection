r"""Non-negative matrix factorisation with a dictionary shared by both recordings.

NMF writes a non-negative spectrogram as a sum of spectral atoms switched on over time,
:math:`V \approx WH` with :math:`W, H \ge 0` (Smaragdis & Brown 2003). Under the
generalised Kullback-Leibler divergence

.. math::
    D_{KL}(V\|WH) = \sum_{ft} V_{ft}\log\frac{V_{ft}}{(WH)_{ft}} - V_{ft} + (WH)_{ft}

the multiplicative updates of Lee & Seung (2001)

.. math::
    H \leftarrow H \odot \frac{W^\top (V \oslash WH)}{W^\top \mathbf{1}}, \qquad
    W \leftarrow W \odot \frac{(V \oslash WH) H^\top}{\mathbf{1} H^\top}

never increase the divergence (their Theorem 2) — a property the tests check.

Why *joint*. Versions v1-v3 fitted NMF on each track separately and then compared the
activation matrices :math:`H_A, H_B` row by row (DTW) and built Markov chains over
"atom index" states. NMF is identifiable only up to a permutation of atoms, so atom 3 of
track A and atom 3 of track B are unrelated objects; the comparison was between
arbitrarily relabelled states. Fitting one dictionary on :math:`[V_A \; V_B]` makes the
state labels mean the same thing in both recordings (see ``tests/test_nmf.py`` and the
ablation in ``experiments/synthetic_benchmark.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_EPS = 1e-12


@dataclass(frozen=True)
class Factorization:
    """``V ≈ W @ H``; ``W`` columns are L1-normalised atoms; ``losses`` per step."""

    W: FloatArray
    H: FloatArray
    losses: FloatArray


def kl_divergence(v: FloatArray, approx: FloatArray) -> float:
    """Generalised KL divergence :math:`D_{KL}(V\\|\\hat V)`, zero iff equal."""
    approx = approx + _EPS
    return float(np.sum(v * np.log((v + _EPS) / approx) - v + approx))


def nmf(
    v: FloatArray, n_components: int, n_iter: int = 200, seed: int = 0
) -> Factorization:
    """KL-NMF by multiplicative updates from a seeded positive random start."""
    v = np.asarray(v, dtype=np.float64)
    assert np.all(v >= 0), "NMF needs a non-negative matrix"
    rng = np.random.default_rng(seed)
    scale = np.sqrt(v.mean() / n_components) + _EPS
    w = rng.uniform(0.5, 1.5, (v.shape[0], n_components)) * scale
    h = rng.uniform(0.5, 1.5, (n_components, v.shape[1])) * scale
    losses = np.empty(n_iter)
    for t in range(n_iter):
        h *= (w.T @ (v / (w @ h + _EPS))) / (w.sum(axis=0)[:, None] + _EPS)
        w *= ((v / (w @ h + _EPS)) @ h.T) / (h.sum(axis=1)[None, :] + _EPS)
        losses[t] = kl_divergence(v, w @ h)
    norms = w.sum(axis=0) + _EPS  # move atom scale into the activations
    return Factorization(w / norms, h * norms[:, None], losses)


def joint_nmf(
    v_a: FloatArray, v_b: FloatArray, n_components: int, n_iter: int = 200, seed: int = 0
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Fit one dictionary on both recordings; return ``(W, H_a, H_b)``."""
    assert v_a.shape[0] == v_b.shape[0], "recordings must share the frequency axis"
    fact = nmf(np.hstack([v_a, v_b]), n_components, n_iter, seed)
    split = v_a.shape[1]
    return fact.W, fact.H[:, :split], fact.H[:, split:]
