r"""First-order Markov chains over NMF atom states, compared by Jensen-Shannon divergence.

Each frame is labelled with its dominant atom, :math:`s_t = \arg\max_k H_{kt}`; the
chain :math:`P_{ij} = \Pr(s_{t+1} = j \mid s_t = i)` is estimated with Lidstone
smoothing :math:`\alpha`. Two chains describe "how the harmony moves" independently of
tempo and of where in the track a progression occurs.

v1 compared chains with :math:`KL(P_A\|P_B)` over the flattened matrix. KL is asymmetric
and unbounded, and flattening weights a never-visited state's smoothed row the same as
the chorus chord. Here rows are weighted by how often each state is actually visited and
compared with the symmetric, bounded Jensen-Shannon divergence (Lin 1991):

.. math::
    \mathrm{JS}(p, q) = \tfrac12 KL(p\|m) + \tfrac12 KL(q\|m),\quad m = \tfrac12(p+q),
    \qquad 0 \le \mathrm{JS} \le \ln 2,

.. math::
    D(P_A, P_B) = \sum_i \pi_i\, \mathrm{JS}(P_A[i], P_B[i]),\qquad
    \pi_i = \tfrac12(\hat\pi^A_i + \hat\pi^B_i),

and similarity :math:`1 - D/\ln 2 \in [0, 1]`. This is meaningful only when state
:math:`i` denotes the same atom in both recordings — see :mod:`musicplag.nmf`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_EPS = 1e-12


def dominant_states(activations: FloatArray) -> NDArray[np.int64]:
    """Index of the most active atom in every frame."""
    return np.argmax(activations, axis=0).astype(np.int64)


def transition_matrix(
    states: NDArray[np.int64], n_states: int, alpha: float
) -> FloatArray:
    """Row-stochastic transition matrix with additive smoothing ``alpha``."""
    counts = np.full((n_states, n_states), alpha, dtype=np.float64)
    np.add.at(counts, (states[:-1], states[1:]), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def occupancy(states: NDArray[np.int64], n_states: int) -> FloatArray:
    """Empirical distribution of visited states."""
    return np.bincount(states, minlength=n_states).astype(np.float64) / len(states)


def js_divergence(p: FloatArray, q: FloatArray) -> FloatArray:
    """Row-wise Jensen-Shannon divergence in nats, for row-stochastic arrays."""
    m = 0.5 * (p + q)

    def kl(a: FloatArray, b: FloatArray) -> FloatArray:
        return np.sum(np.where(a > 0, a * np.log((a + _EPS) / (b + _EPS)), 0.0), axis=-1)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def chain_similarity(
    states_a: NDArray[np.int64], states_b: NDArray[np.int64], n_states: int, alpha: float
) -> float:
    """Occupancy-weighted JS similarity of two sequences' Markov chains, in [0, 1]."""
    pa = transition_matrix(states_a, n_states, alpha)
    pb = transition_matrix(states_b, n_states, alpha)
    weights = 0.5 * (occupancy(states_a, n_states) + occupancy(states_b, n_states))
    divergence = float(weights @ js_divergence(pa, pb))
    return float(np.clip(1.0 - divergence / np.log(2.0), 0.0, 1.0))


def legacy_kl_similarity(pa: FloatArray, pb: FloatArray) -> float:
    """v1 metric, kept for the ablation: ``1 / (1 + KL(flat P_A || flat P_B))``."""
    a = pa.ravel() / pa.sum()
    b = pb.ravel() / pb.sum()
    kl = float(np.sum(a * np.log((a + _EPS) / (b + _EPS))))
    return 1.0 / (1.0 + kl)
