"""NMF convergence guarantees and the atom-permutation problem the joint fit fixes."""

from __future__ import annotations

import numpy as np
import pytest

from musicplag import markov, nmf


def _low_rank(seed: int, f: int = 30, t: int = 80, k: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((f, k)) @ rng.random((k, t))


def test_kl_updates_never_increase_the_divergence() -> None:
    fact = nmf.nmf(_low_rank(0) + 0.01, n_components=4, n_iter=150, seed=1)
    assert np.all(np.diff(fact.losses) <= 1e-8 * fact.losses[:-1])  # Lee & Seung, Thm 2


def test_exact_low_rank_matrix_is_reconstructed() -> None:
    v = _low_rank(1)
    fact = nmf.nmf(v, n_components=4, n_iter=4000, seed=0)
    assert np.abs(fact.W @ fact.H - v).max() / v.max() < 0.01
    np.testing.assert_allclose(fact.W.sum(axis=0), 1.0)


def test_joint_fit_shares_atoms_between_recordings() -> None:
    v = _low_rank(2) + 0.01
    _, h_a, h_b = nmf.joint_nmf(v, v.copy(), n_components=4, n_iter=300)
    s_a, s_b = markov.dominant_states(h_a), markov.dominant_states(h_b)
    np.testing.assert_array_equal(s_a, s_b)  # identical music -> identical labels
    assert markov.chain_similarity(s_a, s_b, 4, 0.5) == 1.0


def test_separate_fits_label_the_same_music_differently() -> None:
    """The v1-v3 bug: two NMF runs on one recording disagree on what atom 0 is."""
    v = _low_rank(3) + 0.01
    s1 = markov.dominant_states(nmf.nmf(v, 4, 300, seed=0).H)
    s2 = markov.dominant_states(nmf.nmf(v, 4, 300, seed=7).H)
    assert np.mean(s1 == s2) < 0.9  # same music, different state labels
    p1 = markov.transition_matrix(s1, 4, 0.5)
    p2 = markov.transition_matrix(s2, 4, 0.5)
    assert markov.legacy_kl_similarity(p1, p2) < 0.99


def test_chain_similarity_is_label_invariant_only_jointly() -> None:
    s = np.tile([0, 1, 2, 3, 4, 5, 2, 3], 40)  # a repeating progression
    perm = np.array([3, 0, 5, 1, 2, 4])
    both = markov.chain_similarity(perm[s], perm[s], 6, 0.5)
    one = markov.chain_similarity(s, perm[s], 6, 0.5)
    assert both == 1.0
    assert one < 0.9


def test_js_divergence_is_bounded_and_symmetric() -> None:
    rng = np.random.default_rng(5)
    p = rng.dirichlet(np.ones(8), size=10)
    q = rng.dirichlet(np.ones(8), size=10)
    js = markov.js_divergence(p, q)
    np.testing.assert_allclose(js, markov.js_divergence(q, p))
    assert np.all((js >= 0) & (js <= np.log(2) + 1e-12))
    disjoint = markov.js_divergence(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert disjoint == pytest.approx(np.log(2))
