"""Pair analysis: one shared context, interchangeable similarity features (Strategy).

:class:`PairContext` computes every representation once (CQT, pooled CQT, chroma, the
optimal transposition, joint-NMF states). Each feature is a small object satisfying the
:class:`PairFeature` protocol, so adding or ablating a feature never touches the others.
:func:`analyze_pair` is the single entry point used by the CLI and the experiments.

Every feature is a similarity: larger means "more alike". None of them is a verdict —
see ``results/`` for how well each separates borrowed from unrelated material.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from musicplag import dtw, invariants, markov, nmf, spectral, structure
from musicplag.config import CONFIG, Config

FloatArray = NDArray[np.float64]


@dataclass
class PairContext:
    """Lazily computed representations of a recording pair (A = original, B = suspect)."""

    cqt_a: FloatArray
    cqt_b: FloatArray
    cfg: Config = field(default_factory=lambda: CONFIG)

    @classmethod
    def from_audio(
        cls, y_a: FloatArray, y_b: FloatArray, cfg: Config = CONFIG
    ) -> PairContext:
        """Build a context from two mono signals at ``cfg.sample_rate``."""
        return cls(spectral.cqt(y_a, cfg), spectral.cqt(y_b, cfg), cfg)

    @property
    def frame_rate(self) -> float:
        """Unpooled CQT frames per second."""
        return self.cfg.sample_rate / self.cfg.hop_length

    @cached_property
    def pooled_a(self) -> FloatArray:
        return spectral.pool(self.cqt_a, self.cfg.pool_frames)

    @cached_property
    def pooled_b(self) -> FloatArray:
        return spectral.pool(self.cqt_b, self.cfg.pool_frames)

    @cached_property
    def chroma_a(self) -> FloatArray:
        return spectral.chroma(self.pooled_a, self.cfg.bins_per_octave)

    @cached_property
    def chroma_b(self) -> FloatArray:
        return spectral.chroma(self.pooled_b, self.cfg.bins_per_octave)

    @cached_property
    def transposition(self) -> int:
        """Semitones (mod 12) that bring B's key onto A's (optimal transposition)."""
        return spectral.optimal_transposition(self.chroma_a, self.chroma_b)

    @cached_property
    def chroma_b_aligned(self) -> FloatArray:
        return np.roll(self.chroma_b, self.transposition, axis=0)

    @cached_property
    def joint_states(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Dominant-atom sequences from one NMF dictionary fitted on A and aligned B."""
        shift = self.transposition if self.transposition <= 6 else self.transposition - 12
        b_aligned = spectral.transpose_cqt(self.pooled_b, shift, self.cfg.bins_per_octave)
        _, h_a, h_b = nmf.joint_nmf(
            np.log1p(self.pooled_a),
            np.log1p(b_aligned),
            self.cfg.nmf_components,
            self.cfg.nmf_iterations,
            self.cfg.seed,
        )
        return markov.dominant_states(h_a), markov.dominant_states(h_b)


@runtime_checkable
class PairFeature(Protocol):
    """A named similarity computed from a :class:`PairContext`."""

    name: str

    def __call__(self, ctx: PairContext) -> float: ...


@dataclass(frozen=True)
class ChromaProfile:
    """Cosine of key-aligned global pitch-class profiles (baseline; ignores order)."""

    name: str = "chroma_profile"

    def __call__(self, ctx: PairContext) -> float:
        a = ctx.chroma_a.mean(axis=1)
        b = ctx.chroma_b_aligned.mean(axis=1)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@dataclass(frozen=True)
class ChromaDTW:
    """Per-step similarity of the best tempo-warped alignment of key-aligned chroma."""

    name: str = "chroma_dtw"

    def __call__(self, ctx: PairContext) -> float:
        cost = dtw.cosine_cost(ctx.chroma_a, ctx.chroma_b_aligned)
        return dtw.dtw(cost, band=ctx.cfg.dtw_band).similarity


@dataclass(frozen=True)
class MarkovChain:
    """Similarity of harmonic-motion Markov chains over shared NMF atoms."""

    name: str = "markov_js"

    def __call__(self, ctx: PairContext) -> float:
        s_a, s_b = ctx.joint_states
        k = ctx.cfg.nmf_components
        return markov.chain_similarity(s_a, s_b, k, ctx.cfg.markov_smoothing)


@dataclass(frozen=True)
class FormSSM:
    """Correlation of song-form self-similarity matrices."""

    name: str = "ssm_form"

    def __call__(self, ctx: PairContext) -> float:
        return structure.ssm_correlation(ctx.chroma_a, ctx.chroma_b, ctx.cfg.ssm_size)


@dataclass(frozen=True)
class FourierMellin:
    """Cosine of transposition- and tempo-invariant CQT descriptors."""

    name: str = "fourier_mellin"

    def __call__(self, ctx: PairContext) -> float:
        rate, bins = ctx.frame_rate, ctx.cfg.mellin_bins
        d_a = invariants.fourier_mellin(ctx.cqt_a, rate, bins)
        d_b = invariants.fourier_mellin(ctx.cqt_b, rate, bins)
        return invariants.descriptor_similarity(d_a, d_b)


DEFAULT_FEATURES: tuple[PairFeature, ...] = (
    ChromaProfile(),
    ChromaDTW(),
    MarkovChain(),
    FormSSM(),
    FourierMellin(),
)


class PairReport(BaseModel):
    """Feature values and the evidence behind them for one pair."""

    model_config = ConfigDict(frozen=True)

    features: dict[str, float] = Field(description="Similarity per feature name.")
    transposition_semitones: int = Field(
        description="Key offset of B relative to A, in (-6, 6]."
    )
    alignment_diagonal_fraction: float = Field(
        ge=0, le=1, description="Share of DTW steps that advance both tracks together."
    )


def analyze_pair(
    ctx: PairContext, features: Sequence[PairFeature] = DEFAULT_FEATURES
) -> PairReport:
    """Evaluate ``features`` on ``ctx`` and attach alignment evidence."""
    values = {f.name: f(ctx) for f in features}
    shift = ctx.transposition if ctx.transposition <= 6 else ctx.transposition - 12
    path = dtw.dtw(dtw.cosine_cost(ctx.chroma_a, ctx.chroma_b_aligned), ctx.cfg.dtw_band)
    return PairReport(
        features=values,
        transposition_semitones=-shift,
        alignment_diagonal_fraction=path.diagonal_fraction,
    )
