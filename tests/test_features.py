"""End-to-end pair analysis on rendered synthetic songs."""

from __future__ import annotations

import numpy as np
import pytest

from musicplag import synth
from musicplag.config import Config
from musicplag.features import DEFAULT_FEATURES, PairContext, PairFeature, analyze_pair


def test_every_default_feature_satisfies_the_protocol() -> None:
    assert all(isinstance(f, PairFeature) for f in DEFAULT_FEATURES)
    assert len({f.name for f in DEFAULT_FEATURES}) == len(DEFAULT_FEATURES)


@pytest.fixture(scope="module")
def cover_pair(fast_cfg: Config) -> tuple[PairContext, synth.Song, synth.Song]:
    rng = np.random.default_rng(11)
    original = synth.random_song(rng)
    cover = synth.edit(original, "cover", rng)
    ctx = PairContext.from_audio(
        synth.render(original, fast_cfg.sample_rate),
        synth.render(cover, fast_cfg.sample_rate),
        fast_cfg,
    )
    return ctx, original, cover


def test_report_recovers_the_transposition(cover_pair) -> None:  # type: ignore[no-untyped-def]
    ctx, original, cover = cover_pair
    report = analyze_pair(ctx)
    assert report.transposition_semitones == cover.tonic - original.tonic
    assert set(report.features) == {f.name for f in DEFAULT_FEATURES}
    assert all(np.isfinite(v) for v in report.features.values())


def test_self_comparison_is_maximal(fast_cfg: Config) -> None:
    song = synth.random_song(np.random.default_rng(12))
    y = synth.render(song, fast_cfg.sample_rate)
    report = analyze_pair(PairContext.from_audio(y, y.copy(), fast_cfg))
    for name, value in report.features.items():
        assert value == pytest.approx(1.0, abs=1e-6), name
    assert report.transposition_semitones == 0
