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


def test_local_match_finds_a_passage_quoted_inside_unrelated_material() -> None:
    from musicplag.config import CONFIG
    from musicplag.features import LocalChromaMatch

    rng = np.random.default_rng(13)
    cfg = CONFIG.model_copy(update={"local_window_seconds": 10.0})
    rate = cfg.sample_rate / cfg.hop_length
    n = round(60 * rate)  # one minute of unpooled frames
    original = rng.random((84, n)) ** 8
    suspect = rng.random((84, n)) ** 8
    start, length = (n // 2) // 8 * 8, round(15 * rate) // 8 * 8  # on pooling blocks
    suspect[:, start : start + length] = original[:, :length]
    unrelated = rng.random((84, n)) ** 8
    feature = LocalChromaMatch()
    quoted = feature(PairContext(original, suspect, cfg))
    assert quoted > 0.99
    # chroma cosine has a high floor (~0.9) even for unrelated material; what matters
    # is that an exact quote reaches 1 and unrelated audio stays clearly below it
    assert quoted > feature(PairContext(original, unrelated, cfg)) + 0.05
