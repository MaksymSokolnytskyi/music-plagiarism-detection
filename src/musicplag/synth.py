"""Seeded synthetic songs and controlled edits: ground truth without copyrighted audio.

A song is a key, a tempo, a looped four-chord progression, a melody over it and two
timbres. Edits change one thing at a time, so each feature's behaviour can be traced to
the operation it is supposed to be invariant to:

Every edited song also gets a random 0-2 bar intro (the first benchmark run without it
gave the SSM feature a perfect score purely because all songs had identical form).

* positives — ``cover`` (transpose + tempo + timbre, same melody and chords) and
  ``melody_lift`` (same melody, new chords and timbre, transposed);
* negatives — ``unrelated`` (everything new) and ``same_loop`` (same key, tempo, chords
  and timbre; new melody). ``same_loop`` is the hard case: shared chord loops are
  common and legal, and every feature that looks only at harmony will flag them.

Rendering is additive synthesis with an exponential decay — crude, but the pitch
content is exactly known.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
DIATONIC_TRIADS = {
    1: (0, 4, 7),
    2: (2, 5, 9),
    4: (5, 9, 12),
    5: (7, 11, 14),
    6: (9, 12, 16),
}
TIMBRES = (  # relative amplitudes of harmonics 1..6
    (1.0, 0.5, 0.33, 0.25, 0.2, 0.16),  # sawtooth-like
    (1.0, 0.0, 0.33, 0.0, 0.2, 0.0),  # square-like (odd harmonics)
    (1.0, 0.1, 0.05, 0.0, 0.0, 0.0),  # near-sine
    (0.6, 1.0, 0.4, 0.3, 0.1, 0.05),  # strong 2nd harmonic
)
BEATS_PER_BAR = 4
BARS = 8


@dataclass(frozen=True)
class Song:
    """Symbolic song; ``melody`` holds (semitones above tonic, onset, duration), beats."""

    tonic: int  # MIDI note of the tonic
    tempo: float  # beats per minute
    progression: tuple[int, ...]  # scale degrees, one chord per bar, looped
    melody: tuple[tuple[int, float, float], ...]
    lead_timbre: int
    pad_timbre: int
    intro_bars: int = 0  # bars of tonic-chord pad before the song proper


def _melody(rng: np.random.Generator) -> tuple[tuple[int, float, float], ...]:
    notes, beat, degree = [], 0.0, int(rng.integers(0, 7))
    while beat < BARS * BEATS_PER_BAR:
        dur = float(rng.choice([0.5, 0.5, 1.0, 1.0, 1.5, 2.0]))
        degree = int(np.clip(degree + rng.choice([-2, -1, -1, 1, 1, 2, 3]), 0, 13))
        octave, step = divmod(degree, 7)
        notes.append((12 + 12 * octave + MAJOR_SCALE[step], beat, dur))
        beat += dur
    return tuple(notes)


def random_song(rng: np.random.Generator) -> Song:
    """A new song with every attribute drawn independently."""
    return Song(
        tonic=int(rng.integers(45, 57)),
        tempo=float(rng.uniform(90, 140)),
        progression=tuple(int(d) for d in rng.choice(list(DIATONIC_TRIADS), 4)),
        melody=_melody(rng),
        lead_timbre=int(rng.integers(len(TIMBRES))),
        pad_timbre=int(rng.integers(len(TIMBRES))),
    )


def _other_timbre(rng: np.random.Generator, current: int) -> int:
    return int((current + rng.integers(1, len(TIMBRES))) % len(TIMBRES))


def edit(song: Song, kind: str, rng: np.random.Generator) -> Song:
    """Derive a second song from ``song`` by one of the four named edits.

    Every edited song also gets a random 0-2 bar intro, so neither length nor global
    alignment gives away the label.
    """
    return replace(_edit(song, kind, rng), intro_bars=int(rng.integers(0, 3)))


def _edit(song: Song, kind: str, rng: np.random.Generator) -> Song:
    transpose = int(rng.choice([-5, -4, -3, -2, 2, 3, 4, 5]))
    tempo = song.tempo * float(rng.uniform(0.8, 1.25))
    fresh = random_song(rng)
    if kind == "cover":
        return replace(
            song,
            tonic=song.tonic + transpose,
            tempo=tempo,
            lead_timbre=_other_timbre(rng, song.lead_timbre),
            pad_timbre=_other_timbre(rng, song.pad_timbre),
        )
    if kind == "melody_lift":
        return replace(
            fresh, tonic=song.tonic + transpose, tempo=tempo, melody=song.melody
        )
    if kind == "unrelated":
        return fresh
    if kind == "same_loop":
        return replace(song, melody=fresh.melody)
    raise ValueError(f"unknown edit {kind!r}")


POSITIVE_EDITS = ("cover", "melody_lift")
NEGATIVE_EDITS = ("unrelated", "same_loop")


def _tone(midi: int, seconds: float, timbre: int, sr: int) -> FloatArray:
    t = np.arange(int(seconds * sr)) / sr
    f0 = 440.0 * 2.0 ** ((midi - 69) / 12)
    wave = sum(
        a * np.sin(2 * np.pi * f0 * (h + 1) * t)
        for h, a in enumerate(TIMBRES[timbre])
        if a and f0 * (h + 1) < sr / 2
    )
    envelope = np.minimum(1.0, t / 0.01) * np.exp(-3.0 * t / max(seconds, 1e-3))
    return np.asarray(wave * envelope, dtype=np.float64)


def render(
    song: Song, sr: int, noise_snr_db: float | None = None, seed: int = 0
) -> FloatArray:
    """Additive-synthesis waveform: lead melody + sustained triad pad + bass."""
    spb = 60.0 / song.tempo
    intro = song.intro_bars * BEATS_PER_BAR * spb
    total = intro + BARS * BEATS_PER_BAR * spb
    out = np.zeros(int(total * sr) + sr)

    def add(midi: int, onset: float, dur: float, timbre: int, gain: float) -> None:
        start = int(onset * sr)
        tone = _tone(midi, dur, timbre, sr) * gain
        out[start : start + len(tone)] += tone[: len(out) - start]

    for rel, onset, dur in song.melody:
        add(song.tonic + 12 + rel, intro + onset * spb, dur * spb, song.lead_timbre, 1.0)
    for bar in range(-song.intro_bars, BARS):
        degree = song.progression[bar % len(song.progression)] if bar >= 0 else 1
        onset, dur = intro + bar * BEATS_PER_BAR * spb, BEATS_PER_BAR * spb
        for interval in DIATONIC_TRIADS[degree]:
            add(song.tonic + interval, onset, dur, song.pad_timbre, 0.35)
        add(song.tonic - 12 + DIATONIC_TRIADS[degree][0], onset, dur, 2, 0.5)
    out = out[: int(total * sr)]
    if noise_snr_db is not None:
        rng = np.random.default_rng(seed)
        power = np.mean(out**2) / 10 ** (noise_snr_db / 10)
        out = out + rng.normal(0.0, np.sqrt(power), len(out))
    return out / (np.max(np.abs(out)) + 1e-12)
