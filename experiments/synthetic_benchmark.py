"""Which features survive which edits, and does the joint-NMF fix matter?

Question: on material where the ground truth is known exactly, how well does each
feature separate derived pairs (``cover``, ``melody_lift``) from unrelated ones
(``unrelated``, ``same_loop``)? And does fitting one NMF dictionary for both
recordings (v4 fix) beat the per-track fits of v1-v3 on the same pairs?

Every suspect is rendered with white noise at 20 dB SNR. Seeded; writes
``results/synthetic/scores.csv`` and ``results/synthetic/REPORT.md``.

Caveat, stated in the report too: additive synthesis with exact pitches is far easier
than produced recordings (vocals, drums, mixing, reverb). Passing here is necessary,
not sufficient.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from musicplag import dtw, markov, metrics, nmf, spectral, synth
from musicplag.config import CONFIG
from musicplag.features import DEFAULT_FEATURES, PairContext, PairFeature

OUT = Path(__file__).resolve().parents[1] / "results" / "synthetic"
N_SONGS = 30
SNR_DB = 20.0
SEED = 2026
CFG = CONFIG.model_copy(update={"sample_rate": 11025, "hop_length": 256})


@dataclass(frozen=True)
class SeparateNMFMarkov:
    """v1-v3 Markov feature: per-track NMF, flattened-matrix KL (ablation baseline)."""

    name: str = "markov_separate_kl"

    def __call__(self, ctx: PairContext) -> float:
        k, it = ctx.cfg.nmf_components, ctx.cfg.nmf_iterations
        h_a = nmf.nmf(np.log1p(ctx.pooled_a), k, it, ctx.cfg.seed).H
        h_b = nmf.nmf(np.log1p(ctx.pooled_b), k, it, ctx.cfg.seed).H
        alpha = ctx.cfg.markov_smoothing
        p_a = markov.transition_matrix(markov.dominant_states(h_a), k, alpha)
        p_b = markov.transition_matrix(markov.dominant_states(h_b), k, alpha)
        return markov.legacy_kl_similarity(p_a, p_b)


@dataclass(frozen=True)
class SeparateNMFDTW:
    """v1 structural feature: DTW between activations of separately fitted NMFs."""

    name: str = "nmf_dtw_separate"

    def __call__(self, ctx: PairContext) -> float:
        k, it = ctx.cfg.nmf_components, ctx.cfg.nmf_iterations
        h_a = nmf.nmf(np.log1p(ctx.pooled_a), k, it, ctx.cfg.seed).H
        h_b = nmf.nmf(np.log1p(ctx.pooled_b), k, it, ctx.cfg.seed).H
        return dtw.dtw(dtw.cosine_cost(h_a, h_b), ctx.cfg.dtw_band).similarity


@dataclass(frozen=True)
class JointNMFDTW:
    """Same DTW, but on activations over one shared, key-aligned dictionary."""

    name: str = "nmf_dtw_joint"

    def __call__(self, ctx: PairContext) -> float:
        shift = ctx.transposition if ctx.transposition <= 6 else ctx.transposition - 12
        b = spectral.transpose_cqt(ctx.pooled_b, shift, ctx.cfg.bins_per_octave)
        _, h_a, h_b = nmf.joint_nmf(
            np.log1p(ctx.pooled_a),
            np.log1p(b),
            ctx.cfg.nmf_components,
            ctx.cfg.nmf_iterations,
            ctx.cfg.seed,
        )
        return dtw.dtw(dtw.cosine_cost(h_a, h_b), ctx.cfg.dtw_band).similarity


FEATURES: tuple[PairFeature, ...] = (
    *DEFAULT_FEATURES,
    SeparateNMFMarkov(),
    SeparateNMFDTW(),
    JointNMFDTW(),
)
EDITS = (*synth.POSITIVE_EDITS, *synth.NEGATIVE_EDITS)


def score_pairs() -> list[dict[str, object]]:
    """Render every original and its four edits; score every feature on every pair."""
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for song_id in range(N_SONGS):
        original = synth.random_song(rng)
        cqt_a = spectral.cqt(synth.render(original, CFG.sample_rate), CFG)
        for kind in EDITS:
            suspect = synth.edit(original, kind, rng)
            y_b = synth.render(suspect, CFG.sample_rate, SNR_DB, seed=song_id)
            ctx = PairContext(cqt_a, spectral.cqt(y_b, CFG), CFG)
            row: dict[str, object] = {
                "song": song_id,
                "edit": kind,
                "label": int(kind in synth.POSITIVE_EDITS),
                "true_transposition": suspect.tonic - original.tonic,
                "found_transposition": -(
                    ctx.transposition
                    if ctx.transposition <= 6
                    else ctx.transposition - 12
                ),
            }
            row.update({f.name: f(ctx) for f in FEATURES})
            assert all(np.isfinite(float(row[f.name])) for f in FEATURES), row  # type: ignore[arg-type]
            rows.append(row)
        print(f"song {song_id + 1}/{N_SONGS}", flush=True)
    return rows


def _auc(rows: list[dict[str, object]], feature: str, pos: str, neg: str) -> float:
    sel = [r for r in rows if r["edit"] in (pos, neg)]
    return metrics.auc([float(r[feature]) for r in sel], [r["edit"] == pos for r in sel])  # type: ignore[arg-type]


def report(rows: list[dict[str, object]], seconds: float) -> str:
    """Markdown report: overall AUC with CI, then the edit-by-edit AUC grid."""
    names = [f.name for f in FEATURES]
    y = [int(r["label"]) for r in rows]  # type: ignore[call-overload]
    pairs = list(product(synth.POSITIVE_EDITS, synth.NEGATIVE_EDITS))
    keyed = [r for r in rows if r["edit"] in ("cover", "melody_lift")]
    key_acc = np.mean(
        [r["true_transposition"] == r["found_transposition"] for r in keyed]
    )
    lines = [
        "# Synthetic benchmark",
        "",
        f"Generated by `experiments/synthetic_benchmark.py` (seed {SEED}, {N_SONGS} "
        f"originals × {len(EDITS)} edits = {len(rows)} pairs, suspects at "
        f"{SNR_DB:.0f} dB SNR with a random 0-2 bar intro, {seconds / 60:.1f} min).",
        "",
        "**Read this as a unit test of the mathematics, not as a detector benchmark.** "
        "Additive synthesis with exact pitches is far easier than produced recordings.",
        "",
        "## Derived vs unrelated, all edits pooled",
        "",
        "| feature | AUC | 95% CI | Cohen's d |",
        "|---|---|---|---|",
    ]
    for name in names:
        s = metrics.summarize([float(r[name]) for r in rows], y, seed=SEED)  # type: ignore[arg-type]
        lines.append(
            f"| `{name}` | **{s.auc:.3f}** | [{s.ci_low:.2f}, {s.ci_high:.2f}] | "
            f"{s.cohens_d:+.2f} |"
        )
    header = " | ".join(f"{p} vs {n}" for p, n in pairs)
    lines += [
        "",
        "## AUC for each positive edit against each negative edit",
        "",
        f"| feature | {header} |",
        "|---|" + "---|" * len(pairs),
    ]
    for name in names:
        cells = " | ".join(f"{_auc(rows, name, p, n):.2f}" for p, n in pairs)
        lines.append(f"| `{name}` | {cells} |")
    lines += [
        "",
        f"Transposition recovered exactly (optimal transposition index) in {key_acc:.0%} "
        "of positive pairs.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Score, save the per-pair CSV, write the report."""
    start = time.time()
    rows = score_pairs()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()}
            for r in rows
        )
    text = report(rows, time.time() - start)
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
