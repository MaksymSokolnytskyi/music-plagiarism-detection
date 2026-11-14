"""Re-derive every historical headline from the saved per-pair scores.

Question: how well did each version actually separate derived from unrelated pairs,
measured the same way for all of them (AUC with a bootstrap 95% CI and a permutation
p-value), and do the recomputed numbers agree with what the old reports claimed?

Reads only ``results/history/`` (see ``import_history.py``); writes
``results/history/REPORT.md``. Seeded; rerunning reproduces the file byte for byte.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from musicplag import metrics
from musicplag.data import load_pairs, shared_tracks

HIST = Path(__file__).resolve().parents[1] / "results" / "history"
SEED = 0


def _rows(name: str) -> list[dict[str, str]]:
    with (HIST / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _summary_row(label: str, scores: list[float], y: list[int], note: str = "") -> str:
    s = metrics.summarize(scores, y, seed=SEED)
    assert 0.0 <= s.ci_low <= s.auc <= s.ci_high <= 1.0, "inconsistent bootstrap"
    return (
        f"| {label} | {s.n_pos}/{s.n_neg} | **{s.auc:.3f}** | "
        f"[{s.ci_low:.2f}, {s.ci_high:.2f}] | {s.p_value:.3f} | "
        f"{s.cohens_d:+.2f} | {note} |"
    )


HEADER = (
    "| score | pos/neg | AUC | 95% CI | p (AUC>0.5) | Cohen's d | note |\n"
    "|---|---|---|---|---|---|---|"
)


def v1_section() -> list[str]:
    known, random = _rows("v1_known_pairs.csv"), _rows("v1_random_pairs.csv")
    assert (len(known), len(random)) == (16, 50), "unexpected v1 table sizes"
    y = [1] * len(known) + [0] * len(random)
    out = ["## v1 — PoC: CQT → per-track NMF → DTW + log-polar + Markov/KL", "", HEADER]
    for col, note in (
        ("similarity_score", "weighted 3-pillar score"),
        ("harmonic_score", "DTW + Markov on NMF activations"),
        ("geometric_score", "log-polar patch correlation"),
        ("physical_score", "LPC pole-zero timbre"),
        ("pci", "z-scored against the random pairs"),
    ):
        scores = [float(r[col]) for r in known + random]
        out.append(_summary_row(f"`{col}`", scores, y, note))
    phys = np.array([float(r["physical_score"]) for r in random])
    out += [
        "",
        f"The LPC channel sits at {phys.mean():.1f} ± {phys.std(ddof=1):.1f} on *random* "
        "pairs: it says every pair of songs is ~96% similar. The original report gave "
        "AUC 0.631 for the PCI; the recomputation agrees. Positives are 16 curated "
        "plagiarism/remake pairs, negatives are 50 random pairings, so the negatives are "
        "easy — this is an optimistic setting, not a hard one.",
        "",
    ]
    return out


def v3_section() -> list[str]:
    unseen, head = _rows("v3_cnn_unseen.csv"), _rows("v3_cnn_head_unseen.csv")
    seen = _rows("v3_cnn_training_pairs.csv")
    assert len(unseen) == len(head) == 27 and len(seen) == 90, "unexpected v3 sizes"
    y_un = [int(r["label"]) for r in unseen]
    y_head = [int(r["label"]) for r in head]
    y_seen = [int(r["label"]) for r in seen]
    probs = np.array([float(r["head_prob"]) for r in head])
    acc = float(np.mean((probs >= 0.5) == np.array(y_head, dtype=bool)))
    pos = np.array([float(r["cnn_cosine_mean"]) for r in unseen if r["label"] == "1"])
    neg = np.array([float(r["cnn_cosine_mean"]) for r in unseen if r["label"] == "0"])
    out = [
        "## v3 — Siamese VGGish CNN + XGBoost",
        "",
        HEADER,
        _summary_row(
            "embedding cosine, training pairs",
            [float(r["cosine"]) for r in seen],
            y_seen,
            "pairs the network was trained on",
        ),
        _summary_row(
            "classifier head, training pairs",
            [float(r["head_prob"]) for r in seen],
            y_seen,
            "pairs the network was trained on",
        ),
        _summary_row(
            "embedding cosine, unseen tracks",
            [float(r["cnn_cosine_mean"]) for r in unseen],
            y_un,
            "mean over segments",
        ),
        _summary_row("classifier head, unseen tracks", probs.tolist(), y_head, ""),
        "",
        f"On unseen tracks the embedding cosine is {pos.mean():.3f} "
        f"± {pos.std(ddof=1):.3f} "
        f"for derived pairs and {neg.mean():.3f} ± {neg.std(ddof=1):.3f} for unrelated "
        "ones: the network maps every pair of songs to 'very similar'. Accuracy of the "
        f"classifier head at 0.5 is {acc:.1%}; majority class "
        f"{max(np.mean(y_head), 1 - np.mean(y_head)):.1%}. (The hackathon slides quoted "
        "53.1% accuracy and AUC 0.508 on 32 unseen pairs; the per-pair file that "
        "survives has 27 of them. Both versions of the number say the same thing.)",
        "",
        "Interpretable features on the 40 training pairs where they were logged "
        "(ratios scored as similarity `-|ln r|`, so 1.0 = identical):",
        "",
        HEADER,
    ]
    math_rows = _rows("v3_math_features_training_pairs.csv")
    y_math = [int(r["label"]) for r in math_rows]
    for col in ("chroma_cosine", "chroma_corr", "mfcc_cosine"):
        out.append(_summary_row(f"`{col}`", [float(r[col]) for r in math_rows], y_math))
    for col in (
        "rms_ratio",
        "spectral_rolloff_ratio",
        "spectral_centroid_ratio",
        "zcr_ratio",
        "tempo_ratio",
    ):
        sims = [-abs(math.log(max(float(r[col]), 1e-9))) for r in math_rows]
        out.append(_summary_row(f"`{col}`", sims, y_math))
    out.append("")
    return out


def v4_section() -> list[str]:
    def split(name: str) -> list:  # type: ignore[type-arg]
        return load_pairs(
            HIST / f"v4_split_{name}.csv", "original", "suspect", "label", ["plagiarism"]
        )

    train, val, test = split("train"), split("val"), split("test")
    leak_test, leak_val = shared_tracks(train, test), shared_tracks(train, val)
    log = json.loads((HIST / "v4_training_log.json").read_text(encoding="utf-8"))
    hist = log["history"]
    best = int(np.argmin(hist["val_loss"]))
    prevalence = sum(p.positive for p in val) / len(val)
    entropy = -(
        prevalence * math.log(prevalence) + (1 - prevalence) * math.log(1 - prevalence)
    )
    out = [
        "## v4 — multi-chunk CNN + interpretable features (training run)",
        "",
        f"* Split sizes {len(train)}/{len(val)}/{len(test)} pairs; recordings shared "
        f"train∩val = {len(leak_val)}, train∩test = {len(leak_test)} — the split is "
        "track-disjoint, so there is no recording-level leakage.",
        f"* Best validation loss {hist['val_loss'][best]:.3f} (epoch {best + 1}); a "
        f"constant predictor at the validation prevalence scores {entropy:.3f}.",
        f"* Training loss fell {hist['train_loss'][0]:.3f} → "
        f"{hist['train_loss'][-1]:.3f} "
        f"while validation loss rose to {hist['val_loss'][-1]:.3f}: memorisation, not "
        "learning.",
        f"* Validation accuracy never exceeded {max(hist['val_accuracy']):.1%} "
        f"(majority class: {max(prevalence, 1 - prevalence):.1%}).",
        "* No test-set evaluation was logged for this run, so there is no v4 test "
        "number.",
        "",
    ]
    assert not leak_test and not leak_val, "historical split leaks recordings"
    return out


def main() -> None:
    """Write ``results/history/REPORT.md``."""
    lines = [
        "# Historical results, recomputed",
        "",
        "Generated by `experiments/reanalyze_history.py` from the per-pair scores each "
        "version saved. Every AUC uses the same estimator (Mann-Whitney), a stratified "
        f"bootstrap ({metrics.N_BOOTSTRAP} resamples) and a one-sided permutation test "
        f"({metrics.N_PERMUTATIONS} shuffles), seed {SEED}.",
        "",
        "**v2 (MVP) is not in this table.** Its submission claimed precision 87.3%, "
        "recall 82.1% and AUC-ROC 0.91 on an 'SMP test set', but no per-pair scores, "
        "predictions or run logs for that evaluation exist, and the pair files in that "
        "folder hold 16 positives and 16 negatives. The claim cannot be reproduced, so "
        "it is not used anywhere in this repository.",
        "",
        *v1_section(),
        *v3_section(),
        *v4_section(),
    ]
    report = "\n".join(lines) + "\n"
    (HIST / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
