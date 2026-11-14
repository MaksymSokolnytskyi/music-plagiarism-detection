"""Training-free features on real recordings, with the hackathon's own pair sets.

Two questions, one script:

* ``--pairs cnn-unseen`` — on the 27 excerpt pairs where the v3 Siamese CNN was scored
  as "unseen tracks", how do the rebuilt features compare with the CNN's saved scores?
* ``--pairs v4`` — on all 90 full-track pairs of the v4 manifest (43 derived, 47
  unrelated), how well do they separate derived from unrelated material? The features
  have no trainable parameters, so every pair is a test pair; the v4 train/val/test
  labels are reported only to compare with where the v4 network was evaluated.

The configuration is the package default, fixed before either run; nothing is tuned on
these pairs. Audio is copyrighted and not in the repository: ``--audio-dir`` must hold
the recordings (any sub-folder layout; files are matched by normalised title).
Writes ``results/real/<pairs>/scores.csv`` and ``REPORT.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from musicplag import metrics, spectral
from musicplag.config import CONFIG
from musicplag.data import normalize_title
from musicplag.features import (
    DEFAULT_FEATURES,
    LocalChromaMatch,
    PairContext,
    analyze_pair,
)

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "results" / "history"
SEED = 0
AUDIO_SUFFIXES = (".wav", ".mp3", ".flac")
CQT_FIELDS = {"sample_rate", "hop_length", "fmin", "bins_per_octave", "n_octaves"}
EXPLORATORY = LocalChromaMatch()


def _manifest(name: str) -> list[dict[str, str]]:
    """Rows with ``original``, ``suspect``, ``label`` (0/1) and set-specific extras."""
    if name == "cnn-unseen":
        with (HIST / "v3_cnn_unseen.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        with (HIST / "v3_cnn_head_unseen.csv").open(encoding="utf-8") as fh:
            head = {r["pair"]: r["head_prob"] for r in csv.DictReader(fh)}
        for r in rows:
            r["cnn_cosine"] = r.pop("cnn_cosine_mean")
            r["cnn_head"] = head[f"{r['original']} vs {r['suspect']}"]
        assert len(rows) == 27, "expected the 27-pair unseen set"
        return rows
    rows = []
    for split in ("train", "val", "test"):
        with (HIST / f"v4_split_{split}.csv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                label = str(int(r["label"] == "plagiarism"))
                rows.append({**r, "label": label, "v4_split": split})
    assert len(rows) == 90, "expected the 90-pair v4 manifest"
    return rows


def _cqt_job(job: tuple[str, str, str]) -> tuple[str, str]:
    """Worker: decode one file and cache its CQT; returns ``(title, cache path)``."""
    import librosa

    title, path, cache = job
    if not Path(cache).exists():
        y, _ = librosa.load(path, sr=CONFIG.sample_rate, mono=True)
        assert len(y) > 4 * CONFIG.sample_rate, f"{path}: shorter than 4 s"
        np.save(cache, spectral.cqt(np.asarray(y, dtype=np.float64), CONFIG))
    return title, cache


def _cqts(
    titles: set[str], audio_dir: Path, cache_dir: Path, workers: int
) -> dict[str, str]:
    """Cached CQT path per title; computed in parallel on first use."""
    files = [p for p in audio_dir.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES]
    index: dict[str, Path] = {}
    for f in sorted(files, key=lambda p: (p.suffix.lower() != ".wav", str(p))):
        index.setdefault(normalize_title(f.name), f)  # WAV preferred over MP3
    missing = titles - set(index)
    assert not missing, f"audio not found for: {sorted(missing)[:5]}"
    front_end = CONFIG.model_dump_json(include=CQT_FIELDS)
    fingerprint = hashlib.sha1(front_end.encode()).hexdigest()[:8]
    jobs = []
    for t in sorted(titles):
        key = hashlib.sha1(f"{t}|{fingerprint}".encode()).hexdigest()[:16]
        jobs.append((t, str(index[t]), str(cache_dir / f"{key}.npy")))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(_cqt_job, jobs))


def _table(rows: list[dict[str, object]], names: list[str]) -> list[str]:
    y = [int(str(r["label"])) for r in rows]
    lines = [
        "| score | AUC | 95% CI | p (AUC>0.5) | Cohen's d |",
        "|---|---|---|---|---|",
    ]
    for name in names:
        s = metrics.summarize([float(str(r[name])) for r in rows], y, seed=SEED)
        kind = "v3 CNN" if name.startswith("cnn") else "rebuilt"
        lines.append(
            f"| `{name}` ({kind}) | **{s.auc:.3f}** | [{s.ci_low:.2f}, {s.ci_high:.2f}] "
            f"| {s.p_value:.3f} | {s.cohens_d:+.2f} |"
        )
    return lines


def main() -> None:
    """Compute CQTs (cached, parallel), score every pair, write scores and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", choices=("cnn-unseen", "v4"), required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/cqt"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    start = time.time()
    pairs = _manifest(args.pairs)
    titles = {r["original"] for r in pairs} | {r["suspect"] for r in pairs}
    cache = _cqts(titles, args.audio_dir, args.cache, args.workers)
    print(f"{len(titles)} CQTs ready after {time.time() - start:.0f} s", flush=True)

    rows: list[dict[str, object]] = []
    extras = ("relation", "v4_split", "cnn_cosine", "cnn_head")
    for p in pairs:
        ctx = PairContext(np.load(cache[p["original"]]), np.load(cache[p["suspect"]]))
        report = analyze_pair(ctx)
        row: dict[str, object] = {
            "original": p["original"],
            "suspect": p["suspect"],
            "label": int(p["label"]),
            "transposition": report.transposition_semitones,
        }
        row.update({k: p[k] for k in extras if k in p})
        row.update(report.features)
        row[EXPLORATORY.name] = EXPLORATORY(ctx)
        rows.append(row)
        print(f"{len(rows)}/{len(pairs)} {p['original'][:40]}", flush=True)

    out = ROOT / "results" / "real" / args.pairs
    out.mkdir(parents=True, exist_ok=True)
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()}
            for r in rows
        )

    features = [f.name for f in DEFAULT_FEATURES]
    n_pos = sum(int(str(r["label"])) for r in rows)
    if args.pairs == "cnn-unseen":
        intro = (
            "The 27 pairs the v3 CNN was evaluated on as 'unseen tracks': 11 derived "
            "(6 plagiarism cases, 5 remakes) and 16 unrelated pairings, curated 9-53 s "
            "excerpts. CNN scores are the ones v3 saved."
        )
        names = ["cnn_cosine", "cnn_head", *features]
    else:
        intro = (
            f"All 90 pairs of the v4 manifest: {n_pos} derived (covers, remakes, "
            f"borrowings) and {len(rows) - n_pos} unrelated, full-length recordings "
            "(median about 4 minutes). No CNN per-pair scores were logged for these."
        )
        names = features
    lines = [
        f"# Real pairs ({args.pairs}): training-free features",
        "",
        f"Generated by `experiments/real_pairs.py --pairs {args.pairs}` "
        f"({(time.time() - start) / 60:.1f} min). {intro} The features use the package "
        "default configuration and were not tuned on these pairs.",
        "",
        *_table(rows, names),
        "",
        "Five features are tested, so the Bonferroni level is "
        f"{0.05 / len(features):.3f}."
        f" With {n_pos} positives, read differences between neighbouring rows as "
        "directions, not rankings.",
        "",
    ]
    lines += [
        "## Exploratory: local matching",
        "",
        "`chroma_local` slides a "
        f"{CONFIG.local_window_seconds:.0f} s window over the original and finds its "
        "best subsequence-DTW match anywhere in the suspect, under all 12 "
        "transpositions. It was designed *after* the full-track table showed global "
        "alignment failing, and is evaluated on the same pairs, so treat it as a "
        "hypothesis check, not a result.",
        "",
        *_table(rows, [EXPLORATORY.name]),
        "",
    ]
    if args.pairs == "v4":
        test = [r for r in rows if r["v4_split"] == "test"]
        lines += [
            "## The v4 test split alone",
            "",
            f"The {len(test)} pairs the v4 network held out "
            "(no v4 test number was ever logged):",
            "",
            *_table(test, [*features, EXPLORATORY.name]),
            "",
        ]
    text = "\n".join(lines) + "\n"
    (out / "REPORT.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
