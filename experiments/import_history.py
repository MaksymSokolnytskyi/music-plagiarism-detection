"""Import the saved v1-v4 outputs into ``results/history/`` (one-off; for provenance).

Question: what exactly did each historical version produce? The original working folders
held the numbers in CSV/JSON files next to the code. This script copies only the score
columns, normalises track titles (drops download-site tags, file extensions and local
Windows paths) and writes the files the re-analysis reads. Audio is never copied.

Usage: ``python experiments/import_history.py --source DIR`` where ``DIR`` contains the
folders ``PoC Audio``, ``CNN Audio`` and ``audio v4`` as they were on the author's disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from musicplag.data import normalize_title

OUT = Path(__file__).resolve().parents[1] / "results" / "history"

# (source file, output name, {output column: source column}, title columns)
TABLES: tuple[tuple[str, str, dict[str, str], tuple[str, ...]], ...] = (
    (
        "PoC Audio/MathDNA/data/processed/batch_results_pci.csv",
        "v1_known_pairs.csv",
        {
            "original": "original_name",
            "suspect": "comparison_name",
            "similarity_score": "similarity_score",
            "geometric_score": "geometric_score",
            "physical_score": "physical_score",
            "harmonic_score": "harmonic_score",
            "pci": "pci",
        },
        ("original", "suspect"),
    ),
    (
        "PoC Audio/MathDNA/data/processed/random_pairs_pci.csv",
        "v1_random_pairs.csv",
        {
            "original": "file1",
            "suspect": "file2",
            "similarity_score": "similarity_score",
            "geometric_score": "geometric_score",
            "physical_score": "physical_score",
            "harmonic_score": "harmonic_score",
            "pci": "pci",
        },
        ("original", "suspect"),
    ),
    (
        "CNN Audio/MathDNA/outputs/cnn_test_results.csv",
        "v3_cnn_unseen.csv",
        {
            "original": "track_a",
            "suspect": "track_b",
            "relation": "relation",
            "label": "label",
            "cnn_cosine_mean": "cnn_cosine_mean",
        },
        ("original", "suspect"),
    ),
    (
        "CNN Audio/MathDNA/outputs/cnn_only_test_results.csv",
        "v3_cnn_head_unseen.csv",
        {
            "pair": "pair",
            "label": "label",
            "cosine": "cosine_sim",
            "head_prob": "model_sim",
        },
        (),
    ),
    (
        "CNN Audio/demo/demo_cnn_original_data_results.csv",
        "v3_cnn_training_pairs.csv",
        {
            "original": "track_a",
            "suspect": "track_b",
            "label": "label",
            "cosine": "cosine_sim",
            "head_prob": "model_prob",
        },
        ("original", "suspect"),
    ),
    (
        "CNN Audio/MathDNA/outputs/cnn_math_comparison.csv",
        "v3_math_features_training_pairs.csv",
        {
            "original": "track_a",
            "suspect": "track_b",
            "label": "is_plagiarism",
            "cnn_cosine": "cnn_cosine",
            "chroma_cosine": "chroma_cosine",
            "chroma_corr": "chroma_corr",
            "mfcc_cosine": "mfcc_cosine",
            "spectral_centroid_ratio": "spectral_centroid_ratio",
            "spectral_rolloff_ratio": "spectral_rolloff_ratio",
            "rms_ratio": "rms_ratio",
            "tempo_ratio": "tempo_ratio",
            "zcr_ratio": "zcr_ratio",
        },
        ("original", "suspect"),
    ),
)
SPLITS = ("train", "val", "test")
V4_RUN = "audio v4/models/run_20260210_161925"


def _copy_table(
    src: Path, dst: Path, columns: dict[str, str], titles: tuple[str, ...]
) -> int:
    with src.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with dst.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            out = {k: row[v] for k, v in columns.items()}
            for t in titles:
                out[t] = normalize_title(out[t])
            if "pair" in out:
                out["pair"] = " vs ".join(
                    normalize_title(p) for p in out["pair"].split(" vs ")
                )
            writer.writerow(out)
    return len(rows)


def main() -> None:
    """Copy every table, the v4 split manifests and the v4 training log."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    source = parser.parse_args().source
    OUT.mkdir(parents=True, exist_ok=True)
    for rel, name, columns, titles in TABLES:
        n = _copy_table(source / rel, OUT / name, columns, titles)
        print(f"{name}: {n} rows")
    for split in SPLITS:
        cols = {"original": "track_a_path", "suspect": "track_b_path", "label": "label"}
        src = source / "audio v4/data/prepared/splits_csv" / f"{split}.csv"
        n = _copy_table(src, OUT / f"v4_split_{split}.csv", cols, ("original", "suspect"))
        print(f"v4_split_{split}.csv: {n} rows")
    run = source / V4_RUN
    history = json.loads((run / "training_history.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "model_summary.json").read_text(encoding="utf-8"))
    (OUT / "v4_training_log.json").write_text(
        json.dumps({"history": history, "summary": summary}, indent=1), encoding="utf-8"
    )
    shutil.copyfile(run / "config.yaml", OUT / "v4_run_config.yaml")
    print("v4 training log and config copied")


if __name__ == "__main__":
    main()
