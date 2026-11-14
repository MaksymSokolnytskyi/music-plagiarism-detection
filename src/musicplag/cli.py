"""Command line: ``musicplag ORIGINAL SUSPECT`` prints a JSON evidence report.

Needs the ``audio`` extra (librosa) to decode files. The report lists feature values,
not a verdict: no threshold in this repository has been validated on enough real pairs
to call anything plagiarism (see ``results/``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from musicplag.config import CONFIG
from musicplag.features import PairContext, analyze_pair


def _load(path: Path, seconds: float | None) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "install the audio extra: pip install 'musicplag[audio]'"
        ) from exc
    y, _ = librosa.load(path, sr=CONFIG.sample_rate, mono=True, duration=seconds)
    return np.asarray(y, dtype=np.float64)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, analyse the pair, print JSON."""
    parser = argparse.ArgumentParser(prog="musicplag", description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("suspect", type=Path)
    parser.add_argument(
        "--seconds", type=float, default=60.0, help="analyse the first N seconds"
    )
    args = parser.parse_args(argv)
    ctx = PairContext.from_audio(
        _load(args.original, args.seconds), _load(args.suspect, args.seconds)
    )
    report = analyze_pair(ctx)
    print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
