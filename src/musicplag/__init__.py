"""musicplag — interpretable music-plagiarism analysis.

Layers, lowest first: ``spectral`` (CQT, chroma, Tonnetz) → ``dtw``, ``nmf``,
``markov``, ``structure``, ``invariants`` (the mathematics) → ``features`` (pair
analysis) → ``metrics``/``data`` (evaluation) → ``cli``. ``synth`` generates
ground-truth test material.
"""

from musicplag.config import CONFIG, Config
from musicplag.features import DEFAULT_FEATURES, PairContext, PairReport, analyze_pair

__all__ = [
    "CONFIG",
    "DEFAULT_FEATURES",
    "Config",
    "PairContext",
    "PairReport",
    "analyze_pair",
]
__version__ = "0.4.0"
