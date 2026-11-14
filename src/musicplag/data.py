"""Pair manifests and the track-disjointness check that guards against leakage.

A pair set is only an honest test if no *recording* in it appears in training: a model
that has seen "Thunderstruck" in a positive pair can recognise the recording instead of
the borrowing. :func:`shared_tracks` is the check; the historical v4 split passes it.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_SOURCE_TAG = re.compile(r"_?\(z3\.fm\)", re.IGNORECASE)
_EXTENSION = re.compile(r"\.(wav|mp3|flac|ogg|m4a)$", re.IGNORECASE)
_NON_WORD = re.compile(r"[\W_]+")  # keeps Cyrillic, Hangul, accents


def normalize_title(raw: str) -> str:
    """Canonical key for a recording: lower-case words, no folder, extension or tags."""
    name = raw.replace("\\", "/").rsplit("/", 1)[-1].lower()
    name = _EXTENSION.sub("", _SOURCE_TAG.sub("", name))
    return _NON_WORD.sub(" ", name).strip()


class Pair(BaseModel):
    """One labelled comparison: is ``suspect`` derived from ``original``?"""

    model_config = ConfigDict(frozen=True)

    original: str = Field(description="Normalised title of the earlier recording.")
    suspect: str = Field(description="Normalised title of the later recording.")
    positive: bool = Field(description="True for plagiarism, cover, remake or sample.")


def load_pairs(
    path: Path, a: str, b: str, label: str, positive: Iterable[str]
) -> list[Pair]:
    """Read a CSV manifest; ``positive`` lists the label values that mean 'derived'."""
    pos = set(positive)
    with path.open(encoding="utf-8") as fh:
        return [
            Pair(
                original=normalize_title(row[a]),
                suspect=normalize_title(row[b]),
                positive=row[label] in pos,
            )
            for row in csv.DictReader(fh)
        ]


def tracks(pairs: Iterable[Pair]) -> set[str]:
    """All recordings referenced by ``pairs``."""
    return {t for p in pairs for t in (p.original, p.suspect)}


def shared_tracks(train: Iterable[Pair], test: Iterable[Pair]) -> set[str]:
    """Recordings present in both sets — must be empty for an unseen-track test."""
    return tracks(train) & tracks(test)
