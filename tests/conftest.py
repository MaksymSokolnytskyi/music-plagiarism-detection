"""Shared fixtures: a light configuration so tests run in seconds."""

from __future__ import annotations

import pytest

from musicplag.config import CONFIG, Config


@pytest.fixture(scope="session")
def fast_cfg() -> Config:
    """11 kHz, same ~43 frames/s as the default; enough bandwidth for C1..B7."""
    return CONFIG.model_copy(update={"sample_rate": 11025, "hop_length": 256})
