"""Analysis configuration: the single source of truth for every numeric parameter.

One frozen :class:`Config` instance, :data:`CONFIG`, is created on import. Experiments
derive variants with ``CONFIG.model_copy(update={...})`` instead of scattering
constants through the code.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, computed_field


class Config(BaseModel):
    """Parameters of the time-frequency front end and of each similarity feature."""

    model_config = ConfigDict(frozen=True)

    sample_rate: int = Field(22050, gt=0, description="Analysis sample rate, Hz.")
    hop_length: int = Field(512, gt=0, description="CQT hop, samples (~23 ms at 22 kHz).")
    fmin: float = Field(32.703, gt=0, description="Lowest CQT bin, Hz (C1).")
    bins_per_octave: int = Field(12, gt=0, description="One bin per semitone.")
    n_octaves: int = Field(7, gt=0, description="C1..B7 covers bass to lead melody.")

    pool_frames: int = Field(
        8, ge=1, description="Average-pool CQT frames before DTW/NMF (~5.4 frames/s)."
    )
    dtw_band: float = Field(
        0.25, gt=0, le=1, description="Sakoe-Chiba half-width as a fraction of length."
    )
    nmf_components: int = Field(12, ge=2, description="Spectral atoms in the joint NMF.")
    nmf_iterations: int = Field(200, ge=1, description="Multiplicative-update steps.")
    markov_smoothing: float = Field(
        0.5, gt=0, description="Additive (Lidstone) smoothing of transition counts."
    )
    ssm_size: int = Field(48, ge=8, description="SSMs are resampled to this square size.")
    mellin_bins: int = Field(32, ge=8, description="Log-spaced tempo-modulation bins.")
    seed: int = Field(0, description="Seed for every stochastic step (NMF init).")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_bins(self) -> int:
        """Total CQT bins."""
        return self.bins_per_octave * self.n_octaves


CONFIG = Config()
