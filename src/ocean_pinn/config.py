"""Default settings for data preparation and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    data_dir: Path = Path("data")
    # Latitude of the KEO mooring (degrees N), used to convert pressure to height.
    latitude: float = 32.3
    # The 1 m sensor has no pressure record, so its pressure (dbar) is fixed.
    surface_pressure: float = 1.0
    # Indices of the 21 nominal levels to exclude: 25 m and 150 m, whose
    # records both begin with a gap.
    drop_levels: tuple[int, ...] = (5, 12)
    # Points are ordered level by level from the surface down, so the first
    # `n_train` cover the upper levels and the rest (the test set) the deeper ones.
    n_train: int = 2_998_271


@dataclass
class LossWeights:
    data: float = 10.0
    physics: float = 0.1
    bc_z0: float = 0.1
    bc_z1: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 1e-6
    batch_size: int = 4096
    seed: int = 2025
    loss_weights: LossWeights = field(default_factory=LossWeights)
    checkpoint_path: Path = Path("checkpoints/checkpoint.pt")
    checkpoint_every: int = 100  # batches
    log_every: int = 10  # batches
