"""Loading and preprocessing of the KEO mooring temperature record.

The raw record is a (time, level) grid of temperature and pressure. It is cleaned,
pressure is converted to height with TEOS-10, and the grid is flattened into
normalised (z, t) -> T points for the network.
"""

from __future__ import annotations

import dataclasses
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import gsw
import numpy as np
import pandas as pd
import torch
import xarray as xr

from .config import DataConfig

# Missing samples are stored as 1e35. The files declare this only as a global
# attribute, so xarray leaves them unmasked.
FILL_THRESHOLD = 1e10


@dataclass(frozen=True)
class Affine:
    """Affine map between physical values and their normalised form."""

    offset: float
    scale: float

    def normalize(self, values):
        return (values - self.offset) / self.scale

    def denormalize(self, values):
        return values * self.scale + self.offset


@dataclass
class MooringRecord:
    """Cleaned mooring record on a (time, level) grid."""

    time: np.ndarray  # (n_time,) datetime64
    z: np.ndarray  # (n_time, n_level) height in m, negative below the sea surface
    temperature: np.ndarray  # (n_time, n_level) in °C


@dataclass
class Observations:
    """The record as normalised (z, t) -> T points, ordered level by level."""

    x: torch.Tensor  # (N, 2) normalised (z, t)
    temperature: torch.Tensor  # (N,) normalised temperature
    z_scale: Affine  # z in m
    t_scale: Affine  # t in seconds since start_time
    temperature_scale: Affine  # temperature in °C
    start_time: np.datetime64

    def __len__(self) -> int:
        return len(self.x)

    def to(self, device: torch.device | str) -> Observations:
        return dataclasses.replace(
            self, x=self.x.to(device), temperature=self.temperature.to(device)
        )

    def times(self) -> np.ndarray:
        """Timestamp of each point."""
        seconds = self.t_scale.denormalize(self.x[:, 1].cpu().double().numpy())
        return self.start_time + np.round(seconds).astype("timedelta64[s]")

    def depths(self) -> np.ndarray:
        """Depth of each point in m, positive down."""
        return -self.z_scale.denormalize(self.x[:, 0].cpu().double().numpy())


@dataclass
class PointSet:
    """Observations and collocation points for one data split."""

    x: torch.Tensor  # (n, 2) normalised (z, t) of the observations
    temperature: torch.Tensor  # (n,) normalised observed temperature
    x_colloc: torch.Tensor  # (n, 2) points where the PDE residual is penalised

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> PointSet:
        return PointSet(self.x[idx], self.temperature[idx], self.x_colloc[idx])


def temperature_file(data_dir: Path) -> Path:
    """Path to the temperature NetCDF file, extracting it from its zip archive if needed.

    The file (~140 MB) exceeds GitHub's 100 MB limit, so the repository stores it zipped.
    """
    path = data_dir / "temp.cdf"
    if not path.exists():
        partial = path.with_name(path.name + ".part")
        with zipfile.ZipFile(data_dir / "temp.cdf.zip") as archive:
            (member,) = archive.namelist()
            with archive.open(member) as src, open(partial, "wb") as dst:
                shutil.copyfileobj(src, dst)
        partial.replace(path)
    return path


def fill_gaps(values: np.ndarray) -> np.ndarray:
    """Replace missing samples with the previous valid sample along the time axis (axis 0).

    Gaps at the start of a record, which have no previous sample, take the first valid one.
    """
    values = np.where(values > FILL_THRESHOLD, np.nan, values)
    frame = pd.DataFrame(values.reshape(len(values), -1))
    filled = frame.ffill().bfill().to_numpy().reshape(values.shape)
    if np.isnan(filled).any():
        raise ValueError("Cannot fill a record that has no valid samples.")
    return filled


def load_record(cfg: DataConfig) -> MooringRecord:
    """Load the KEO temperature and pressure records and clean them."""
    with xr.open_dataset(temperature_file(Path(cfg.data_dir)), engine="netcdf4") as ds:
        n_time = ds.sizes["time"]
        time = ds["time"].to_numpy()
        temperature = ds["T_20"].to_numpy().reshape(n_time, -1)
        pressure = ds["P_1"].to_numpy().reshape(n_time, -1).copy()

    pressure[:, 0] = cfg.surface_pressure
    drop = list(cfg.drop_levels)
    temperature = fill_gaps(np.delete(temperature, drop, axis=1))
    pressure = fill_gaps(np.delete(pressure, drop, axis=1))
    # The mooring line moves, so sensor depths come from the measured pressure
    # rather than the nominal depths.
    z = gsw.z_from_p(pressure, cfg.latitude)
    return MooringRecord(time=time, z=z, temperature=temperature)


def build_observations(record: MooringRecord) -> Observations:
    """Flatten the (time, level) grid level by level and normalise it.

    z and t are min-max scaled to [0, 1] and temperature is standardised.
    """
    n_level = record.temperature.shape[1]
    t = np.tile((record.time - record.time[0]) / np.timedelta64(1, "s"), n_level)
    z = record.z.T.ravel()
    temperature = record.temperature.T.ravel()

    z_scale = Affine(offset=float(z.min()), scale=float(z.max() - z.min()))
    t_scale = Affine(offset=float(t.min()), scale=float(t.max() - t.min()))
    temperature_scale = Affine(
        offset=float(temperature.mean(dtype=np.float64)),
        scale=float(temperature.std(dtype=np.float64, ddof=1)),
    )
    x = np.stack([z_scale.normalize(z), t_scale.normalize(t)], axis=1)
    return Observations(
        x=torch.as_tensor(x, dtype=torch.float32),
        temperature=torch.as_tensor(temperature_scale.normalize(temperature), dtype=torch.float32),
        z_scale=z_scale,
        t_scale=t_scale,
        temperature_scale=temperature_scale,
        start_time=record.time[0],
    )


def load_observations(cfg: DataConfig) -> Observations:
    """Load, clean and normalise the mooring record."""
    return build_observations(load_record(cfg))


def sample_collocation_points(n: int, seed: int) -> torch.Tensor:
    """Uniform random (z, t) points in the unit square."""
    generator = torch.Generator().manual_seed(seed)
    z = torch.rand(n, generator=generator)
    t = torch.rand(n, generator=generator)
    return torch.stack([z, t], dim=1)


def train_test_split(
    observations: Observations, n_train: int, seed: int
) -> tuple[PointSet, PointSet]:
    """Split the points at index `n_train`, giving each split its own collocation points."""
    x_colloc = sample_collocation_points(len(observations), seed).to(observations.x.device)
    points = PointSet(observations.x, observations.temperature, x_colloc)
    return points[:n_train], points[n_train:]
