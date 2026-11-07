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
# Temperature quality codes accepted as measurements. 0 marks missing samples, and
# the record also holds a few thousand samples coded 4 and 5.
GOOD_QUALITY = (1, 2, 3)


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
    # (n_time, n_level) True where the sample was measured rather than filled in.
    observed: np.ndarray | None = None


@dataclass
class Observations:
    """The record as normalised (z, t) -> T points, ordered level by level."""

    x: torch.Tensor  # (N, 2) normalised (z, t)
    temperature: torch.Tensor  # (N,) normalised temperature
    z_scale: Affine  # z in m
    t_scale: Affine  # t in seconds since start_time
    temperature_scale: Affine  # temperature in °C
    start_time: np.datetime64
    n_time: int  # samples per level
    n_train_times: int  # the first n_train_times samples of each level are for training

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
    # Deepest and shallowest sensors at the time of each observation.
    x_bottom: torch.Tensor  # (n, 2)
    temperature_bottom: torch.Tensor  # (n,)
    x_top: torch.Tensor  # (n, 2)
    temperature_top: torch.Tensor  # (n,)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> PointSet:
        return PointSet(
            self.x[idx],
            self.temperature[idx],
            self.x_colloc[idx],
            self.x_bottom[idx],
            self.temperature_bottom[idx],
            self.x_top[idx],
            self.temperature_top[idx],
        )


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
        quality = ds["QT_5020"].to_numpy().reshape(n_time, -1)

    pressure[:, 0] = cfg.surface_pressure
    observed = (temperature < FILL_THRESHOLD) & (pressure < FILL_THRESHOLD)
    observed &= np.isin(quality, GOOD_QUALITY)
    drop = list(cfg.drop_levels)
    temperature = fill_gaps(np.delete(temperature, drop, axis=1))
    pressure = fill_gaps(np.delete(pressure, drop, axis=1))
    # The mooring line moves, so sensor depths come from the measured pressure
    # rather than the nominal depths.
    z = gsw.z_from_p(pressure, cfg.latitude)
    return MooringRecord(
        time=time, z=z, temperature=temperature, observed=np.delete(observed, drop, axis=1)
    )


def build_observations(record: MooringRecord, test_fraction: float) -> Observations:
    """Flatten the (time, level) grid level by level and normalise it.

    z and t are min-max scaled to [0, 1]. Temperature is standardised with the
    statistics of the training period, so nothing about the test period reaches the model.
    """
    if not 0 <= test_fraction < 1:
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")
    n_time, n_level = record.temperature.shape
    n_train_times = int(n_time * (1 - test_fraction))
    t = np.tile((record.time - record.time[0]) / np.timedelta64(1, "s"), n_level)
    z = record.z.T.ravel()
    temperature = record.temperature.T.ravel()
    train_temperature = record.temperature[:n_train_times]

    z_scale = Affine(offset=float(z.min()), scale=float(z.max() - z.min()))
    t_scale = Affine(offset=float(t.min()), scale=float(t.max() - t.min()))
    temperature_scale = Affine(
        offset=float(train_temperature.mean(dtype=np.float64)),
        scale=float(train_temperature.std(dtype=np.float64, ddof=1)),
    )
    x = np.stack([z_scale.normalize(z), t_scale.normalize(t)], axis=1)
    return Observations(
        x=torch.as_tensor(x, dtype=torch.float32),
        temperature=torch.as_tensor(temperature_scale.normalize(temperature), dtype=torch.float32),
        z_scale=z_scale,
        t_scale=t_scale,
        temperature_scale=temperature_scale,
        start_time=record.time[0],
        n_time=n_time,
        n_train_times=n_train_times,
    )


def load_observations(cfg: DataConfig) -> Observations:
    """Load, clean and normalise the mooring record."""
    return build_observations(load_record(cfg), cfg.test_fraction)


def sample_collocation_points(
    n: int, seed: int, t_range: tuple[float, float] = (0.0, 1.0)
) -> torch.Tensor:
    """Uniform random (z, t) points with z in [0, 1] and t in `t_range`."""
    generator = torch.Generator().manual_seed(seed)
    z = torch.rand(n, generator=generator)
    t = t_range[0] + (t_range[1] - t_range[0]) * torch.rand(n, generator=generator)
    return torch.stack([z, t], dim=1)


def train_test_split(observations: Observations, seed: int) -> tuple[PointSet, PointSet]:
    """Split the record in time: the training period first, then the test period.

    Every level contributes to both. Each split gets collocation points from its own
    period, and boundary values from the sensors at the times of its observations.
    """
    x, temperature = observations.x, observations.temperature
    n_time = observations.n_time
    time_index = torch.arange(len(x), device=x.device) % n_time
    is_train = time_index < observations.n_train_times
    is_test = ~is_train
    n_train, n_test = int(is_train.sum()), int(is_test.sum())
    # Points run level by level from the surface down, so the first and last
    # n_time points are the shallowest and deepest sensors.
    t_cut = x[observations.n_train_times, 1].item() if n_test else 1.0

    def select(mask: torch.Tensor, n: int, colloc_seed: int, t_range) -> PointSet:
        index = time_index[mask]
        return PointSet(
            x[mask],
            temperature[mask],
            sample_collocation_points(n, colloc_seed, t_range).to(x.device),
            x[-n_time:][index],
            temperature[-n_time:][index],
            x[:n_time][index],
            temperature[:n_time][index],
        )

    return (
        select(is_train, n_train, seed, (0.0, t_cut)),
        select(is_test, n_test, seed + 1, (t_cut, 1.0)),
    )
