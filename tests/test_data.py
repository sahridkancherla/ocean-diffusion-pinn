import zipfile

import gsw
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from ocean_pinn.config import DataConfig
from ocean_pinn.data import (
    MooringRecord,
    build_observations,
    fill_gaps,
    load_record,
    sample_collocation_points,
    train_test_split,
)

MISSING = 1e35


def write_record(data_dir, temperature, pressure):
    """Write a record laid out like the KEO download: a zip holding one NetCDF-3 file."""
    n_time, n_level = temperature.shape
    dims = ("time", "depth", "lat", "lon")
    ds = xr.Dataset(
        {
            "T_20": (dims, temperature.reshape(n_time, n_level, 1, 1)),
            "P_1": (dims, pressure.reshape(n_time, n_level, 1, 1)),
        },
        coords={
            "time": pd.date_range("2015-01-01", periods=n_time, freq="10min"),
            "depth": np.arange(n_level, dtype=np.float32),
            "lat": [32.3],
            "lon": [144.6],
        },
    )
    raw = data_dir / "raw.cdf"
    ds.to_netcdf(raw, format="NETCDF3_CLASSIC")
    with zipfile.ZipFile(data_dir / "temp.cdf.zip", "w") as archive:
        archive.write(raw, arcname="temp.cdf.gz")
    raw.unlink()


def test_fill_gaps_carries_last_valid_sample_forward():
    values = np.array([[1, 2], [MISSING, 3], [MISSING, MISSING], [4, 5]], dtype=np.float32)
    expected = np.array([[1, 2], [1, 3], [1, 3], [4, 5]], dtype=np.float32)
    np.testing.assert_array_equal(fill_gaps(values), expected)


def test_fill_gaps_backfills_leading_gap():
    values = np.array([MISSING, MISSING, 2.0, MISSING])
    np.testing.assert_array_equal(fill_gaps(values), [2.0, 2.0, 2.0, 2.0])


def test_fill_gaps_rejects_record_without_valid_samples():
    with pytest.raises(ValueError):
        fill_gaps(np.full((3, 2), MISSING))


def test_load_record_extracts_archive_and_cleans_levels(tmp_path):
    temperature = np.array(
        [[20.0, 19.0, 18.0, 17.0], [20.5, MISSING, 18.5, 17.5], [21.0, 19.5, 19.0, 18.0]],
        dtype=np.float32,
    )
    pressure = np.tile(np.array([MISSING, 5.0, 10.0, 20.0], dtype=np.float32), (3, 1))
    write_record(tmp_path, temperature, pressure)
    cfg = DataConfig(data_dir=tmp_path, drop_levels=(2,))

    record = load_record(cfg)

    assert (tmp_path / "temp.cdf").exists()
    np.testing.assert_array_equal(
        record.temperature, [[20.0, 19.0, 17.0], [20.5, 19.0, 17.5], [21.0, 19.5, 18.0]]
    )
    # The missing surface pressure is replaced, and z is height (negative downwards).
    np.testing.assert_allclose(record.z[:, 0], gsw.z_from_p(cfg.surface_pressure, cfg.latitude))
    assert (np.diff(record.z, axis=1) < 0).all()


def test_build_observations_flattens_level_by_level():
    time = pd.date_range("2020-01-01", periods=3, freq="10min").to_numpy()
    z = np.array([[-1.0, -10.0], [-1.5, -11.0], [-2.0, -12.0]])
    temperature = np.array([[20.0, 10.0], [21.0, 11.0], [22.0, 12.0]], dtype=np.float32)

    obs = build_observations(MooringRecord(time, z, temperature))

    assert obs.x.shape == (6, 2)
    assert obs.x.min() == 0 and obs.x.max() == 1
    observed = obs.temperature_scale.denormalize(obs.temperature.double().numpy())
    np.testing.assert_allclose(observed, [20, 21, 22, 10, 11, 12], rtol=1e-6)
    np.testing.assert_allclose(obs.depths(), [1, 1.5, 2, 10, 11, 12], rtol=1e-6)
    np.testing.assert_array_equal(obs.times(), np.tile(time, 2))


def test_train_test_split_is_deterministic():
    time = pd.date_range("2020-01-01", periods=5, freq="10min").to_numpy()
    z = -np.tile([1.0, 5.0], (5, 1))
    temperature = np.arange(10, dtype=np.float32).reshape(5, 2)
    obs = build_observations(MooringRecord(time, z, temperature))

    train, test = train_test_split(obs, n_train=7, seed=0)
    train_again, _ = train_test_split(obs, n_train=7, seed=0)

    assert (len(train), len(test)) == (7, 3)
    assert train.x_colloc.shape == (7, 2)
    torch.testing.assert_close(train.x, obs.x[:7])
    torch.testing.assert_close(train.x_colloc, train_again.x_colloc)


def test_collocation_points_lie_in_unit_square():
    points = sample_collocation_points(1000, seed=0)
    assert points.shape == (1000, 2)
    assert points.min() >= 0 and points.max() <= 1
