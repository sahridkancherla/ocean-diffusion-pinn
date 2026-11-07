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


def write_record(data_dir, temperature, pressure, quality=None):
    """Write a record laid out like the KEO download: a zip holding one NetCDF-3 file."""
    n_time, n_level = temperature.shape
    if quality is None:
        quality = np.where(temperature > 1e10, 0.0, 2.0).astype(np.float32)
    dims = ("time", "depth", "lat", "lon")
    ds = xr.Dataset(
        {
            "T_20": (dims, temperature.reshape(n_time, n_level, 1, 1)),
            "P_1": (dims, pressure.reshape(n_time, n_level, 1, 1)),
            "QT_5020": (dims, quality.reshape(n_time, n_level, 1, 1)),
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


def test_load_record_marks_only_measured_good_samples_as_observed(tmp_path):
    temperature = np.array([[20.0, 19.0], [20.5, MISSING], [21.0, 19.5]], dtype=np.float32)
    pressure = np.array([[MISSING, 5.0], [MISSING, MISSING], [MISSING, MISSING]], dtype=np.float32)
    quality = np.array([[2, 2], [4, 0], [2, 2]], dtype=np.float32)
    write_record(tmp_path, temperature, pressure, quality)

    record = load_record(DataConfig(data_dir=tmp_path, drop_levels=()))

    # Bad quality, missing temperature, and missing pressure at a sensor that has one.
    np.testing.assert_array_equal(record.observed, [[True, True], [False, False], [True, False]])


def test_build_observations_flattens_level_by_level():
    time = pd.date_range("2020-01-01", periods=3, freq="10min").to_numpy()
    z = np.array([[-1.0, -10.0], [-1.5, -11.0], [-2.0, -12.0]])
    temperature = np.array([[20.0, 10.0], [21.0, 11.0], [22.0, 12.0]], dtype=np.float32)

    obs = build_observations(MooringRecord(time, z, temperature), test_fraction=0.0)

    assert obs.x.shape == (6, 2)
    assert obs.x.min() == 0 and obs.x.max() == 1
    observed = obs.temperature_scale.denormalize(obs.temperature.double().numpy())
    np.testing.assert_allclose(observed, [20, 21, 22, 10, 11, 12], rtol=1e-6)
    np.testing.assert_allclose(obs.depths(), [1, 1.5, 2, 10, 11, 12], rtol=1e-6)
    np.testing.assert_array_equal(obs.times(), np.tile(time, 2))


def small_observations(n_time=10, test_fraction=0.2):
    time = pd.date_range("2020-01-01", periods=n_time, freq="10min").to_numpy()
    z = -np.tile([1.0, 5.0, 9.0], (n_time, 1))
    temperature = np.arange(3 * n_time, dtype=np.float32).reshape(n_time, 3)
    return build_observations(MooringRecord(time, z, temperature), test_fraction)


def test_train_test_split_divides_every_level_in_time():
    obs = small_observations()

    train, test = train_test_split(obs, seed=0)

    assert (len(train), len(test)) == (24, 6)
    assert train.x[:, 1].max() < test.x[:, 1].min()
    # Each period contains all three depths.
    assert len(train.x[:, 0].unique()) == len(test.x[:, 0].unique()) == 3


def test_train_test_split_is_deterministic():
    obs = small_observations()
    train, _ = train_test_split(obs, seed=0)
    train_again, _ = train_test_split(obs, seed=0)
    torch.testing.assert_close(train.x_colloc, train_again.x_colloc)


def test_collocation_points_stay_in_their_own_period():
    train, test = train_test_split(small_observations(), seed=0)
    cut = test.x[:, 1].min()
    assert train.x_colloc.shape == train.x.shape
    assert train.x_colloc[:, 1].max() <= cut <= test.x_colloc[:, 1].min()


def test_boundaries_are_the_extreme_sensors_at_the_same_times():
    obs = small_observations()
    train, _ = train_test_split(obs, seed=0)
    assert (train.x_bottom[:, 0] == 0).all()
    assert (train.x_top[:, 0] == 1).all()
    torch.testing.assert_close(train.x_top[:, 1], train.x[:, 1])
    torch.testing.assert_close(train.x_bottom[:, 1], train.x[:, 1])
    # Temperature steps by one per time and by n_time per level in the fixture.
    steps = obs.temperature_scale.scale * (train.temperature_top - train.temperature_bottom)
    torch.testing.assert_close(steps, torch.full_like(steps, -2.0), rtol=1e-5, atol=1e-5)


def test_temperature_is_normalised_with_training_statistics_only():
    obs = small_observations()
    n_train = obs.n_train_times
    assert n_train == 8
    expected = np.arange(30, dtype=np.float32).reshape(10, 3)[:n_train].mean()
    assert obs.temperature_scale.offset == pytest.approx(float(expected))


def test_build_observations_rejects_invalid_test_fraction():
    with pytest.raises(ValueError):
        small_observations(test_fraction=1.0)


def test_collocation_points_lie_in_requested_range():
    points = sample_collocation_points(1000, seed=0, t_range=(0.25, 0.5))
    assert points.shape == (1000, 2)
    assert points[:, 0].min() >= 0 and points[:, 0].max() <= 1
    assert points[:, 1].min() >= 0.25 and points[:, 1].max() <= 0.5
