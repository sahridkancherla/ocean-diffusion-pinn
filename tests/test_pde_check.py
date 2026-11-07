import numpy as np
import pandas as pd
import pytest

from ocean_pinn.data import MooringRecord
from ocean_pinn.pde_check import average_windows, budget_terms, check_pde

DEPTHS = np.array([10.0, 30.0, 55.0, 75.0, 100.0, 130.0, 150.0, 180.0, 200.0, 230.0, 250.0, 280.0])
N_TIME = 144 * 240  # 240 days of 10-minute samples
DAY = 86400.0
WAVENUMBER = 2 * np.pi / 500  # 1/m
SKILLS_TOLERANCE = 0.02


def seconds() -> np.ndarray:
    return np.arange(N_TIME) * 600.0


def make_record(temperature_at, sensor_motion=0.0) -> MooringRecord:
    """Sample T(z, t) at sensors whose heights move by up to `sensor_motion` metres."""
    t = seconds()
    swing = sensor_motion * np.sin(2 * np.pi * t / (13 * DAY))[:, None]
    z = -DEPTHS[None, :] + swing
    return MooringRecord(
        time=pd.date_range("2015-01-01", periods=N_TIME, freq="10min").to_numpy(),
        z=z,
        temperature=temperature_at(z, t[:, None]),
        observed=np.ones_like(z, dtype=bool),
    )


def decaying_wave(diffusivity, velocity=0.0):
    """T = exp(-D k² t) sin(k (z - w t)) solves dT/dt + w dT/dz = D d²T/dz²."""

    def temperature_at(z, t):
        decay = np.exp(-diffusivity * WAVENUMBER**2 * t)
        return 15 + 5 * decay * np.sin(WAVENUMBER * (z - velocity * t))

    return temperature_at


def skills(check):
    return {score.name: score.skill for score in check.scores}


def test_diffusive_record_obeys_diffusion_and_recovers_diffusivity():
    check = check_pde(make_record(decaying_wave(5e-4)), window_days=1, test_fraction=0.2)

    assert skills(check)["diffusion"] > 1 - SKILLS_TOLERANCE
    np.testing.assert_allclose(check.diffusivity[1:-1], 5e-4, rtol=0.05)


def test_steady_advection_is_detected_and_recovered():
    velocity = 2e-5  # m/s, about 1.7 m/day
    record = make_record(decaying_wave(5e-4, velocity))

    check = check_pde(record, window_days=1, test_fraction=0.2)
    result = skills(check)

    assert result["steady advection"] > 1 - SKILLS_TOLERANCE
    assert result["diffusion"] < 0.5
    np.testing.assert_allclose(check.steady_velocity[1:-1], velocity, rtol=0.1)
    np.testing.assert_allclose(check.advective_diffusivity[1:-1], 5e-4, rtol=0.1)


def test_moving_advection_explains_coherent_heave_beyond_steady_velocity():
    # Layers heaving up and down at 20-day period, with no diffusion. Temperature is
    # stratified throughout, so every sensor sees the same motion through dT/dz.
    def temperature_at(z, t):
        heave = 40 * np.sin(2 * np.pi * t / (20 * DAY))
        return 20 + 0.03 * (z - heave) + 2 * np.sin(WAVENUMBER * (z - heave))

    result = skills(check_pde(make_record(temperature_at), window_days=1, test_fraction=0.2))

    assert result["moving advection"] > 0.9
    assert result["steady advection"] < 0.2
    assert result["diffusion"] < 0.2


def test_incoherent_noise_obeys_no_candidate():
    rng = np.random.default_rng(0)
    # Independent random walks per sensor around a stratified profile.
    noise = np.cumsum(rng.normal(0, 0.05, size=(N_TIME, len(DEPTHS))), axis=0)

    def temperature_at(z, t):
        return 20 + 0.03 * z + noise

    result = skills(check_pde(make_record(temperature_at), window_days=7, test_fraction=0.2))

    assert max(result.values()) < 0.3


def test_sensor_motion_is_not_mistaken_for_temperature_change():
    # Temperature is steady, so everything a moving sensor sees comes from its own motion.
    record = make_record(lambda z, t: 20 + 0.03 * z + 0 * t, sensor_motion=10.0)
    temperature, z, window_seconds = average_windows(record, window_days=1)

    terms = budget_terms(temperature, z, window_seconds)

    raw_rate = np.diff(temperature, axis=0) / window_seconds
    assert np.abs(raw_rate).max() > 1e-7
    assert np.nanmax(np.abs(terms.tendency)) < 1e-10


def test_unmeasured_samples_have_no_effect():
    def with_gap(fill_value):
        record = make_record(decaying_wave(5e-4))
        gap = slice(N_TIME // 4, N_TIME // 2)
        record.temperature[gap, 5] = fill_value
        record.z[gap, 5] = -fill_value
        record.observed[gap, 5] = False
        return check_pde(record, window_days=1, test_fraction=0.2)

    # Stale forward-filled values or outright garbage must give the identical answer.
    stale, garbage = with_gap(15.0), with_gap(1e6)

    np.testing.assert_array_equal(stale.diffusivity, garbage.diffusivity)
    assert skills(stale) == skills(garbage)
    assert skills(stale)["diffusion"] > 1 - SKILLS_TOLERANCE
    np.testing.assert_allclose(stale.diffusivity[1:-1], 5e-4, rtol=0.05)


def test_coefficient_profiles_stay_smooth_when_diffusivity_and_velocity_compete():
    # Free per-level values collapse here into a zig-zag D and wild w, despite noise-free data.
    check = check_pde(make_record(decaying_wave(5e-4, 2e-5)), window_days=1, test_fraction=0.2)
    assert (check.diffusivity >= 0).all()
    assert np.abs(np.diff(check.steady_velocity)).max() < 2e-5


def test_window_too_long_for_record_is_rejected():
    with pytest.raises(ValueError):
        check_pde(make_record(decaying_wave(5e-4)), window_days=200, test_fraction=0.2)
