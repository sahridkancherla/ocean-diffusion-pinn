"""Check how well the observations themselves obey the heat equations, with no network.

Temperature tendencies and vertical gradients are estimated by finite differences from
measured samples only, then transport coefficients are fitted to the first part of the
record and scored on the rest. The score is the share of the tendency explained
out of sample: 1 is a perfect budget, 0 is no better than predicting no change, and
negative is worse than that.

Four candidates are compared, written with z positive upwards:

- diffusion:          dT/dt = d/dz (D(z) dT/dz)
- steady advection:   dT/dt + w(z) dT/dz = d/dz (D(z) dT/dz)
- moving advection:   as above with w(z) + w'(z, t)
- neighbour average:  dT/dt copied from the sensors nearby, with no physics

A velocity free to take any value at every depth and time could close any budget, so it
would prove nothing. Here w' at each sensor is estimated only from the sensors around it,
which keeps the score out of sample. The neighbour average is the reference for it:
any depth-coherent change favours both, including horizontal advection, which one
mooring cannot observe. Moving advection therefore only earns credit by beating that
reference, and even then it is an upper bound on what vertical motion explains.

D(z) and w(z) are smooth profiles built from a few piecewise-linear functions of depth,
not free values at every sensor. On a fixed sensor grid, dT/dz is an exact combination of
the two diffusive fluxes at a level, so free per-level D and w cannot be told apart and the
fit collapses into a zig-zag. Smoothness and D >= 0 are what separate them.

Noise in the measured gradients biases fitted coefficients towards zero, and a fitted D
absorbs whatever the budget leaves out, so D is an effective value and not a measurement
of turbulent mixing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from .data import MooringRecord

MIN_VALID_FRACTION = 0.5  # of the samples in a window, for a sensor to count in it
NEIGHBOURS = 2  # sensors each side of a level used to estimate its w'
GRADIENT_FLOOR = 1e-3  # °C/m, stops w' blowing up where the neighbours are well mixed
FIT_ROUNDS = 3  # alternations between fitting (D, w) and estimating w'
N_KNOTS = 6  # piecewise-linear functions making up each of the profiles D(z) and w(z)


@dataclass
class BudgetTerms:
    """Finite-difference terms at interior levels, each (n_window, n_interior).

    Entries are NaN where a sensor needed for them has no valid data.
    """

    tendency: np.ndarray  # dT/dt at a fixed depth, °C/s
    gradient: np.ndarray  # dT/dz, °C/m
    flux_above: np.ndarray  # coefficient of D at the interface above, °C/m²
    flux_below: np.ndarray  # coefficient of D at the interface below, °C/m²
    depth: np.ndarray  # (n_interior,) mean sensor depth, m, positive down
    interface_depth: np.ndarray  # (n_interior + 1,) mean interface depth, m, positive down


@dataclass
class Score:
    """Out-of-sample skill of one candidate."""

    name: str
    skill: float  # share of the tendency explained, over all scored samples
    skill_by_level: np.ndarray  # (n_interior,)


@dataclass
class PDECheck:
    """Result of the check at one averaging window."""

    window_days: float
    n_train_samples: int
    n_test_samples: int
    depth: np.ndarray  # (n_interior,) m
    interface_depth: np.ndarray  # (n_interior + 1,) m
    scores: list[Score]
    diffusivity: np.ndarray  # (n_interior + 1,) m²/s, from the diffusion fit
    advective_diffusivity: np.ndarray  # (n_interior + 1,) m²/s, from the moving advection fit
    steady_velocity: np.ndarray  # (n_interior,) m/s, positive up, from the steady fit
    moving_velocity_rms: float  # m/s, of w' over the test period


def average_windows(
    record: MooringRecord, window_days: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Mean temperature and height per window, from measured samples only.

    Returns the temperature and z, each (n_window, n_level) and NaN where a sensor has
    too few valid samples, and the window length in seconds.
    """
    if record.observed is None:
        raise ValueError("The check needs to know which samples were measured.")
    step = float(np.median(np.diff(record.time) / np.timedelta64(1, "s")))
    per_window = round(window_days * 86400 / step)
    n_window = len(record.time) // per_window
    if per_window < 1 or n_window < 3:
        raise ValueError(f"A {window_days}-day window leaves too few windows in this record.")
    n_used = n_window * per_window

    def blocks(values: np.ndarray) -> np.ndarray:
        return values[:n_used].reshape(n_window, per_window, -1)

    valid = blocks(record.observed)
    count = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        temperature = (blocks(record.temperature).astype(np.float64) * valid).sum(axis=1) / count
        z = (blocks(record.z).astype(np.float64) * valid).sum(axis=1) / count
    too_few = count < MIN_VALID_FRACTION * per_window
    temperature[too_few] = np.nan
    z[too_few] = np.nan
    return temperature, z, per_window * step


def budget_terms(temperature: np.ndarray, z: np.ndarray, window_seconds: float) -> BudgetTerms:
    """Finite-difference budget terms on the moving, unevenly spaced sensor grid.

    A sensor at z_s(t) records T(z_s(t), t), so its rate of change is
    dT/dt + (dz_s/dt) dT/dz. The second term is removed to get the fixed-depth tendency.
    """
    with np.errstate(invalid="ignore"):
        spacing_above = z[:, :-2] - z[:, 1:-1]
        spacing_below = z[:, 1:-1] - z[:, 2:]
        spacing_above[spacing_above <= 0] = np.nan
        spacing_below[spacing_below <= 0] = np.nan

    gradient_above = (temperature[:, :-2] - temperature[:, 1:-1]) / spacing_above
    gradient_below = (temperature[:, 1:-1] - temperature[:, 2:]) / spacing_below
    total_spacing = spacing_above + spacing_below
    gradient = (gradient_above * spacing_below + gradient_below * spacing_above) / total_spacing

    def rate(values: np.ndarray) -> np.ndarray:
        result = np.full_like(values, np.nan)
        result[1:-1] = (values[2:] - values[:-2]) / (2 * window_seconds)
        return result

    tendency = rate(temperature)[:, 1:-1] - rate(z)[:, 1:-1] * gradient
    return BudgetTerms(
        tendency=tendency,
        gradient=gradient,
        flux_above=2 * gradient_above / total_spacing,
        flux_below=2 * gradient_below / total_spacing,
        depth=-np.nanmean(z[:, 1:-1], axis=0),
        interface_depth=-np.nanmean((z[:, :-1] + z[:, 1:]) / 2, axis=0),
    )


def _diffusive_tendency(terms: BudgetTerms, diffusivity: np.ndarray) -> np.ndarray:
    """d/dz (D dT/dz) for a diffusivity on the interfaces."""
    return terms.flux_above * diffusivity[:-1] - terms.flux_below * diffusivity[1:]


def _profile_basis(n: int) -> np.ndarray:
    """Piecewise-linear functions of level index, shape (n, n_knots), summing to 1 at each level.

    Non-negative weights give a non-negative profile, and sensors are densest near the
    surface, so spacing the knots by level index resolves the upper ocean best.
    """
    n_knots = min(N_KNOTS, n)
    knots = np.linspace(0, n - 1, n_knots)
    return np.stack(
        [np.interp(np.arange(n), knots, row) for row in np.eye(n_knots)],
        axis=1,
    )


def _fit_budget(
    terms: BudgetTerms,
    rows: np.ndarray,
    target: np.ndarray,
    with_velocity: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares profiles D(z) >= 0 and, optionally, w(z), fitted on `rows`.

    `target` is the tendency to explain, and `rows` an (n_window, n_interior) mask.
    Returns D on the interfaces and w on the interior levels.
    """
    n_interior = terms.tendency.shape[1]
    D_basis = _profile_basis(n_interior + 1)
    w_basis = _profile_basis(n_interior)
    window, level = np.nonzero(rows)
    columns = [
        terms.flux_above[window, level, None] * D_basis[level]
        - terms.flux_below[window, level, None] * D_basis[level + 1]
    ]
    if with_velocity:
        columns.append(-terms.gradient[window, level, None] * w_basis[level])
    design = np.concatenate(columns, axis=1)

    # The columns differ by orders of magnitude, so solve in units where they match.
    scale = np.linalg.norm(design, axis=0)
    scale[scale == 0] = 1.0
    n_D = D_basis.shape[1]
    lower = np.full(design.shape[1], -np.inf)
    lower[:n_D] = 0.0
    solution = lsq_linear(
        design / scale, target[window, level], bounds=(lower, np.inf), method="bvls"
    ).x
    solution = solution / scale
    velocity = w_basis @ solution[n_D:] if with_velocity else np.zeros(n_interior)
    return D_basis @ solution[:n_D], velocity


def _nearby_sum(values: np.ndarray) -> np.ndarray:
    """Sum over the sensors within NEIGHBOURS levels of each level, excluding the level."""
    total = np.zeros_like(values)
    for offset in range(1, NEIGHBOURS + 1):
        total[:, offset:] += values[:, :-offset]
        total[:, :-offset] += values[:, offset:]
    return total


def _nearby_velocity(residual: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """Least-squares w' per sensor from its neighbours, closing their budgets.

    The result is 0 where no neighbour has data.
    """
    valid = np.isfinite(residual) & np.isfinite(gradient)
    moment = np.where(valid, residual * gradient, 0.0)
    weight = np.where(valid, gradient**2, 0.0)
    return -_nearby_sum(moment) / (_nearby_sum(weight) + GRADIENT_FLOOR**2)


def _nearby_mean(tendency: np.ndarray) -> np.ndarray:
    """Mean tendency of the neighbours of each sensor, NaN where none has data."""
    valid = np.isfinite(tendency)
    count = _nearby_sum(valid.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        return _nearby_sum(np.where(valid, tendency, 0.0)) / count


def _score(name: str, tendency: np.ndarray, predicted: np.ndarray, rows: np.ndarray) -> Score:
    """Share of the tendency explained on `rows`, overall and per level.

    Samples a candidate cannot predict count as predicting no change.
    """
    target = np.where(rows, tendency, 0.0)
    error = np.where(rows, tendency - np.nan_to_num(predicted), 0.0)
    total = (target**2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        by_level = 1 - (error**2).sum(axis=0) / total
    return Score(name, float(1 - (error**2).sum() / total.sum()), by_level)


def check_pde(record: MooringRecord, window_days: float, test_fraction: float) -> PDECheck:
    """Fit each candidate on the start of the record and score it on the end."""
    temperature, z, window_seconds = average_windows(record, window_days)
    terms = budget_terms(temperature, z, window_seconds)

    usable = (
        np.isfinite(terms.tendency)
        & np.isfinite(terms.gradient)
        & np.isfinite(terms.flux_above)
        & np.isfinite(terms.flux_below)
    )
    is_train = np.arange(len(usable)) < int(len(usable) * (1 - test_fraction))
    train = usable & is_train[:, None]
    test = usable & ~is_train[:, None]
    if not train.any() or not test.any():
        raise ValueError(f"No usable samples in one split at a {window_days}-day window.")

    def fill(values: np.ndarray) -> np.ndarray:
        return np.nan_to_num(values)

    tendency = terms.tendency
    gradient = terms.gradient

    D_diffusion, _ = _fit_budget(terms, train, fill(tendency), with_velocity=False)
    predicted_diffusion = _diffusive_tendency(terms, D_diffusion)

    D_steady, w_steady = _fit_budget(terms, train, fill(tendency), with_velocity=True)
    predicted_steady = _diffusive_tendency(terms, D_steady) - w_steady * gradient

    moving = np.zeros_like(tendency)
    D_moving, w_mean = D_steady, w_steady
    for _ in range(FIT_ROUNDS):
        # The tendency residual is estimated at every usable sample, in either period:
        # w' at a sensor never uses that sensor's own tendency.
        residual = tendency - _diffusive_tendency(terms, D_moving) + w_mean * gradient
        moving = _nearby_velocity(np.where(usable, residual, np.nan), gradient)
        D_moving, w_mean = _fit_budget(
            terms, train, fill(tendency + moving * gradient), with_velocity=True
        )
    residual = tendency - _diffusive_tendency(terms, D_moving) + w_mean * gradient
    moving = _nearby_velocity(np.where(usable, residual, np.nan), gradient)
    predicted_moving = _diffusive_tendency(terms, D_moving) - (w_mean + moving) * gradient

    predicted_neighbours = _nearby_mean(np.where(usable, tendency, np.nan))

    scores = [
        _score("diffusion", tendency, predicted_diffusion, test),
        _score("steady advection", tendency, predicted_steady, test),
        _score("moving advection", tendency, predicted_moving, test),
        _score("neighbour average", tendency, predicted_neighbours, test),
    ]
    return PDECheck(
        window_days=window_days,
        n_train_samples=int(train.sum()),
        n_test_samples=int(test.sum()),
        depth=terms.depth,
        interface_depth=terms.interface_depth,
        scores=scores,
        diffusivity=D_diffusion,
        advective_diffusivity=D_moving,
        steady_velocity=w_steady,
        moving_velocity_rms=float(np.sqrt(np.mean(moving[test] ** 2))),
    )
