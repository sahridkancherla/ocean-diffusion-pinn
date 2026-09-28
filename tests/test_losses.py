import math

import torch
from torch import nn

from ocean_pinn.config import LossWeights
from ocean_pinn.data import PointSet
from ocean_pinn.losses import boundary_losses, compute_losses, pde_residual


class Analytic(nn.Module):
    """Stand-in for the network that returns closed-form T(z, t) and D(z, t)."""

    def __init__(self, temperature, diffusivity):
        super().__init__()
        self.temperature = temperature
        self.diffusivity = diffusivity

    def forward(self, x):
        z, t = x[:, :1], x[:, 1:]
        return self.temperature(z, t), self.diffusivity(z, t)


def random_points(n=50, seed=0):
    return torch.rand(n, 2, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def test_residual_vanishes_for_decaying_mode():
    # T = exp(-D k^2 t) sin(k z) solves dT/dt = D d2T/dz2 for constant D.
    D, k = 0.3, 2.0
    model = Analytic(
        lambda z, t: torch.exp(-D * k**2 * t) * torch.sin(k * z),
        lambda z, t: torch.full_like(z, D),
    )
    assert pde_residual(model, random_points()).abs().max() < 1e-10


def test_residual_uses_flux_form_for_variable_diffusivity():
    # With D = 1 + z, T = log(1 + z) carries a constant flux D dT/dz = 1, so it is
    # a steady solution. The non-conservative D d2T/dz2 would give -1 / (1 + z).
    model = Analytic(lambda z, t: torch.log1p(z), lambda z, t: 1 + z)
    assert pde_residual(model, random_points()).abs().max() < 1e-10


def test_residual_of_non_solution():
    # T = z^2 with D = 1 gives dT/dt - d2T/dz2 = -2.
    model = Analytic(lambda z, t: z**2, lambda z, t: torch.ones_like(z))
    residual = pde_residual(model, random_points())
    torch.testing.assert_close(residual, torch.full((50, 1), -2.0, dtype=torch.float64))


def test_boundary_losses():
    # T = z + t: T(0, t) = t and dT/dz = 1 everywhere.
    model = Analytic(lambda z, t: z + t, lambda z, t: torch.ones_like(z))
    t = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    dirichlet, neumann = boundary_losses(model, t)
    assert math.isclose(dirichlet.item(), (0 + 0.25 + 1) / 3)
    assert math.isclose(neumann.item(), 1.0)


def test_data_loss_compares_each_prediction_with_its_own_target():
    # A model that reproduces the targets exactly must have zero data loss;
    # broadcasting (n, 1) predictions against (n,) targets would not.
    model = Analytic(lambda z, t: torch.sin(z + t), lambda z, t: torch.ones_like(z))
    x = random_points()
    points = PointSet(x, torch.sin(x[:, 0] + x[:, 1]), random_points(seed=1))
    assert compute_losses(model, points, LossWeights())["data"].item() == 0.0


def test_total_is_weighted_sum_of_terms():
    model = Analytic(lambda z, t: torch.sin(z + t), lambda z, t: 1 + z)
    x = random_points()
    points = PointSet(x, torch.zeros(len(x), dtype=torch.float64), random_points(seed=1))
    weights = LossWeights(data=2.0, physics=3.0, bc_z0=5.0, bc_z1=7.0)
    losses = compute_losses(model, points, weights)
    expected = (
        2 * losses["data"] + 3 * losses["physics"] + 5 * losses["bc_z0"] + 7 * losses["bc_z1"]
    )
    torch.testing.assert_close(losses["total"], expected)
