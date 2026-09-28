"""Loss terms: data misfit, boundary conditions and the heat diffusion PDE residual.

Everything is in normalised units: z and t are scaled to [0, 1] and the
temperature is standardised.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import LossWeights
from .data import PointSet


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Per-sample gradient of `outputs` with respect to `inputs`, kept differentiable."""
    return torch.autograd.grad(outputs, inputs, torch.ones_like(outputs), create_graph=True)[0]


def pde_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Residual of dT/dt = d/dz (D dT/dz) at points x = (z, t), shape (n, 1)."""
    x = x.detach().requires_grad_(True)
    T, D = model(x)
    dT_dz, dT_dt = _grad(T, x).split(1, dim=1)
    d_flux_dz = _grad(D * dT_dz, x)[:, :1]
    return dT_dt - d_flux_dz


def boundary_losses(model: nn.Module, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean squared boundary-condition misfits at times t.

    Returns the misfits of the Dirichlet condition T = 0 at z = 0 and the Neumann
    condition dT/dz = 0 at z = 1.
    """
    t = t.detach()
    x0 = torch.stack([torch.zeros_like(t), t], dim=1)
    x1 = torch.stack([torch.ones_like(t), t], dim=1).requires_grad_(True)
    T0, _ = model(x0)
    T1, _ = model(x1)
    dT1_dz = _grad(T1, x1)[:, 0]
    return T0.pow(2).mean(), dT1_dz.pow(2).mean()


def compute_losses(
    model: nn.Module, points: PointSet, weights: LossWeights
) -> dict[str, torch.Tensor]:
    """Loss terms for a batch of points and their weighted total.

    The boundary conditions are applied at the times of the observations.
    """
    T_pred, _ = model(points.x)
    losses = {
        "data": (T_pred.squeeze(1) - points.temperature).pow(2).mean(),
        "physics": pde_residual(model, points.x_colloc).pow(2).mean(),
    }
    losses["bc_z0"], losses["bc_z1"] = boundary_losses(model, points.x[:, 1])
    losses["total"] = (
        weights.data * losses["data"]
        + weights.physics * losses["physics"]
        + weights.bc_z0 * losses["bc_z0"]
        + weights.bc_z1 * losses["bc_z1"]
    )
    return losses
