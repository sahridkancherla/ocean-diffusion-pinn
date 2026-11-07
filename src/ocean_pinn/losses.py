"""Loss terms: data misfit, boundary conditions and the PDE residual.

Everything is in normalised units: z and t are scaled to [0, 1] and the
temperature is standardised.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from .config import LossWeights
from .data import PointSet


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Per-sample gradient of `outputs` with respect to `inputs`, kept differentiable."""
    return torch.autograd.grad(outputs, inputs, torch.ones_like(outputs), create_graph=True)[0]


def pde_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Residual of the model's PDE at points x = (z, t), shape (n, 1).

    The PDE is dT/dt = d/dz (D dT/dz), with an added w dT/dz on the left if the
    model has a vertical velocity.
    """
    x = x.detach().requires_grad_(True)
    T, D = model(x)
    dT_dz, dT_dt = _grad(T, x).split(1, dim=1)
    residual = dT_dt - _grad(D * dT_dz, x)[:, :1]
    velocity = getattr(model, "velocity", None)
    if velocity is not None:
        residual = residual + velocity(x) * dT_dz
    return residual


def velocity_penalties(
    velocity: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean squared size of w and of dw/dz at points x.

    The depth term keeps w coherent over neighbouring depths, as eddy motion is.
    """
    x = x.detach().requires_grad_(True)
    w = velocity(x)
    return w.pow(2).mean(), _grad(w, x)[:, 0].pow(2).mean()


def boundary_losses(model: nn.Module, points: PointSet) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean squared misfits to the deepest (z = 0 end) and shallowest (z = 1 end) sensors."""
    T_bottom, _ = model(points.x_bottom)
    T_top, _ = model(points.x_top)
    return (
        (T_bottom.squeeze(1) - points.temperature_bottom).pow(2).mean(),
        (T_top.squeeze(1) - points.temperature_top).pow(2).mean(),
    )


def compute_losses(
    model: nn.Module, points: PointSet, weights: LossWeights
) -> dict[str, torch.Tensor]:
    """Loss terms for a batch of points and their weighted total."""
    T_pred, _ = model(points.x)
    losses = {
        "data": (T_pred.squeeze(1) - points.temperature).pow(2).mean(),
        "physics": pde_residual(model, points.x_colloc).pow(2).mean(),
    }
    losses["bc_z0"], losses["bc_z1"] = boundary_losses(model, points)
    total = (
        weights.data * losses["data"]
        + weights.physics * losses["physics"]
        + weights.bc_z0 * losses["bc_z0"]
        + weights.bc_z1 * losses["bc_z1"]
    )
    velocity = getattr(model, "velocity", None)
    if velocity is not None:
        losses["w_size"], losses["w_rough"] = velocity_penalties(velocity, points.x_colloc)
        total = (
            total
            + weights.velocity_size * losses["w_size"]
            + weights.velocity_roughness * losses["w_rough"]
        )
    losses["total"] = total
    return losses
