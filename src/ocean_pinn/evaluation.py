"""Evaluation of the loss terms and inference."""

from __future__ import annotations

import torch
from torch import nn

from .config import LossWeights
from .data import PointSet
from .losses import compute_losses
from .utils import minibatches


def evaluate(
    model: nn.Module, points: PointSet, weights: LossWeights, batch_size: int = 4096
) -> dict[str, float]:
    """Loss terms averaged over every point of a split."""
    model.eval()
    totals: dict[str, float] = {}
    for idx in minibatches(len(points), batch_size, points.x.device):
        losses = compute_losses(model, points[idx], weights)
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + value.item() * len(idx)
    return {name: total / len(points) for name, total in totals.items()}


@torch.no_grad()
def predict(
    model: nn.Module, x: torch.Tensor, batch_size: int = 65536
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised temperature and diffusivity at points x, each of shape (n,)."""
    model.eval()
    outputs = [model(x[idx]) for idx in minibatches(len(x), batch_size, x.device)]
    temperature = torch.cat([T for T, _ in outputs]).squeeze(1)
    diffusivity = torch.cat([D for _, D in outputs]).squeeze(1)
    return temperature, diffusivity
