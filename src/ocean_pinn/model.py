"""Network mapping normalised depth and time to temperature and diffusivity."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class HeatDiffusionPINN(nn.Module):
    """MLP with a shared encoder and separate temperature and diffusivity heads.

    Takes normalised (z, t) of shape (n, 2) and returns the normalised
    temperature T and diffusivity D, each of shape (n, 1).
    """

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.temp_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.D_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return self.temp_head(encoded), self.D_head(encoded)


def load_model(path: str | Path, device: torch.device | str = "cpu") -> HeatDiffusionPINN:
    """Load a model from a checkpoint written during training."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = HeatDiffusionPINN().to(device)
    model.load_state_dict(checkpoint["model"])
    return model
