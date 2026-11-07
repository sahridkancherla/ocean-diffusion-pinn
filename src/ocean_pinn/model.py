"""Networks mapping normalised depth and time to temperature and the transport fields."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn.functional import softplus


class HeatDiffusionPINN(nn.Module):
    """MLP with a shared encoder and separate temperature and diffusivity heads.

    Takes normalised (z, t) of shape (n, 2) and returns the normalised
    temperature T and the positive diffusivity D(z, t), each of shape (n, 1).
    """

    physics = "diffusion"

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
        return self.temp_head(encoded), softplus(self.D_head(encoded))


class AdvectionDiffusionPINN(HeatDiffusionPINN):
    """Adds a vertical velocity w(z, t) and restricts the diffusivity to D(z).

    If D and w could both vary freely in z and t, almost any D could be balanced by
    some w. Holding D constant in time means fast changes must be explained by w.
    """

    physics = "advection-diffusion"

    def __init__(self, hidden: int = 64) -> None:
        super().__init__(hidden)
        self.D_head = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.w_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.temp_head(self.encoder(x)), softplus(self.D_head(x[:, :1]))

    def velocity(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised vertical velocity at x, positive upwards, of shape (n, 1)."""
        return self.w_head(self.encoder(x))


_MODELS = {cls.physics: cls for cls in (HeatDiffusionPINN, AdvectionDiffusionPINN)}


def load_model(path: str | Path, device: torch.device | str = "cpu") -> HeatDiffusionPINN:
    """Load a model from a checkpoint written during training."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "physics" not in checkpoint:
        raise ValueError(
            f"{path} was written by an older version of this code, whose diffusivity head "
            "had no positivity constraint. Retrain with scripts/train.py."
        )
    model = _MODELS[checkpoint["physics"]]().to(device)
    model.load_state_dict(checkpoint["model"])
    return model
