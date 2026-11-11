"""Networks mapping normalised depth and time to temperature and the transport fields."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
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

class SpectralConv2D(nn.Module):
    """Want to try purely data-driven FNO, because physics is clearly more compelex
    than simply advection-diffusion.

    Input is (z, t, T_0), i.e. initial temp is included in input. Thus, in_channels=3
    """
    def __init__(self, in_channels, out_channels, k_z, k_t):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_z = k_z
        self.k_t = k_t

        scale = 1 / (in_channels * out_channels)
        shape = (in_channels, out_channels, k_z, k_t)
        self.w_pos = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w_neg = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    def forward(self, x):   # (b, c, z, t)
        z, t = x.shape[-2:]
        if 2 * self.k_z > z or self.k_t > t // 2 + 1:
            raise ValueError(f"Modes ({self.k_z}, {self.k_t}) do not fit a ({z}, {t}) grid.")
        xf = torch.fft.rfft2(x) # (b, c, z, t//2+1)
        out = torch.zeros(x.shape[0], self.out_channels, z, t//2+1, dtype=torch.cfloat,
                          device=x.device)
        mul = lambda a, w: torch.einsum('bczt,cozt->bozt', a, w)
        out[..., :self.k_z, :self.k_t] = mul(xf[..., :self.k_z, :self.k_t], self.w_pos)
        out[..., -self.k_z:, :self.k_t] = mul(xf[..., -self.k_z:, :self.k_t], self.w_neg)

        return torch.fft.irfft2(out, s=(z, t))

class FNO2D(nn.Module):
    def __init__(self, k_z, k_t, width=16, in_channels=3, out_channels=1, num_layers=4):
        super().__init__()
        self.k_z = k_z
        self.k_t = k_t
        self.width = width
        self.num_layers = num_layers

        self.lift = nn.Conv2d(in_channels, self.width, 1)

        self.fourier_layers = nn.ModuleList([
            SpectralConv2D(self.width, self.width, self.k_z, self.k_t) for _ in range(self.num_layers)
        ])
        self.w_layers = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1) for _ in range(self.num_layers)
        ])

        self.project = nn.Sequential(
            nn.Conv2d(self.width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 1)
        )

    def forward(self, x):
        x = self.lift(x)
        
        for n in range(self.num_layers):
            x = F.gelu(self.fourier_layers[n](x) + self.w_layers[n](x))

        x = self.project(x)
        return x

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
