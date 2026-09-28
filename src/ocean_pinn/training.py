"""Mini-batch training loop and checkpointing."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from .config import TrainConfig
from .data import PointSet
from .losses import compute_losses
from .utils import minibatches

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    batch: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "batch": batch,
        },
        path,
    )


def format_losses(losses: dict[str, torch.Tensor]) -> str:
    return "  ".join(f"{name} {value.item():.4g}" for name, value in losses.items())


def train(model: nn.Module, points: PointSet, cfg: TrainConfig) -> list[float]:
    """Train `model` with Adam and return the mean total loss of each epoch.

    A checkpoint is written every `cfg.checkpoint_every` batches and at the end
    of each epoch.
    """
    # Faster, but less precise, float32 matrix multiplication on recent GPUs.
    torch.set_float32_matmul_precision("high")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    generator = torch.Generator().manual_seed(cfg.seed)
    history = []

    model.train()
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        batches = minibatches(len(points), cfg.batch_size, points.x.device, generator)
        for batch, idx in enumerate(batches):
            optimizer.zero_grad()
            losses = compute_losses(model, points[idx], cfg.loss_weights)
            losses["total"].backward()
            optimizer.step()

            epoch_loss += losses["total"].item()
            if batch % cfg.log_every == 0:
                logger.info("epoch %d  batch %d  %s", epoch, batch, format_losses(losses))
            if batch % cfg.checkpoint_every == 0:
                save_checkpoint(cfg.checkpoint_path, model, optimizer, epoch, batch)

        history.append(epoch_loss / (batch + 1))
        logger.info("epoch %d  mean loss %.6g", epoch, history[-1])
        save_checkpoint(cfg.checkpoint_path, model, optimizer, epoch, batch)
    return history
