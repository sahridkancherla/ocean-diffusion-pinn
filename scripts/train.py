"""Train a PINN on the training period of the KEO mooring record."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from ocean_pinn.config import DataConfig, LossWeights, TrainConfig
from ocean_pinn.data import load_observations, train_test_split
from ocean_pinn.model import AdvectionDiffusionPINN, HeatDiffusionPINN
from ocean_pinn.training import train
from ocean_pinn.utils import resolve_device


def parse_args() -> argparse.Namespace:
    data, cfg, weights = DataConfig(), TrainConfig(), LossWeights()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data.data_dir)
    parser.add_argument("--checkpoint", type=Path, default=cfg.checkpoint_path)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--lr", type=float, default=cfg.learning_rate)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, ...")
    parser.add_argument("--w-data", type=float, default=weights.data, help="data loss weight")
    parser.add_argument("--w-physics", type=float, default=weights.physics, help="PDE loss weight")
    parser.add_argument("--w-bc0", type=float, default=weights.bc_z0, help="deepest sensor")
    parser.add_argument("--w-bc1", type=float, default=weights.bc_z1, help="shallowest sensor")
    parser.add_argument(
        "--advection",
        action="store_true",
        help="fit the advection-diffusion model, with D(z) and a penalised w(z, t)",
    )
    parser.add_argument("--w-velocity", type=float, default=weights.velocity_size, help="size of w")
    parser.add_argument(
        "--w-velocity-rough", type=float, default=weights.velocity_roughness, help="size of dw/dz"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    device = resolve_device(args.device)
    data_cfg = DataConfig(data_dir=args.data_dir)
    cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        loss_weights=LossWeights(
            data=args.w_data,
            physics=args.w_physics,
            bc_z0=args.w_bc0,
            bc_z1=args.w_bc1,
            velocity_size=args.w_velocity,
            velocity_roughness=args.w_velocity_rough,
        ),
        checkpoint_path=args.checkpoint,
    )

    observations = load_observations(data_cfg).to(device)
    train_points, _ = train_test_split(observations, cfg.seed)

    torch.manual_seed(cfg.seed)
    model_class = AdvectionDiffusionPINN if args.advection else HeatDiffusionPINN
    model = model_class().to(device)
    logging.info("Training %s on %d points (%s)", model.physics, len(train_points), device)
    train(model, train_points, cfg)
    logging.info("Saved model to %s", cfg.checkpoint_path)


if __name__ == "__main__":
    main()
