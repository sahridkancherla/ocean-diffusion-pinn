"""Report the loss terms of a trained PINN on the train or test split."""

from __future__ import annotations

import argparse
from pathlib import Path

from ocean_pinn.config import DataConfig, LossWeights, TrainConfig
from ocean_pinn.data import load_observations, train_test_split
from ocean_pinn.evaluation import evaluate
from ocean_pinn.model import load_model
from ocean_pinn.utils import resolve_device


def parse_args() -> argparse.Namespace:
    data, cfg = DataConfig(), TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=cfg.checkpoint_path)
    parser.add_argument("--data-dir", type=Path, default=data.data_dir)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--seed", type=int, default=cfg.seed, help="collocation point seed")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, ...")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data_cfg = DataConfig(data_dir=args.data_dir)

    observations = load_observations(data_cfg).to(device)
    train_points, test_points = train_test_split(observations, data_cfg.n_train, args.seed)
    points = train_points if args.split == "train" else test_points
    model = load_model(args.checkpoint, device)
    losses = evaluate(model, points, LossWeights(), args.batch_size)

    print(f"{args.split} split: {len(points):,} points")
    for name, value in losses.items():
        print(f"  {name:<8} {value:.6g}")
    rmse = losses["data"] ** 0.5 * observations.temperature_scale.scale
    print(f"  temperature RMSE {rmse:.3f} °C")


if __name__ == "__main__":
    main()
