"""Plot observed and PINN-predicted temperature over the whole mooring record."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ocean_pinn.config import DataConfig, TrainConfig
from ocean_pinn.data import load_observations
from ocean_pinn.evaluation import predict
from ocean_pinn.model import load_model
from ocean_pinn.plotting import plot_temperature_comparison
from ocean_pinn.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=TrainConfig().checkpoint_path)
    parser.add_argument("--data-dir", type=Path, default=DataConfig().data_dir)
    parser.add_argument("--output", type=Path, default=Path("figures/temperature_comparison.png"))
    parser.add_argument("--show", action="store_true", help="also open the figure in a window")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, ...")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    observations = load_observations(DataConfig(data_dir=args.data_dir))
    model = load_model(args.checkpoint, device)
    predicted, _ = predict(model, observations.x.to(device))

    scale = observations.temperature_scale
    fig = plot_temperature_comparison(
        time=observations.times(),
        depth=observations.depths(),
        observed=scale.denormalize(observations.temperature.numpy()),
        predicted=scale.denormalize(predicted.cpu().numpy()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Saved figure to {args.output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
