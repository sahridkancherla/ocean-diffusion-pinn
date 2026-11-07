"""Check how well the mooring data obey the diffusion and advection-diffusion PDEs.

Uses only measured samples and no network. See ocean_pinn/pde_check.py for the method.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from ocean_pinn.config import DataConfig
from ocean_pinn.data import load_record
from ocean_pinn.pde_check import PDECheck, check_pde


def parse_args() -> argparse.Namespace:
    data = DataConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data.data_dir)
    parser.add_argument(
        "--window-days",
        type=float,
        nargs="+",
        default=[1, 7, 30],
        help="averaging windows: shorter ones keep more variability, longer ones less noise",
    )
    parser.add_argument("--test-fraction", type=float, default=data.test_fraction)
    return parser.parse_args()


def report(check: PDECheck) -> None:
    print(f"\n=== {check.window_days:g}-day averages ===")
    print(f"fitted on {check.n_train_samples:,} samples, scored on {check.n_test_samples:,}")

    print("\nShare of the temperature tendency explained out of sample")
    print("(1 = perfect, 0 = no better than predicting no change)")
    for score in check.scores:
        print(f"  {score.name:<18} {score.skill:7.3f}")

    names = "".join(f"{score.name.split()[0]:>11}" for score in check.scores)
    print(f"\n  depth (m){names}")
    for level, depth in enumerate(check.depth):
        values = "".join(f"{score.skill_by_level[level]:11.3f}" for score in check.scores)
        print(f"  {depth:9.1f}{values}")

    print("\nFitted coefficients")
    print("  depth (m)  D diffusion  D moving adv  (m²/s, on interfaces between sensors)")
    for depth, diffusion, advection in zip(
        check.interface_depth, check.diffusivity, check.advective_diffusivity, strict=True
    ):
        print(f"  {depth:9.1f}  {diffusion:11.2e}  {advection:12.2e}")
    n_zero = int((check.diffusivity == 0).sum())
    print(f"  {n_zero} of {len(check.diffusivity)} diffusion D values hit the D >= 0 limit")
    print(f"  rms of w' over the test period: {check.moving_velocity_rms:.2e} m/s")


def main() -> None:
    args = parse_args()
    # The levels dropped for training only had to be dropped because filling their
    # leading gaps would invent data. This check uses measured samples only, so keep them.
    cfg = dataclasses.replace(DataConfig(data_dir=args.data_dir), drop_levels=())
    record = load_record(cfg)
    for window_days in args.window_days:
        report(check_pde(record, window_days, args.test_fraction))


if __name__ == "__main__":
    main()
