from pathlib import Path

import pytest
import torch

from ocean_pinn.model import HeatDiffusionPINN, load_model

CHECKPOINT = Path(__file__).parents[1] / "checkpoints" / "checkpoint.pt"


def test_outputs_temperature_and_diffusivity():
    T, D = HeatDiffusionPINN()(torch.rand(8, 2))
    assert T.shape == (8, 1)
    assert D.shape == (8, 1)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="no checkpoint in the repository")
def test_committed_checkpoint_loads():
    model = load_model(CHECKPOINT)
    T, D = model(torch.rand(4, 2))
    assert torch.isfinite(T).all() and torch.isfinite(D).all()
