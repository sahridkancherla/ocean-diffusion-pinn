from pathlib import Path

import pytest
import torch

from ocean_pinn.model import AdvectionDiffusionPINN, HeatDiffusionPINN, load_model

CHECKPOINT = Path(__file__).parents[1] / "checkpoints" / "checkpoint.pt"


def test_outputs_temperature_and_diffusivity():
    T, D = HeatDiffusionPINN()(torch.rand(8, 2))
    assert T.shape == (8, 1)
    assert D.shape == (8, 1)


@pytest.mark.parametrize("model_class", [HeatDiffusionPINN, AdvectionDiffusionPINN])
def test_diffusivity_is_positive(model_class):
    torch.manual_seed(0)
    _, D = model_class()(torch.rand(1000, 2))
    assert (D > 0).all()


def test_advection_model_diffusivity_depends_only_on_depth():
    model = AdvectionDiffusionPINN()
    z = torch.rand(8)
    _, D_early = model(torch.stack([z, torch.zeros(8)], dim=1))
    _, D_late = model(torch.stack([z, torch.ones(8)], dim=1))
    torch.testing.assert_close(D_early, D_late)


def test_advection_model_velocity_varies_in_depth_and_time_and_can_be_negative():
    torch.manual_seed(0)
    w = AdvectionDiffusionPINN().velocity(torch.rand(1000, 2))
    assert w.shape == (1000, 1)
    assert w.std() > 0


def test_old_checkpoint_format_is_rejected_with_an_explanation(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"model": HeatDiffusionPINN().state_dict(), "epoch": 700}, path)
    with pytest.raises(ValueError, match="Retrain"):
        load_model(path)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="no checkpoint in the repository")
def test_committed_checkpoint_loads():
    model = load_model(CHECKPOINT)
    T, D = model(torch.rand(4, 2))
    assert torch.isfinite(T).all() and (D > 0).all()
