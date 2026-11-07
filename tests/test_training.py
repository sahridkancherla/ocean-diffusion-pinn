import math

import pytest
import torch

from ocean_pinn.config import LossWeights, TrainConfig
from ocean_pinn.data import PointSet
from ocean_pinn.evaluation import evaluate, predict
from ocean_pinn.model import AdvectionDiffusionPINN, HeatDiffusionPINN, load_model
from ocean_pinn.training import train


def synthetic_points(n=512):
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(n, 2, generator=generator)
    temperature = torch.cos(math.pi * x[:, 0])
    t = x[:, 1]
    x_bottom = torch.stack([torch.zeros(n), t], dim=1)
    x_top = torch.stack([torch.ones(n), t], dim=1)
    return PointSet(
        x,
        temperature,
        torch.rand(n, 2, generator=generator),
        x_bottom,
        torch.ones(n),
        x_top,
        -torch.ones(n),
    )


@pytest.mark.parametrize("model_class", [HeatDiffusionPINN, AdvectionDiffusionPINN])
def test_training_reduces_loss_and_saves_checkpoint(tmp_path, model_class):
    points = synthetic_points()
    cfg = TrainConfig(
        epochs=20, learning_rate=1e-2, batch_size=128, checkpoint_path=tmp_path / "model.pt"
    )
    torch.manual_seed(0)
    model = model_class()

    history = train(model, points, cfg)

    assert len(history) == cfg.epochs
    assert history[-1] < 0.5 * history[0]
    reloaded = load_model(cfg.checkpoint_path)
    assert type(reloaded) is model_class
    torch.testing.assert_close(predict(reloaded, points.x), predict(model, points.x))


def test_evaluate_averages_every_term():
    points = synthetic_points()
    losses = evaluate(HeatDiffusionPINN(), points, LossWeights(), batch_size=100)
    assert set(losses) == {"data", "physics", "bc_z0", "bc_z1", "total"}
    assert all(math.isfinite(value) for value in losses.values())


def test_evaluate_reports_velocity_terms_for_advection_model():
    losses = evaluate(AdvectionDiffusionPINN(), synthetic_points(), LossWeights())
    assert {"w_size", "w_rough"} <= set(losses)


def test_predict_returns_one_value_per_point():
    points = synthetic_points(n=10)
    temperature, diffusivity = predict(HeatDiffusionPINN(), points.x, batch_size=3)
    assert temperature.shape == diffusivity.shape == (10,)
