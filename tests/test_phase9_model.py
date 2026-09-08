from __future__ import annotations

import pytest
import torch

from gold_forecasting.phase9.model import FuturePathGRU, Phase9ModelError


def _inputs(batch: int = 6) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    sequences = {
        "1min": torch.randn(batch, 12, 6),
        "3min": torch.randn(batch, 8, 6),
        "15min": torch.randn(batch, 4, 6),
    }
    availability = torch.tensor(
        [
            [True, True, True],
            [False, True, True],
            [True, True, False],
            [False, True, False],
            [True, False, True],
            [True, True, True],
        ]
    )
    return sequences, availability


def test_future_path_gru_outputs_ordered_quantiles_and_valid_shapes() -> None:
    torch.manual_seed(17)
    model = FuturePathGRU(
        ("1min", "3min", "15min"),
        input_size=6,
        hidden_size=16,
        fusion_size=12,
        parameter_budget=100_000,
    )
    sequences, availability = _inputs()

    output = model(sequences, availability)

    assert output.direction_logits.shape == (6, 3)
    assert output.path_quantiles.shape == (6, 5, 4, 3)
    assert output.aggregate_quantiles.shape == (6, 4, 3)
    assert output.fusion_weights.shape == (6, 3)
    assert torch.all(output.path_quantiles[..., 1] >= output.path_quantiles[..., 0])
    assert torch.all(output.path_quantiles[..., 2] >= output.path_quantiles[..., 1])
    assert torch.all(output.aggregate_quantiles[..., 1] >= output.aggregate_quantiles[..., 0])
    assert torch.all(output.aggregate_quantiles[..., 2] >= output.aggregate_quantiles[..., 1])
    assert torch.all(output.path_quantiles[:, :, 2:, 0] >= 0.0)
    assert torch.allclose(output.fusion_weights.sum(dim=1), torch.ones(6))
    assert model.parameter_count < 100_000


def test_future_path_parameter_budget_fails_closed() -> None:
    with pytest.raises(Phase9ModelError, match="parameter count"):
        FuturePathGRU(
            ("1min", "3min"),
            input_size=6,
            hidden_size=64,
            fusion_size=64,
            parameter_budget=20_000,
        )
