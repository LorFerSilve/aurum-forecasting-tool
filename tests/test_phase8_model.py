from __future__ import annotations

import pytest
import torch

from gold_forecasting.phase8.model import MultiTimeframeGRU, Phase8ModelError


def test_multitimeframe_gru_outputs_valid_shapes_and_masks_missing_branch() -> None:
    torch.manual_seed(7)
    model = MultiTimeframeGRU(
        ("1min", "3min", "15min"),
        input_size=6,
        hidden_size=16,
        fusion_size=12,
        parameter_budget=100_000,
    )
    batch = 5
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
        ]
    )

    output = model(sequences, availability)

    assert output.direction_logits.shape == (batch, 3)
    assert output.normalized_return.shape == (batch,)
    assert output.normalized_range.shape == (batch,)
    assert output.normalized_volatility.shape == (batch,)
    assert output.fusion_weights.shape == (batch, 3)
    assert torch.isfinite(output.direction_logits).all()
    assert (output.normalized_range >= 0).all()
    assert (output.normalized_volatility >= 0).all()
    assert output.fusion_weights[1, 0].item() == pytest.approx(0.0)
    assert torch.allclose(
        output.fusion_weights.sum(dim=1),
        torch.ones(batch),
    )
    assert model.parameter_count < 100_000


def test_parameter_budget_fails_closed() -> None:
    with pytest.raises(Phase8ModelError, match="parameter count"):
        MultiTimeframeGRU(
            ("1min", "3min"),
            input_size=6,
            hidden_size=64,
            fusion_size=64,
            parameter_budget=1000,
        )


def test_every_sample_requires_at_least_one_available_encoder() -> None:
    model = MultiTimeframeGRU(
        ("3min",),
        input_size=6,
        hidden_size=8,
        fusion_size=8,
        parameter_budget=20_000,
    )
    with pytest.raises(Phase8ModelError, match="at least one"):
        model(
            {"3min": torch.zeros(2, 4, 6)},
            torch.zeros(2, 1, dtype=torch.bool),
        )
