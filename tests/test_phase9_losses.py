from __future__ import annotations

import torch

from gold_forecasting.phase9.losses import (
    aggregate_path_representation_torch,
    pinball_loss,
    reconstruct_path_torch,
    temporal_consistency_loss,
)


def test_pinball_loss_is_zero_for_exact_quantiles() -> None:
    target = torch.tensor([[[1.0, -2.0, 3.0, 4.0]]])
    predicted = target.unsqueeze(-1).repeat(1, 1, 1, 3)

    loss = pinball_loss(predicted, target)

    assert loss.item() == 0.0


def test_differentiable_reconstruction_respects_ohlc_invariants() -> None:
    representation = torch.randn(4, 5, 4) * 20.0
    representation[:, :, 2:] = representation[:, :, 2:].abs()

    path = reconstruct_path_torch(
        representation,
        clip_log_bps=5_000.0,
    )

    assert path.shape == (4, 5, 4)
    assert torch.all(path > 0.0)
    assert torch.all(path[:, :, 1] >= torch.maximum(path[:, :, 0], path[:, :, 3]))
    assert torch.all(path[:, :, 2] <= torch.minimum(path[:, :, 0], path[:, :, 3]))


def test_temporal_consistency_is_zero_when_direct_head_matches_path_median() -> None:
    path_median = torch.randn(3, 5, 4) * 5.0
    path_median[:, :, 2:] = path_median[:, :, 2:].abs()
    path_quantiles = torch.stack(
        (
            path_median - 1.0,
            path_median,
            path_median + 1.0,
        ),
        dim=-1,
    )
    aggregate = aggregate_path_representation_torch(
        path_median,
        clip_log_bps=5_000.0,
    )
    aggregate_quantiles = torch.stack(
        (
            aggregate - 1.0,
            aggregate,
            aggregate + 1.0,
        ),
        dim=-1,
    )

    loss = temporal_consistency_loss(
        path_quantiles,
        aggregate_quantiles,
        clip_log_bps=5_000.0,
    )

    assert loss.item() < 1e-10
