"""Robust phase-9 path losses and differentiable temporal consistency."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from gold_forecasting.phase9.model import QUANTILES
from gold_forecasting.phase9.targets import PATH_COMPONENTS, PATH_STEPS


class Phase9LossError(ValueError):
    """Raised when a future-path loss receives malformed tensors."""


def pinball_loss(
    predicted_quantiles: Tensor,
    target: Tensor,
) -> Tensor:
    """Mean pinball loss for the frozen 10/50/90% quantiles."""

    if predicted_quantiles.shape[:-1] != target.shape:
        raise Phase9LossError("quantile prediction and target shapes do not align")
    if predicted_quantiles.shape[-1] != len(QUANTILES):
        raise Phase9LossError("phase9-v1 requires exactly three quantiles")
    if not torch.isfinite(predicted_quantiles).all() or not torch.isfinite(target).all():
        raise Phase9LossError("quantile loss inputs must be finite")
    q = torch.tensor(
        QUANTILES,
        dtype=predicted_quantiles.dtype,
        device=predicted_quantiles.device,
    )
    error = target.unsqueeze(-1) - predicted_quantiles
    return torch.maximum((q - 1.0) * error, q * error).mean()


def reconstruct_path_torch(
    representation: Tensor,
    *,
    clip_log_bps: float,
) -> Tensor:
    """Differentiably reconstruct normalized OHLC with anchor close fixed at 1."""

    if representation.ndim != 3 or representation.shape[1:] != (
        PATH_STEPS,
        len(PATH_COMPONENTS),
    ):
        raise Phase9LossError("representation must have shape [batch,5,4]")
    if not torch.isfinite(representation).all():
        raise Phase9LossError("path representation must be finite")
    previous = torch.ones(
        len(representation),
        dtype=representation.dtype,
        device=representation.device,
    )
    candles: list[Tensor] = []
    for step in range(PATH_STEPS):
        gap = representation[:, step, 0].clamp(
            -clip_log_bps,
            clip_log_bps,
        )
        body = representation[:, step, 1].clamp(
            -clip_log_bps,
            clip_log_bps,
        )
        upper = representation[:, step, 2].clamp(0.0, clip_log_bps)
        lower = representation[:, step, 3].clamp(0.0, clip_log_bps)
        open_price = previous * torch.exp(gap / 10_000.0)
        close = open_price * torch.exp(body / 10_000.0)
        high = torch.maximum(open_price, close) * torch.exp(upper / 10_000.0)
        low = torch.minimum(open_price, close) * torch.exp(-lower / 10_000.0)
        candles.append(torch.stack((open_price, high, low, close), dim=-1))
        previous = close
    result = torch.stack(candles, dim=1)
    if not torch.isfinite(result).all():
        raise Phase9LossError("differentiable reconstruction became non-finite")
    return result


def aggregate_path_representation_torch(
    representation: Tensor,
    *,
    clip_log_bps: float,
) -> Tensor:
    """Convert a five-step representation into one equivalent 15min candle."""

    path = reconstruct_path_torch(
        representation,
        clip_log_bps=clip_log_bps,
    )
    open_price = path[:, 0, 0]
    high = path[:, :, 1].amax(dim=1)
    low = path[:, :, 2].amin(dim=1)
    close = path[:, -1, 3]
    gap = 10_000.0 * torch.log(open_price)
    body = 10_000.0 * torch.log(close / open_price)
    upper = 10_000.0 * torch.log(high / torch.maximum(open_price, close))
    lower = 10_000.0 * torch.log(torch.minimum(open_price, close) / low)
    result = torch.stack((gap, body, upper, lower), dim=-1)
    if not torch.isfinite(result).all():
        raise Phase9LossError("aggregate path representation became non-finite")
    return result


def temporal_consistency_loss(
    path_quantiles: Tensor,
    aggregate_quantiles: Tensor,
    *,
    clip_log_bps: float,
) -> Tensor:
    """Huber consistency between median five-step path and direct 15min head."""

    if path_quantiles.ndim != 4 or aggregate_quantiles.ndim != 3:
        raise Phase9LossError("invalid path/direct aggregate quantile ranks")
    path_median = path_quantiles[..., 1]
    aggregate_median = aggregate_quantiles[..., 1]
    derived = aggregate_path_representation_torch(
        path_median,
        clip_log_bps=clip_log_bps,
    )
    if derived.shape != aggregate_median.shape:
        raise Phase9LossError("derived and direct aggregate shapes do not align")
    return F.huber_loss(
        aggregate_median,
        derived,
        delta=1.0,
    )


__all__ = [
    "Phase9LossError",
    "aggregate_path_representation_torch",
    "pinball_loss",
    "reconstruct_path_torch",
    "temporal_consistency_loss",
]
