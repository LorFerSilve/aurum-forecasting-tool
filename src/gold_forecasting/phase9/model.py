"""Direct multi-step future-path head built on the phase-8 fused representation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gold_forecasting.phase8.model import MultiTimeframeGRU, Phase8ModelError
from gold_forecasting.phase9.targets import PATH_COMPONENTS, PATH_STEPS

QUANTILES = (0.10, 0.50, 0.90)
_WICK_COMPONENTS = (2, 3)


class Phase9ModelError(ValueError):
    """Raised when the future-path architecture violates its structural contract."""


class OrderedQuantileHead(nn.Module):
    """Three monotonically ordered quantiles for each target component."""

    def __init__(
        self,
        input_size: int,
        *,
        steps: int,
        components: int,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.components = components
        self.linear = nn.Linear(input_size, steps * components * len(QUANTILES))

    def forward(self, fused: Tensor) -> Tensor:
        raw = self.linear(fused).reshape(
            len(fused),
            self.steps,
            self.components,
            len(QUANTILES),
        )
        low_raw = raw[..., 0]
        low = low_raw.clone()
        for component in _WICK_COMPONENTS:
            if component < self.components:
                low[..., component] = F.softplus(low_raw[..., component])
        median = low + F.softplus(raw[..., 1])
        high = median + F.softplus(raw[..., 2])
        result = torch.stack((low, median, high), dim=-1)
        if not torch.isfinite(result).all():
            raise Phase9ModelError("quantile head produced non-finite values")
        return result


class RecursiveQuantilePathHead(nn.Module):
    """Free-running one-step decoder used only as the recursive baseline."""

    def __init__(
        self,
        fusion_size: int,
        *,
        steps: int = PATH_STEPS,
        components: int = len(PATH_COMPONENTS),
    ) -> None:
        super().__init__()
        self.steps = steps
        self.components = components
        self.initial_state = nn.Linear(fusion_size, fusion_size)
        self.cell = nn.GRUCell(components, fusion_size)
        self.step_head = OrderedQuantileHead(
            fusion_size,
            steps=1,
            components=components,
        )

    def forward(self, fused: Tensor) -> Tensor:
        hidden = torch.tanh(self.initial_state(fused))
        previous_median = torch.zeros(
            len(fused),
            self.components,
            dtype=fused.dtype,
            device=fused.device,
        )
        outputs: list[Tensor] = []
        for _ in range(self.steps):
            hidden = self.cell(previous_median, hidden)
            quantiles = self.step_head(hidden).squeeze(1)
            outputs.append(quantiles)
            previous_median = quantiles[..., 1]
        result = torch.stack(outputs, dim=1)
        if not torch.isfinite(result).all():
            raise Phase9ModelError(
                "recursive path head produced non-finite values"
            )
        return result


@dataclass(frozen=True, slots=True)
class Phase9Outputs:
    direction_logits: Tensor
    normalized_return: Tensor
    normalized_range: Tensor
    normalized_volatility: Tensor
    path_quantiles: Tensor
    aggregate_quantiles: Tensor
    fusion_weights: Tensor


class FuturePathGRU(nn.Module):
    """Phase-8 neural core plus direct five-step and direct 15min quantile heads."""

    def __init__(
        self,
        timeframes: tuple[str, ...],
        *,
        input_size: int,
        hidden_size: int,
        fusion_size: int,
        parameter_budget: int,
    ) -> None:
        super().__init__()
        try:
            self.core = MultiTimeframeGRU(
                timeframes,
                input_size=input_size,
                hidden_size=hidden_size,
                fusion_size=fusion_size,
                parameter_budget=parameter_budget,
            )
        except Phase8ModelError as exc:
            raise Phase9ModelError(
                f"phase-9 core violates model contract: {exc}"
            ) from exc
        self.path_head = OrderedQuantileHead(
            fusion_size,
            steps=PATH_STEPS,
            components=len(PATH_COMPONENTS),
        )
        self.aggregate_head = OrderedQuantileHead(
            fusion_size,
            steps=1,
            components=len(PATH_COMPONENTS),
        )
        self.parameter_count = sum(
            parameter.numel()
            for parameter in self.parameters()
        )
        if self.parameter_count > parameter_budget:
            raise Phase9ModelError(
                f"phase-9 parameter count {self.parameter_count} exceeds budget "
                f"{parameter_budget}"
            )

    @property
    def timeframes(self) -> tuple[str, ...]:
        return self.core.timeframes

    def forward(
        self,
        sequences: dict[str, Tensor],
        availability: Tensor,
    ) -> Phase9Outputs:
        fused, weights = self.core.encode(sequences, availability)
        direction = self.core.direction_head(fused)
        normalized_return = self.core.return_head(fused).squeeze(-1)
        normalized_range = F.softplus(self.core.range_head(fused).squeeze(-1))
        normalized_volatility = F.softplus(
            self.core.volatility_head(fused).squeeze(-1)
        )
        path = self.path_head(fused)
        aggregate = self.aggregate_head(fused).squeeze(1)
        tensors = (
            direction,
            normalized_return,
            normalized_range,
            normalized_volatility,
            path,
            aggregate,
            weights,
        )
        if any(not torch.isfinite(value).all() for value in tensors):
            raise Phase9ModelError("phase-9 model produced non-finite outputs")
        if (path[..., 1] < path[..., 0]).any() or (path[..., 2] < path[..., 1]).any():
            raise Phase9ModelError("path quantiles are not ordered")
        if (
            (aggregate[..., 1] < aggregate[..., 0]).any()
            or (aggregate[..., 2] < aggregate[..., 1]).any()
        ):
            raise Phase9ModelError("aggregate quantiles are not ordered")
        return Phase9Outputs(
            direction_logits=direction,
            normalized_return=normalized_return,
            normalized_range=normalized_range,
            normalized_volatility=normalized_volatility,
            path_quantiles=path,
            aggregate_quantiles=aggregate,
            fusion_weights=weights,
        )


class RecursiveFuturePathGRU(FuturePathGRU):
    """Free-running recursive one-step baseline with the same auxiliary heads."""

    def __init__(
        self,
        timeframes: tuple[str, ...],
        *,
        input_size: int,
        hidden_size: int,
        fusion_size: int,
        parameter_budget: int,
    ) -> None:
        super().__init__(
            timeframes,
            input_size=input_size,
            hidden_size=hidden_size,
            fusion_size=fusion_size,
            parameter_budget=parameter_budget,
        )
        self.path_head = RecursiveQuantilePathHead(
            fusion_size,
        )
        self.parameter_count = sum(
            parameter.numel()
            for parameter in self.parameters()
        )
        if self.parameter_count > parameter_budget:
            raise Phase9ModelError(
                f"recursive phase-9 parameter count {self.parameter_count} "
                f"exceeds budget {parameter_budget}"
            )


__all__ = [
    "QUANTILES",
    "FuturePathGRU",
    "OrderedQuantileHead",
    "RecursiveFuturePathGRU",
    "RecursiveQuantilePathHead",
    "Phase9ModelError",
    "Phase9Outputs",
]
