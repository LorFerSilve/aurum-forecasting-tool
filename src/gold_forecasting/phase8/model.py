"""Compact gated multi-timeframe GRU used as the phase-8 neural challenger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

DIRECTION_CLASSES: Final[int] = 3


class Phase8ModelError(ValueError):
    """Raised when the compact neural architecture violates its frozen budget."""


class GRUEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, values: Tensor) -> Tensor:
        output, hidden = self.gru(values)
        del output
        return cast(Tensor, self.norm(hidden[-1]))


@dataclass(frozen=True, slots=True)
class NeuralOutputs:
    direction_logits: Tensor
    normalized_return: Tensor
    normalized_range: Tensor
    normalized_volatility: Tensor
    fusion_weights: Tensor


class MultiTimeframeGRU(nn.Module):
    """One GRU per timeframe plus availability-aware gated fusion and four heads."""

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
        if not timeframes or len(set(timeframes)) != len(timeframes):
            raise Phase8ModelError("timeframes must be nonempty and unique")
        self.timeframes = timeframes
        self.encoders = nn.ModuleDict(
            {name: GRUEncoder(input_size, hidden_size) for name in timeframes}
        )
        self.projections = nn.ModuleDict(
            {name: nn.Linear(hidden_size, fusion_size) for name in timeframes}
        )
        self.gates = nn.ModuleDict(
            {name: nn.Linear(hidden_size, 1) for name in timeframes}
        )
        self.fusion_norm = nn.LayerNorm(fusion_size)
        self.direction_head = nn.Linear(fusion_size, DIRECTION_CLASSES)
        self.return_head = nn.Linear(fusion_size, 1)
        self.range_head = nn.Linear(fusion_size, 1)
        self.volatility_head = nn.Linear(fusion_size, 1)
        self.parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if self.parameter_count > parameter_budget:
            raise Phase8ModelError(
                f"neural parameter count {self.parameter_count} exceeds budget {parameter_budget}"
            )

    def forward(
        self,
        sequences: dict[str, Tensor],
        availability: Tensor,
    ) -> NeuralOutputs:
        if availability.ndim != 2 or availability.shape[1] != len(self.timeframes):
            raise Phase8ModelError("availability must have shape [batch, timeframes]")
        embeddings: list[Tensor] = []
        gate_logits: list[Tensor] = []
        for index, name in enumerate(self.timeframes):
            if name not in sequences:
                raise Phase8ModelError(f"missing sequence tensor for {name}")
            encoded = self.encoders[name](sequences[name])
            embeddings.append(torch.tanh(self.projections[name](encoded)))
            gate_logits.append(self.gates[name](encoded).squeeze(-1))
            if not torch.isfinite(encoded).all():
                raise Phase8ModelError(f"non-finite encoder output for {name}")
            if availability[:, index].ndim != 1:
                raise Phase8ModelError("invalid availability column")
        stacked = torch.stack(embeddings, dim=1)
        logits = torch.stack(gate_logits, dim=1)
        mask = availability.to(dtype=torch.bool)
        if (~mask).all(dim=1).any():
            raise Phase8ModelError("every sample requires at least one available timeframe")
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        fused = self.fusion_norm((stacked * weights.unsqueeze(-1)).sum(dim=1))
        direction = self.direction_head(fused)
        normalized_return = self.return_head(fused).squeeze(-1)
        normalized_range = F.softplus(self.range_head(fused).squeeze(-1))
        normalized_volatility = F.softplus(self.volatility_head(fused).squeeze(-1))
        tensors = (
            direction,
            normalized_return,
            normalized_range,
            normalized_volatility,
            weights,
        )
        if any(not torch.isfinite(value).all() for value in tensors):
            raise Phase8ModelError("model produced non-finite outputs")
        return NeuralOutputs(
            direction_logits=direction,
            normalized_return=normalized_return,
            normalized_range=normalized_range,
            normalized_volatility=normalized_volatility,
            fusion_weights=weights,
        )


__all__ = [
    "DIRECTION_CLASSES",
    "GRUEncoder",
    "MultiTimeframeGRU",
    "NeuralOutputs",
    "Phase8ModelError",
]
