"""Frozen structural contract for the phase-9 future-candle path challenger."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from gold_forecasting.config import _load_yaml_mapping

_PATH_TIMEFRAMES = ("1min", "3min", "15min")


class Phase9Config(BaseModel):
    """Precommitted architecture/target budget before phase-9 empirical results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    protocol_version: Literal["phase9-v1"] = "phase9-v1"
    data_config: str = "phase5.yaml"

    path_timeframe: Literal["3min"] = "3min"
    path_steps: Literal[5] = 5
    path_minutes: Literal[15] = 15
    quantiles: tuple[float, float, float] = (0.10, 0.50, 0.90)
    neutral_threshold_bps: float = Field(default=6.0, gt=0.0, le=100.0)
    reconstruction_clip_log_bps: float = Field(default=5_000.0, ge=500.0, le=9_000.0)

    timeframes: tuple[str, ...] = _PATH_TIMEFRAMES
    sequence_lengths: dict[str, StrictInt] = {
        "1min": 60,
        "3min": 20,
        "15min": 8,
    }
    encoder_hidden_size: StrictInt = Field(default=32, ge=8, le=128)
    fusion_size: StrictInt = Field(default=32, ge=8, le=128)
    parameter_budget: StrictInt = Field(default=200_000, ge=10_000, le=1_000_000)

    direction_loss_weight: float = Field(default=1.0, gt=0.0, le=10.0)
    path_quantile_loss_weight: float = Field(default=1.0, gt=0.0, le=10.0)
    aggregate_quantile_loss_weight: float = Field(default=0.25, ge=0.0, le=10.0)
    temporal_consistency_loss_weight: float = Field(default=0.10, ge=0.0, le=10.0)

    output_directory: str = "reports/phase9_runs"

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if self.path_steps * 3 != self.path_minutes:
            raise ValueError("phase-9 path must contain exactly five 3min candles")
        if self.quantiles != (0.10, 0.50, 0.90):
            raise ValueError("phase9-v1 quantiles are frozen at 0.10/0.50/0.90")
        if self.timeframes != _PATH_TIMEFRAMES:
            raise ValueError("phase9-v1 core timeframes must be 1min, 3min, 15min")
        if tuple(self.sequence_lengths) != self.timeframes:
            raise ValueError("sequence_lengths must follow the timeframe order")
        if any(value < 2 or value > 120 for value in self.sequence_lengths.values()):
            raise ValueError("sequence lengths must be between 2 and 120 candles")
        data = Path(self.data_config)
        if data.is_absolute() or ".." in data.parts or data.suffix.lower() not in {
            ".yaml",
            ".yml",
        }:
            raise ValueError("data_config must be a relative YAML path")
        output = Path(self.output_directory)
        if (
            output.is_absolute()
            or ".." in output.parts
            or output.parts[:1] != ("reports",)
            or len(output.parts) < 2
        ):
            raise ValueError("output_directory must be a dedicated reports subdirectory")
        return self


def load_phase9_config(path: str | Path) -> Phase9Config:
    return Phase9Config.model_validate(_load_yaml_mapping(Path(path)))


__all__ = ["Phase9Config", "load_phase9_config"]
