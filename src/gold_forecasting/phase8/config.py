"""Frozen configuration contract for the phase-8 neural challenger."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from gold_forecasting.config import _load_yaml_mapping

_ALLOWED_HORIZONS = (3, 6, 9, 12, 15, 30, 60, 180)
_ALLOWED_TIMEFRAMES = ("1min", "3min", "15min", "1h")


class Phase8Config(BaseModel):
    """Research-only budget for the compact multi-timeframe GRU challenger."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    protocol_version: Literal["phase8-v1"] = "phase8-v1"
    data_config: str = "phase5.yaml"
    phase7_reference_directory: str = (
        "reports/phase7_runs/20260907T014255645674Z-b2afe281"
    )
    phase7_champion_config: str = "phase7_champion.yaml"
    horizons: tuple[StrictInt, ...] = _ALLOWED_HORIZONS
    test_years: tuple[StrictInt, ...] = (2022, 2023, 2024)
    seeds: tuple[StrictInt, ...] = (20260906, 20260907)

    timeframes: tuple[str, ...] = _ALLOWED_TIMEFRAMES
    sequence_lengths: dict[str, StrictInt] = {
        "1min": 60,
        "3min": 20,
        "15min": 8,
        "1h": 4,
    }
    horizon_timeframes: dict[StrictInt, tuple[str, ...]] = {
        3: ("1min", "3min", "15min"),
        6: ("1min", "3min", "15min"),
        9: ("1min", "3min", "15min"),
        12: ("1min", "3min", "15min"),
        15: ("1min", "3min", "15min"),
        30: ("3min", "15min", "1h"),
        60: ("3min", "15min", "1h"),
        180: ("3min", "15min", "1h"),
    }

    encoder_hidden_size: StrictInt = Field(default=32, ge=8, le=128)
    fusion_size: StrictInt = Field(default=32, ge=8, le=128)
    parameter_budget: StrictInt = Field(default=150_000, ge=1_000, le=1_000_000)

    batch_size: StrictInt = Field(default=2048, ge=32, le=16_384)
    max_epochs: StrictInt = Field(default=24, ge=2, le=100)
    early_stopping_patience: StrictInt = Field(default=4, ge=1, le=20)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=0.1)
    weight_decay: float = Field(default=1e-4, ge=0.0, le=0.1)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0, le=10.0)
    direction_loss_weight: float = Field(default=1.0, gt=0.0, le=10.0)
    return_loss_weight: float = Field(default=0.25, ge=0.0, le=10.0)
    range_loss_weight: float = Field(default=0.10, ge=0.0, le=10.0)
    volatility_loss_weight: float = Field(default=0.10, ge=0.0, le=10.0)
    class_weighting: Literal["none", "balanced"] = "none"

    device: Literal["auto", "cpu", "cuda"] = "auto"
    mixed_precision: bool = True
    deterministic_algorithms: bool = True
    num_data_workers: StrictInt = Field(default=0, ge=0, le=8)
    minimum_policy_trades: StrictInt = Field(default=20, ge=1, le=10_000)
    output_directory: str = "reports/phase8_runs"

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if self.horizons != _ALLOWED_HORIZONS:
            raise ValueError("phase-8 horizons must preserve the frozen v0.2 horizon order")
        if self.test_years != (2022, 2023, 2024):
            raise ValueError("phase-8 outer years must remain 2022, 2023, 2024")
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("phase-8 requires at least two unique research seeds")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative")
        if self.timeframes != _ALLOWED_TIMEFRAMES:
            raise ValueError(
                "phase-8-v1 timeframes must be exactly 1min, 3min, 15min, 1h"
            )
        if tuple(self.sequence_lengths) != self.timeframes:
            raise ValueError("sequence_lengths must follow the configured timeframe order")
        if any(value < 2 or value > 120 for value in self.sequence_lengths.values()):
            raise ValueError("sequence lengths must be between 2 and 120 candles")
        if tuple(self.horizon_timeframes) != self.horizons:
            raise ValueError("horizon_timeframes must cover every horizon in order")
        for horizon, timeframes in self.horizon_timeframes.items():
            if not timeframes or len(set(timeframes)) != len(timeframes):
                raise ValueError(f"horizon {horizon} requires unique nonempty timeframes")
            if any(name not in self.timeframes for name in timeframes):
                raise ValueError(f"horizon {horizon} uses an unavailable timeframe")
            if "3min" not in timeframes:
                raise ValueError(f"horizon {horizon} must retain the 3min anchor encoder")
        if self.early_stopping_patience >= self.max_epochs:
            raise ValueError("early stopping patience must be smaller than max_epochs")
        if not 0.0 < self.learning_rate <= 0.1:
            raise ValueError("learning_rate must be in (0, 0.1]")
        for field_name in ("data_config", "phase7_champion_config"):
            value = Path(getattr(self, field_name))
            if value.is_absolute() or ".." in value.parts or value.suffix.lower() not in {
                ".yaml",
                ".yml",
            }:
                raise ValueError(f"{field_name} must be a relative YAML path")
        reference = Path(self.phase7_reference_directory)
        if (
            reference.is_absolute()
            or ".." in reference.parts
            or reference.parts[:2] != ("reports", "phase7_runs")
            or len(reference.parts) != 3
        ):
            raise ValueError(
                "phase7_reference_directory must name one relative reports/phase7_runs run"
            )
        output = Path(self.output_directory)
        if (
            output.is_absolute()
            or ".." in output.parts
            or output.parts[:1] != ("reports",)
            or len(output.parts) < 2
        ):
            raise ValueError("output_directory must be a dedicated reports subdirectory")
        return self


def load_phase8_config(path: str | Path) -> Phase8Config:
    return Phase8Config.model_validate(_load_yaml_mapping(Path(path)))


__all__ = ["Phase8Config", "load_phase8_config"]
