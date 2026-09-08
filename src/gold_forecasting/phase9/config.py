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

    phase7_reference_run: Literal["20260907T014255645674Z-b2afe281"] = (
        "20260907T014255645674Z-b2afe281"
    )
    phase7_reference_code: Literal[
        "3f0a703568224fe9169b1e9f8d61dad131f0005b"
    ] = "3f0a703568224fe9169b1e9f8d61dad131f0005b"
    phase7_reference_completion: Literal[
        "sha256:beac58092d06350bb067fbd2df144bb9cb2507f45953c19487474c87d0ceca0f"
    ] = (
        "sha256:beac58092d06350bb067fbd2df144bb9cb2507f45953c19487474c87d0ceca0f"
    )
    phase7_champion_config: str = "phase7_champion.yaml"

    phase8_reference_run: Literal["20260908T020044407464Z-24f4c0d4"] = (
        "20260908T020044407464Z-24f4c0d4"
    )
    phase8_reference_code: Literal[
        "75945fe70606c8200cebc66678d0e220db5fb0ad"
    ] = "75945fe70606c8200cebc66678d0e220db5fb0ad"
    phase8_reference_completion: Literal[
        "sha256:9bce4a80c7e0d65fc40ccd4e1fea3c1ef2d56022f8b60720f3d1a8b54cfbb922"
    ] = (
        "sha256:9bce4a80c7e0d65fc40ccd4e1fea3c1ef2d56022f8b60720f3d1a8b54cfbb922"
    )

    test_years: tuple[StrictInt, ...] = (2022, 2023, 2024)
    seeds: tuple[StrictInt, ...] = (20260906, 20260907)
    benchmark_variants: tuple[Literal["direct", "recursive"], ...] = (
        "direct",
        "recursive",
    )

    path_timeframe: Literal["3min"] = "3min"
    path_steps: Literal[5] = 5
    path_minutes: Literal[15] = 15
    quantiles: tuple[float, float, float] = (0.10, 0.50, 0.90)
    neutral_threshold_bps: float = Field(default=6.0, gt=0.0, le=100.0)
    reconstruction_clip_log_bps: float = Field(
        default=5_000.0,
        ge=500.0,
        le=9_000.0,
    )

    timeframes: tuple[str, ...] = _PATH_TIMEFRAMES
    sequence_lengths: dict[str, StrictInt] = {
        "1min": 60,
        "3min": 20,
        "15min": 8,
    }
    encoder_hidden_size: StrictInt = Field(default=32, ge=8, le=128)
    fusion_size: StrictInt = Field(default=32, ge=8, le=128)
    parameter_budget: StrictInt = Field(
        default=200_000,
        ge=10_000,
        le=1_000_000,
    )

    batch_size: StrictInt = Field(default=2048, ge=32, le=16_384)
    max_epochs: StrictInt = Field(default=24, ge=2, le=100)
    early_stopping_patience: StrictInt = Field(default=4, ge=1, le=20)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=0.1)
    weight_decay: float = Field(default=1e-4, ge=0.0, le=0.1)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0, le=10.0)
    class_weighting: Literal["none", "balanced"] = "none"
    minimum_policy_trades: StrictInt = Field(default=20, ge=1, le=10_000)

    direction_loss_weight: float = Field(default=1.0, gt=0.0, le=10.0)
    return_loss_weight: float = Field(default=0.25, ge=0.0, le=10.0)
    range_loss_weight: float = Field(default=0.10, ge=0.0, le=10.0)
    volatility_loss_weight: float = Field(default=0.10, ge=0.0, le=10.0)
    path_quantile_loss_weight: float = Field(default=1.0, gt=0.0, le=10.0)
    aggregate_quantile_loss_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=10.0,
    )
    cumulative_return_loss_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=10.0,
    )
    temporal_consistency_loss_weight: float = Field(
        default=0.10,
        ge=0.0,
        le=10.0,
    )

    device: Literal["auto", "cpu", "cuda"] = "auto"
    mixed_precision: bool = True
    deterministic_algorithms: bool = True
    output_directory: str = "reports/phase9_runs"

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if self.test_years != (2022, 2023, 2024):
            raise ValueError(
                "phase-9 outer years must remain 2022, 2023, 2024"
            )
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError(
                "phase-9 requires at least two unique research seeds"
            )
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("phase-9 seeds must be nonnegative")
        if self.benchmark_variants != ("direct", "recursive"):
            raise ValueError(
                "phase9-v1 must benchmark direct first and recursive second"
            )
        if self.path_steps * 3 != self.path_minutes:
            raise ValueError(
                "phase-9 path must contain exactly five 3min candles"
            )
        if self.quantiles != (0.10, 0.50, 0.90):
            raise ValueError(
                "phase9-v1 quantiles are frozen at 0.10/0.50/0.90"
            )
        if self.timeframes != _PATH_TIMEFRAMES:
            raise ValueError(
                "phase9-v1 core timeframes must be 1min, 3min, 15min"
            )
        if tuple(self.sequence_lengths) != self.timeframes:
            raise ValueError(
                "sequence_lengths must follow the timeframe order"
            )
        if any(
            value < 2 or value > 120
            for value in self.sequence_lengths.values()
        ):
            raise ValueError(
                "sequence lengths must be between 2 and 120 candles"
            )
        if self.early_stopping_patience >= self.max_epochs:
            raise ValueError(
                "early stopping patience must be smaller than max_epochs"
            )
        for field_name in ("data_config", "phase7_champion_config"):
            value = Path(getattr(self, field_name))
            if (
                value.is_absolute()
                or ".." in value.parts
                or value.suffix.lower() not in {".yaml", ".yml"}
            ):
                raise ValueError(
                    f"{field_name} must be a relative YAML path"
                )
        output = Path(self.output_directory)
        if (
            output.is_absolute()
            or ".." in output.parts
            or output.parts[:1] != ("reports",)
            or len(output.parts) < 2
        ):
            raise ValueError(
                "output_directory must be a dedicated reports subdirectory"
            )
        return self


def load_phase9_config(path: str | Path) -> Phase9Config:
    return Phase9Config.model_validate(
        _load_yaml_mapping(Path(path))
    )


__all__ = ["Phase9Config", "load_phase9_config"]
