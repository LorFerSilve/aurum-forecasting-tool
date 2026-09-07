"""Strict configuration for the separately versioned phase-6 benchmark."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from gold_forecasting.config import _load_yaml_mapping


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    protocol_version: Literal["phase6-v1"] = "phase6-v1"
    data_config: str = "phase5.yaml"
    horizons: tuple[StrictInt, ...] = (3, 6, 9, 12, 15, 30, 60, 180)
    test_years: tuple[StrictInt, ...] = (2022, 2023, 2024)
    seed: int = Field(default=20260906, ge=0, strict=True)
    execution_latency_minutes: Literal[1] = 1
    neutral_threshold_bps: float = Field(default=6.0, allow_inf_nan=False)
    xgb_max_rounds: int = Field(default=120, ge=2, le=120, strict=True)
    xgb_early_stopping_rounds: int = Field(default=10, ge=1, le=10, strict=True)
    minimum_policy_trades: int = Field(default=20, ge=1, strict=True)
    output_directory: str = "reports/benchmark_runs"

    @property
    def full_protocol_scope(self) -> bool:
        return (
            self.horizons == (3, 6, 9, 12, 15, 30, 60, 180)
            and self.test_years == (2022, 2023, 2024)
            and self.seed == 20260906
            and self.xgb_max_rounds == 120
            and self.xgb_early_stopping_rounds == 10
            and self.minimum_policy_trades == 20
        )

    @model_validator(mode="after")
    def validate_scope(self) -> "BenchmarkConfig":
        if self.neutral_threshold_bps != 6.0:
            raise ValueError("neutral threshold is frozen at 6.0 bps in phase6-v1")
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be nonempty, unique and sorted")
        if not set(self.horizons) <= {3, 6, 9, 12, 15, 30, 60, 180}:
            raise ValueError("horizon is outside the frozen phase-6 protocol")
        if not self.test_years or not set(self.test_years) <= {2022, 2023, 2024}:
            raise ValueError("test years must remain in development 2022-2024")
        if tuple(sorted(set(self.test_years))) != self.test_years:
            raise ValueError("test years must be sorted and unique")
        output = Path(self.output_directory)
        if output.is_absolute() or ".." in output.parts or output.parts[:1] != ("reports",):
            raise ValueError("output_directory must be a relative reports subdirectory")
        if len(output.parts) < 2:
            raise ValueError("use a dedicated reports subdirectory")
        data_path = Path(self.data_config)
        if (
            not self.data_config.strip()
            or data_path.is_absolute()
            or ".." in data_path.parts
            or data_path.suffix.lower() not in {".yaml", ".yml"}
        ):
            raise ValueError("data_config must be a relative YAML configuration path")
        if self.xgb_early_stopping_rounds > self.xgb_max_rounds:
            raise ValueError("early-stopping patience cannot exceed the maximum round budget")
        return self


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(_load_yaml_mapping(Path(path)))
