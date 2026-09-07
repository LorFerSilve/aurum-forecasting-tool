"""Configuration contract for the phase-7 ablation benchmark."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from gold_forecasting.config import _load_yaml_mapping


class Phase7Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    protocol_version: Literal["phase7-v1"] = "phase7-v1"
    data_config: str = "phase5.yaml"
    features_config: str = "features_phase7.yaml"
    horizons: tuple[StrictInt, ...] = (3, 6, 9, 12, 15, 30, 60, 180)
    test_years: tuple[StrictInt, ...] = (2022, 2023, 2024)
    seed: int = Field(default=20260906, ge=0, strict=True)
    xgb_max_rounds: int = Field(default=120, ge=2, le=120, strict=True)
    xgb_early_stopping_rounds: int = Field(default=10, ge=1, le=10, strict=True)
    minimum_policy_trades: int = Field(default=20, ge=1, strict=True)
    output_directory: str = "reports/phase7_runs"

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be nonempty, sorted and unique")
        if not set(self.horizons) <= {3, 6, 9, 12, 15, 30, 60, 180}:
            raise ValueError("phase-7 horizon is outside the frozen development protocol")
        if not self.test_years or tuple(sorted(set(self.test_years))) != self.test_years:
            raise ValueError("test_years must be nonempty, sorted and unique")
        if not set(self.test_years) <= {2022, 2023, 2024}:
            raise ValueError("phase-7 outer years must remain inside development 2022-2024")
        if self.xgb_early_stopping_rounds > self.xgb_max_rounds:
            raise ValueError("early stopping patience cannot exceed max rounds")
        for field_name in ("data_config", "features_config"):
            value = Path(getattr(self, field_name))
            if (
                value.is_absolute()
                or ".." in value.parts
                or value.suffix.lower() not in {".yaml", ".yml"}
            ):
                raise ValueError(f"{field_name} must be a relative YAML path")
        output = Path(self.output_directory)
        if (
            output.is_absolute()
            or ".." in output.parts
            or output.parts[:1] != ("reports",)
            or len(output.parts) < 2
        ):
            raise ValueError("output_directory must be a dedicated relative reports subdirectory")
        return self


def load_phase7_config(path: str | Path) -> Phase7Config:
    return Phase7Config.model_validate(_load_yaml_mapping(Path(path)))


__all__ = ["Phase7Config", "load_phase7_config"]
