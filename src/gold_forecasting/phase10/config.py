"""Frozen configuration for independently registered Phase-10 source ablations."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, StrictInt, field_validator, model_validator

from gold_forecasting.config import _load_yaml_mapping
from gold_forecasting.phase10.profiles import ContextProfile, profile_for_protocol


class Phase10Config(BaseModel):
    """Only resource locations vary; the research methodology is precommitted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    protocol_version: Literal[
        "phase10-silver-modeled-v1",
        "phase10-dollar-eurusd-modeled-v1",
        "phase10-rate-dfii10-modeled-v1",
    ] = "phase10-silver-modeled-v1"
    data_config: str = "phase5.yaml"
    features_config: str = "features_phase7.yaml"
    source_config: str = "phase10_silver_exploratory.yaml"
    phase7_champion_config: str = "phase7_champion.yaml"
    phase7_reference_run: Literal["20260907T014255645674Z-b2afe281"] = (
        "20260907T014255645674Z-b2afe281"
    )
    phase7_reference_code: Literal["3f0a703568224fe9169b1e9f8d61dad131f0005b"] = (
        "3f0a703568224fe9169b1e9f8d61dad131f0005b"
    )
    phase7_reference_completion: Literal[
        "sha256:beac58092d06350bb067fbd2df144bb9cb2507f45053c19487474c87d0ceca0f"
    ] = "sha256:beac58092d06350bb067fbd2df144bb9cb2507f45053c19487474c87d0ceca0f"
    bundle_path: str = "data/context/phase10/silver/silver.bundle-set.json"
    archive_directory: str = "data/raw/phase10/silver"
    output_directory: str = "reports/phase10_runs"
    test_years: tuple[StrictInt, ...] = (2022, 2023, 2024)
    horizon_minutes: StrictInt = 15
    gap_minutes: StrictInt = 181
    seed: StrictInt = 20260906
    minimum_policy_trades: StrictInt = 20

    @property
    def profile(self) -> ContextProfile:
        """Resolve the immutable source profile bound to this protocol."""
        return profile_for_protocol(self.protocol_version)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _integer_schema(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @model_validator(mode="after")
    def _frozen_scope(self) -> Self:
        for name, expected in (
            ("schema_version", 1),
            ("horizon_minutes", 15),
            ("gap_minutes", 181),
            ("seed", 20260906),
            ("minimum_policy_trades", 20),
        ):
            value = getattr(self, name)
            if type(value) is not int or value != expected:
                raise ValueError(f"{name} must remain frozen at {expected}")
        if self.test_years != (2022, 2023, 2024):
            raise ValueError("outer years must remain exactly 2022, 2023, 2024")
        yaml_fields = (
            "data_config",
            "features_config",
            "source_config",
            "phase7_champion_config",
        )
        for name in (*yaml_fields, "bundle_path", "archive_directory", "output_directory"):
            text = getattr(self, name)
            path = Path(text)
            if (
                not text
                or text.strip() != text
                or path.is_absolute()
                or path.drive
                or ".." in path.parts
                or "\\" in text
                or ":" in text
            ):
                raise ValueError(f"{name} must be a canonical relative path")
            if name in yaml_fields and path.suffix.lower() not in {".yaml", ".yml"}:
                raise ValueError(f"{name} must be a relative YAML file")
        if Path(self.bundle_path).suffix != ".json":
            raise ValueError("bundle_path must be a bundle-set JSON file")
        if Path(self.output_directory).parts[:1] != ("reports",):
            raise ValueError("output_directory must remain under reports/")
        if len(Path(self.output_directory).parts) < 2:
            raise ValueError("output_directory must be a dedicated reports subdirectory")
        return self


def load_phase10_config(path: str | Path) -> Phase10Config:
    return Phase10Config.model_validate(_load_yaml_mapping(Path(path)))


__all__ = ["Phase10Config", "load_phase10_config"]
