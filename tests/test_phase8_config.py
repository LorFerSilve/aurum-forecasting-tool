from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from gold_forecasting.phase8.config import Phase8Config, load_phase8_config
from gold_forecasting.phase8.pipeline import _require_clean_code_version

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repository_phase8_config_preserves_frozen_development_scope() -> None:
    config = load_phase8_config(PROJECT_ROOT / "configs" / "phase8.yaml")

    assert config.protocol_version == "phase8-v1"
    assert config.horizons == (3, 6, 9, 12, 15, 30, 60, 180)
    assert config.test_years == (2022, 2023, 2024)
    assert len(config.seeds) == 2
    assert config.timeframes == ("1min", "3min", "15min", "1h")
    assert "3h" not in config.timeframes
    assert all("3min" in value for value in config.horizon_timeframes.values())


def test_phase8_output_directory_is_gitignored() -> None:
    config = load_phase8_config(PROJECT_ROOT / "configs" / "phase8.yaml")
    rules = {
        line.strip()
        for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    assert f"{config.output_directory}/**" in rules


def test_phase8_requires_multiple_seeds_and_frozen_years() -> None:
    with pytest.raises(ValidationError):
        Phase8Config(seeds=(20260906,))
    with pytest.raises(ValidationError):
        Phase8Config(test_years=(2022, 2023, 2024, 2025))


@pytest.mark.parametrize("value", ["unavailable", "uncommitted", "a" * 40 + "+dirty"])
def test_phase8_formal_run_rejects_unreproducible_git_identity(value: str) -> None:
    with pytest.raises(RuntimeError):
        _require_clean_code_version(value)


def test_phase8_formal_run_accepts_clean_commit() -> None:
    _require_clean_code_version("a" * 40)
