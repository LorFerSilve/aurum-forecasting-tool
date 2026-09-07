from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from gold_forecasting.phase7.config import Phase7Config, load_phase7_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase7_repository_config_preserves_phase6_development_contract() -> None:
    config = load_phase7_config(PROJECT_ROOT / "configs" / "phase7.yaml")

    assert config.seed == 20260906
    assert config.horizons == (3, 6, 9, 12, 15, 30, 60, 180)
    assert config.test_years == (2022, 2023, 2024)
    assert config.data_config == "phase5.yaml"
    assert config.features_config == "features_phase7.yaml"


def test_phase7_config_rejects_final_holdout_year() -> None:
    with pytest.raises(ValidationError):
        Phase7Config(test_years=(2022, 2023, 2024, 2025))
