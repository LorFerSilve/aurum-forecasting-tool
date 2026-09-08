from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from gold_forecasting.phase9.config import Phase9Config, load_phase9_config


def test_phase9_defaults_freeze_five_three_minute_candles() -> None:
    config = Phase9Config()

    assert config.protocol_version == "phase9-v1"
    assert config.path_timeframe == "3min"
    assert config.path_steps == 5
    assert config.path_minutes == 15
    assert config.quantiles == (0.10, 0.50, 0.90)
    assert config.timeframes == ("1min", "3min", "15min")
    assert config.parameter_budget == 200_000


def test_phase9_rejects_changed_quantiles() -> None:
    with pytest.raises(ValidationError, match="quantiles"):
        Phase9Config(quantiles=(0.05, 0.50, 0.95))


def test_phase9_config_loads_repository_yaml() -> None:
    config = load_phase9_config(Path("configs/phase9.yaml"))

    assert config.path_steps == 5
    assert config.output_directory == "reports/phase9_runs"
