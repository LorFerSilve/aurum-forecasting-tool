from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from gold_forecasting.phase9.config import Phase9Config, load_phase9_config


def test_phase9_defaults_freeze_five_three_minute_candles() -> None:
    config = Phase9Config()

    assert config.protocol_version == "phase9-v1"
    assert config.phase7_reference_run == "20260907T014255645674Z-b2afe281"
    assert config.phase7_reference_code == "3f0a703568224fe9169b1e9f8d61dad131f0005b"
    assert config.phase7_reference_completion == (
        "sha256:beac58092d06350bb067fbd2df144bb9cb2507f45053c19487474c87d0ceca0f"
    )
    assert config.phase8_reference_run == "20260908T020044407464Z-24f4c0d4"
    assert config.phase8_reference_code == "75945fe70606c8200cebc66678d0e220db5fb0ad"
    assert config.phase8_reference_completion == (
        "sha256:9bce4a80c7e0d65fc40ccd4e1fea3c1ef2d56022f8b60720f3d1a8b54cfbb922"
    )
    assert config.test_years == (2022, 2023, 2024)
    assert len(config.seeds) == 2
    assert config.benchmark_variants == ("direct", "recursive")
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


def test_phase9_rejects_changed_benchmark_variant_order() -> None:
    with pytest.raises(ValidationError, match="direct first"):
        Phase9Config(
            benchmark_variants=("recursive", "direct"),
        )
