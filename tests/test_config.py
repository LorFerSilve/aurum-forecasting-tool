"""Tests for strict, cross-file project configuration validation."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from gold_forecasting.config import ConfigError, load_project_config
from gold_forecasting.ingestion import (
    ProviderConfigurationError,
    create_m1_provider,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config_copy(tmp_path: Path) -> Path:
    destination = tmp_path / "configs"
    shutil.copytree(PROJECT_ROOT / "configs", destination)
    return destination / "mvp.yaml"


def _change_yaml(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _threshold_rules(classes: dict[str, Any]) -> None:
    threshold = str(classes["total_threshold_bps"])
    classes["up_rule"] = f"gross_return_bps > {threshold}"
    classes["down_rule"] = f"gross_return_bps < -{threshold}"
    classes["neutral_rule"] = f"-{threshold} <= gross_return_bps <= {threshold}"


def test_loads_root_and_all_components_with_resolved_paths() -> None:
    loaded = load_project_config(PROJECT_ROOT / "configs" / "mvp.yaml")

    assert loaded.root.protocol_version == "v0.0"
    assert loaded.instrument.instrument.id == "XAU_USD"
    assert loaded.features.input_timeframe == "3min"
    assert loaded.features.maximum_prior_candles == 20
    assert loaded.labels.prediction.primary_horizon_minutes == 15
    assert loaded.costs.cost_model.total_round_trip_bps == 4.0
    assert loaded.splits.purging.gap_minutes == 16
    assert loaded.model.release_version == "v0.1.0"
    assert loaded.backtest.backtest.confidence_threshold == 0.5
    assert loaded.config_path.is_absolute()
    assert set(loaded.component_paths) == {
        "instrument",
        "features",
        "labels",
        "costs",
        "splits",
        "model",
        "backtest",
    }
    assert all(path.is_absolute() for path in loaded.component_paths.values())


def test_relative_component_paths_are_resolved_from_root_config(tmp_path: Path) -> None:
    root_path = _config_copy(tmp_path)

    loaded = load_project_config(root_path)

    assert loaded.component_paths["instrument"] == root_path.parent / "instrument.yaml"


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"source_timezone_offset": "+00:00"}, "source_timezone_offset"),
        ({"source_observes_dst": True}, "source_observes_dst"),
        ({"timestamp_semantics": "candle_close"}, "timestamp_semantics"),
        ({"price_side": "ask"}, "price_side"),
        ({"volume_reliable": True}, "volume_reliable"),
        ({"source_url": "https://example.com/data"}, "histdata.com source_url"),
    ],
)
def test_histdata_factory_rejects_config_that_misstates_hard_coded_source_contract(
    update: dict[str, object],
    message: str,
) -> None:
    loaded = load_project_config(PROJECT_ROOT / "configs" / "mvp.yaml")
    provider = loaded.instrument.provider.model_copy(update=update)

    with pytest.raises(ProviderConfigurationError, match=message):
        create_m1_provider(provider)


def test_unknown_root_field_fails_with_file_context(tmp_path: Path) -> None:
    root_path = _config_copy(tmp_path)
    _change_yaml(root_path, lambda config: config.update({"unexpected": True}))

    with pytest.raises(
        ConfigError,
        match=r"(?s)mvp\.yaml.*Extra inputs are not permitted",
    ):
        load_project_config(root_path)


def test_missing_component_reference_fails_clearly(tmp_path: Path) -> None:
    root_path = _config_copy(tmp_path)
    _change_yaml(
        root_path,
        lambda config: config.update({"instrument_config": "does_not_exist.yaml"}),
    )

    with pytest.raises(ConfigError, match="Referenced component configuration does not exist"):
        load_project_config(root_path)


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
    root_path = _config_copy(tmp_path)
    root_path.write_text(
        root_path.read_text(encoding="utf-8") + "\nschema_version: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="duplicate key 'schema_version'"):
        load_project_config(root_path)


def test_instrument_must_be_xau_usd(tmp_path: Path) -> None:
    root_path = _config_copy(tmp_path)
    instrument_path = root_path.parent / "instrument.yaml"
    _change_yaml(
        instrument_path,
        lambda config: config["instrument"].update({"id": "EUR_USD"}),
    )

    with pytest.raises(ConfigError, match="XAU_USD"):
        load_project_config(root_path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda config: config["data"].update({"raw_timeframe": "1m"}),
            "unambiguous code",
        ),
        (
            lambda config: config["data"].update({"derived_timeframes": ["3min"]}),
            "derived timeframes must include 15min",
        ),
        (
            lambda config: config["data"].update(
                {"derived_timeframes": ["3min", "15min", "15min"]}
            ),
            "derived timeframes must be unique",
        ),
    ],
)
def test_timeframes_are_unambiguous_and_cover_the_mvp(
    tmp_path: Path,
    change: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    root_path = _config_copy(tmp_path)
    _change_yaml(root_path.parent / "instrument.yaml", change)

    with pytest.raises(ConfigError, match=message):
        load_project_config(root_path)


def test_primary_horizon_must_remain_15_minutes(tmp_path: Path) -> None:
    root_path = _config_copy(tmp_path)
    _change_yaml(
        root_path.parent / "labels.yaml",
        lambda config: config["prediction"].update({"primary_horizon_minutes": 30}),
    )

    with pytest.raises(ConfigError, match="primary_horizon_minutes"):
        load_project_config(root_path)


def test_label_threshold_must_equal_cost_and_noise_buffers(tmp_path: Path) -> None:
    root_path = _config_copy(tmp_path)

    def change(config: dict[str, Any]) -> None:
        config["classes"]["total_threshold_bps"] = 6.5
        _threshold_rules(config["classes"])

    _change_yaml(root_path.parent / "labels.yaml", change)

    with pytest.raises(ConfigError, match="must equal estimated_cost_buffer_bps"):
        load_project_config(root_path)


def test_label_cost_buffer_must_match_cost_model(tmp_path: Path) -> None:
    root_path = _config_copy(tmp_path)

    def change(config: dict[str, Any]) -> None:
        config["classes"]["estimated_cost_buffer_bps"] = 3.0
        config["classes"]["total_threshold_bps"] = 5.0
        _threshold_rules(config["classes"])

    _change_yaml(root_path.parent / "labels.yaml", change)

    with pytest.raises(ConfigError, match="label estimated_cost_buffer_bps"):
        load_project_config(root_path)


def test_base_cost_total_must_equal_its_components(tmp_path: Path) -> None:
    root_path = _config_copy(tmp_path)
    _change_yaml(
        root_path.parent / "costs.yaml",
        lambda config: config["cost_model"].update({"total_round_trip_bps": 4.2}),
    )

    with pytest.raises(ConfigError, match=r"spread \+ two-sided slippage"):
        load_project_config(root_path)


def test_stress_cost_total_must_equal_its_components(tmp_path: Path) -> None:
    root_path = _config_copy(tmp_path)
    _change_yaml(
        root_path.parent / "costs.yaml",
        lambda config: config["stress_model"].update({"total_round_trip_bps": 6.0}),
    )

    with pytest.raises(ConfigError, match="stressed spread"):
        load_project_config(root_path)


def test_splits_must_be_chronological_and_non_overlapping(tmp_path: Path) -> None:
    root_path = _config_copy(tmp_path)
    _change_yaml(
        root_path.parent / "splits_mvp.yaml",
        lambda config: config["splits"]["validation"].update({"start": "2022-12-01T00:00:00Z"}),
    )

    with pytest.raises(ConfigError, match="train must end no later than validation starts"):
        load_project_config(root_path)


def test_purging_gap_covers_latency_plus_horizon(tmp_path: Path) -> None:
    root_path = _config_copy(tmp_path)
    _change_yaml(
        root_path.parent / "splits_mvp.yaml",
        lambda config: config["purging"].update({"gap_minutes": 15}),
    )

    with pytest.raises(ConfigError, match="purging gap must be at least 16 minutes"):
        load_project_config(root_path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda config: config["development_guard"].update(
                {"reject_at_or_after": "2024-12-01T00:00:00Z"}
            ),
            "test must end no later",
        ),
        (
            lambda config: config["development_guard"].update({"allow_holdout_override": True}),
            "allow_holdout_override",
        ),
        (
            lambda config: config["preprocessing"].update({"random_shuffle": True}),
            "random_shuffle",
        ),
    ],
)
def test_holdout_and_chronological_development_guards_cannot_be_disabled(
    tmp_path: Path,
    change: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    root_path = _config_copy(tmp_path)
    _change_yaml(root_path.parent / "splits_mvp.yaml", change)

    with pytest.raises(ConfigError, match=message):
        load_project_config(root_path)
