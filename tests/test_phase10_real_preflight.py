"""Real source authentication, metadata-first guards and gold-universe preservation."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gold_forecasting.artifacts import sha256_file
from gold_forecasting.phase10 import real_preflight
from gold_forecasting.phase10.config import Phase10Config, load_phase10_config
from gold_forecasting.phase10.contracts import ContextSource, load_source
from gold_forecasting.phase10.real_preflight import (
    SILVER_MODEL_FEATURES,
    Phase10PreflightError,
    augment_silver_table,
    inspect_silver_metadata,
    load_verified_silver,
    silver_coverage,
    validate_modeled_silver_source,
)
from gold_forecasting.phase10.silver_histdata import import_histdata_silver_archives


@pytest.fixture
def silver_source() -> ContextSource:
    return load_source("configs/phase10_silver_exploratory.yaml")


@pytest.fixture
def real_bundle(
    tmp_path: Path,
    silver_source: ContextSource,
) -> tuple[Path, Phase10Config, ContextSource]:
    config = Phase10Config()
    archives = tmp_path / config.archive_directory
    archives.mkdir(parents=True)
    for year in range(2020, 2025):
        filename = archives / f"HISTDATA_COM_ASCII_XAGUSD_M1_{year}.zip"
        rows = []
        for index, stamp in enumerate(
            pd.date_range(f"{year}-01-02T12:00", periods=150, freq="min")
        ):
            close = 18 + index / 100 + 0.005 * np.sin(index / 3)
            rows.append(
                f"{stamp:%Y%m%d %H%M%S};{close:.6f};{close + 0.1:.6f};"
                f"{close - 0.1:.6f};{close:.6f};0"
            )
        with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            zipped.writestr(f"DAT_ASCII_XAGUSD_M1_{year}.csv", "\n".join(rows))
    import_histdata_silver_archives(
        archives,
        (tmp_path / config.bundle_path).parent,
        silver_source,
        ingested_at_utc=pd.Timestamp("2026-09-09T00:00:00Z"),
    )
    return tmp_path, config, silver_source


def _change_json(path: Path, **updates: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _gold() -> tuple[pd.DataFrame, pd.DataFrame]:
    times = pd.date_range("2020-01-02T17:03Z", periods=45, freq="3min")
    table = pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(len(times))],
            "instrument": "XAU_USD",
            "source": "histdata",
            "prediction_time_utc": times,
            "source_candle_open_utc": times - pd.Timedelta(minutes=3),
            "gold_feature": np.arange(len(times), dtype=float),
        }
    )
    candles = pd.DataFrame(
        {
            "instrument": "XAU_USD",
            "source": "histdata",
            "timestamp_open_utc": times - pd.Timedelta(minutes=3),
            "timestamp_close_utc": times,
            "bid_close": 1500 + np.arange(len(times)) + np.sin(np.arange(len(times))),
        }
    )
    return table, candles


def test_checked_in_phase10_configuration_is_frozen() -> None:
    config = load_phase10_config("configs/phase10_silver_ablation.yaml")
    assert config == Phase10Config()
    assert config.phase7_reference_completion.endswith("b45053c19487474c87d0ceca0f") is False
    assert config.phase7_reference_completion == (
        "sha256:beac58092d06350bb067fbd2df144bb9cb2507f45053c19487474c87d0ceca0f"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("protocol_version", "phase10-context-v1"),
        ("test_years", [2022, 2023, 2025]),
        ("test_years", [2023, 2024]),
        ("horizon_minutes", 3),
        ("gap_minutes", 180),
        ("seed", 42),
        ("minimum_policy_trades", 1),
        ("phase7_reference_run", "other"),
        ("phase7_reference_completion", "sha256:" + "0" * 64),
        ("source_config", "../source.yaml"),
        ("bundle_path", "C:/outside.json"),
        ("archive_directory", "data/../holdout"),
        ("output_directory", "data/runs"),
    ],
)
def test_config_rejects_protocol_drift(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        Phase10Config.model_validate({field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("publication_delay_seconds", 0),
        ("publication_delay_seconds", 120),
        ("stale_after_seconds", 1200),
        ("enabled", False),
        ("availability_basis", "provider_timestamp"),
        ("revision_policy", "vintages"),
        ("source_timezone", "America/New_York"),
        ("source_url", "https://example.com/"),
    ],
)
def test_source_contract_cannot_relax_frozen_latency(
    silver_source: ContextSource,
    field: str,
    value: object,
) -> None:
    source = ContextSource.model_validate({**silver_source.model_dump(), field: value})
    with pytest.raises(Phase10PreflightError, match="frozen modeled"):
        validate_modeled_silver_source(source)


def test_real_bundle_roundtrip_authenticates_every_archive_and_partition(
    real_bundle: tuple[Path, Phase10Config, ContextSource],
) -> None:
    root, config, source = real_bundle
    observations, evidence = load_verified_silver(root, config, source)
    assert len(observations) == 750
    assert len(evidence["bundles"]) == 5
    assert evidence["bundle_set_sha256"] == sha256_file(root / config.bundle_path)
    assert observations["raw_sha256"].nunique() == 5
    assert (
        observations["available_at_utc"]
        .sub(observations["observed_at_utc"])
        .eq(pd.Timedelta(seconds=60))
        .all()
    )


@pytest.mark.parametrize("attack", ["last_partition", "source_year", "bundle_year"])
def test_all_metadata_guarded_before_any_observation_or_archive_open(
    real_bundle: tuple[Path, Phase10Config, ContextSource],
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    root, config, source = real_bundle
    directory = (root / config.bundle_path).parent
    if attack == "last_partition":
        _change_json(
            directory / "silver-2024.manifest.json", observation_end_utc="2026-01-01T00:00:00Z"
        )
    elif attack == "source_year":
        path = directory / "source_archives.json"
        payload = json.loads(path.read_text())
        payload["archives"][-1]["year"] = 2025
        path.write_text(json.dumps(payload))
    else:
        _change_json(root / config.bundle_path, bundles=["silver-2025.manifest.json"])

    def forbid(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("metadata guard must run before all market payload access")

    monkeypatch.setattr(real_preflight, "parse_histdata_xagusd_archive", forbid)
    monkeypatch.setattr(real_preflight, "load_context_bundle", forbid)
    with pytest.raises(ValueError, match=r"2020-2024|source/year"):
        load_verified_silver(root, config, source)


def test_archive_modification_is_detected(
    real_bundle: tuple[Path, Phase10Config, ContextSource],
) -> None:
    root, config, source = real_bundle
    archive = root / config.archive_directory / "HISTDATA_COM_ASCII_XAGUSD_M1_2020.zip"
    with zipfile.ZipFile(archive, "a") as zipped:
        zipped.writestr("extra.txt", "changed archive bytes")
    with pytest.raises(Phase10PreflightError, match="hash/metadata"):
        load_verified_silver(root, config, source)


def test_rehashed_forged_silver_observation_cannot_masquerade_as_archive(
    real_bundle: tuple[Path, Phase10Config, ContextSource],
) -> None:
    root, config, source = real_bundle
    directory = (root / config.bundle_path).parent
    parquet = directory / "silver-2020.parquet"
    frame = pd.read_parquet(parquet)
    frame.loc[0, "value"] = 99.0
    frame.to_parquet(parquet, index=False)
    _change_json(directory / "silver-2020.manifest.json", sha256=sha256_file(parquet))
    with pytest.raises(Phase10PreflightError, match="differ from authenticated"):
        load_verified_silver(root, config, source)


def test_metadata_change_between_guard_and_read_is_detected(
    real_bundle: tuple[Path, Phase10Config, ContextSource],
) -> None:
    root, config, source = real_bundle
    metadata = inspect_silver_metadata(root, config, source)
    path = root / config.bundle_path
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(Phase10PreflightError, match="changed after"):
        load_verified_silver(root, config, source, metadata=metadata)


def test_missing_archives_fail_before_parquet_access(
    real_bundle: tuple[Path, Phase10Config, ContextSource],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config, source = real_bundle
    archive = root / config.archive_directory / "HISTDATA_COM_ASCII_XAGUSD_M1_2024.zip"
    archive.unlink()
    monkeypatch.setattr(
        real_preflight, "load_context_bundle", lambda *args, **kwargs: pytest.fail()
    )
    with pytest.raises(Phase10PreflightError, match="required local XAGUSD archive is missing"):
        load_verified_silver(root, config, source)


def test_silver_features_preserve_exact_gold_rows_and_route_history_to_fallback(
    real_bundle: tuple[Path, Phase10Config, ContextSource],
) -> None:
    root, config, source = real_bundle
    observations, _ = load_verified_silver(root, config, source)
    table, gold = _gold()
    result = augment_silver_table(table, gold, observations, source)
    pd.testing.assert_frame_equal(result.loc[:, table.columns], table)
    assert not result["silver_usable"].iloc[:20].any()
    assert result["silver_usable"].iloc[20:].all()
    assert not set(("silver_is_missing", "silver_is_stale")) & set(SILVER_MODEL_FEATURES)
    assert result["silver_available_at_utc"].le(result["prediction_time_utc"]).all()
    assert result["silver_age_seconds"].eq(60).all()
    coverage = silver_coverage(result)
    assert coverage["fallback_rows"] == 20
    assert coverage["insufficient_history_rows"] == 20
    assert coverage["age_quantiles_seconds"]["0.95"] == 60.0


def test_future_silver_mutation_does_not_rewrite_earlier_features(
    real_bundle: tuple[Path, Phase10Config, ContextSource],
) -> None:
    root, config, source = real_bundle
    observations, _ = load_verified_silver(root, config, source)
    table, gold = _gold()
    original = augment_silver_table(table, gold, observations, source)
    cutoff = table["prediction_time_utc"].iloc[30]
    changed = observations.copy()
    changed.loc[changed["available_at_utc"].gt(cutoff), "value"] *= 7
    mutated = augment_silver_table(table, gold, changed, source)
    pd.testing.assert_frame_equal(original.iloc[:31], mutated.iloc[:31])


def test_missing_or_stale_or_repeated_silver_never_reduces_gold_universe(
    real_bundle: tuple[Path, Phase10Config, ContextSource],
) -> None:
    root, config, source = real_bundle
    observations, _ = load_verified_silver(root, config, source)
    table, gold = _gold()
    result = augment_silver_table(table, gold, observations.iloc[:1].copy(), source)
    assert len(result) == len(table)
    assert not result["silver_usable"].any()
    assert result["silver_is_stale"].any()
    assert result["silver_return_1_bps"].isna().all()
    missing = augment_silver_table(table, gold, observations.iloc[:0].copy(), source)
    assert missing["silver_is_missing"].all()
    assert not missing["silver_usable"].any()


@pytest.mark.parametrize("future", [True, False])
def test_gold_close_requires_exact_closed_source_candle(
    real_bundle: tuple[Path, Phase10Config, ContextSource],
    future: bool,
) -> None:
    root, config, source = real_bundle
    observations, _ = load_verified_silver(root, config, source)
    table, gold = _gold()
    if future:
        gold.loc[0, "timestamp_close_utc"] += pd.Timedelta(minutes=1)
    else:
        gold = gold.iloc[1:]
    with pytest.raises(Phase10PreflightError, match="exact closed source candle"):
        augment_silver_table(table, gold, observations, source)
