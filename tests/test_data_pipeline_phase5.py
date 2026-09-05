from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
import pytest
import yaml

from gold_forecasting.artifacts import DatasetManifest, sha256_file
from gold_forecasting.config import ConfigError, load_project_config
from gold_forecasting.data_pipeline import (
    DataBuildError,
    build_mvp_data,
    validate_existing_mvp_data,
)
from gold_forecasting.ingestion import HistDataM1Provider
from tests.test_data_pipeline import _isolated_config, _provider


def test_hardened_build_reproduces_and_keeps_mvp_data(tmp_path: Path) -> None:
    mvp = _isolated_config(tmp_path)
    provider, requests = _provider()
    baseline = build_mvp_data(mvp, provider=provider, years=(2024,))
    baseline_hashes = {
        str(path): sha256_file(path)
        for path in (
            baseline.raw_manifest_path,
            *baseline.curated_manifest_paths.values(),
        )
    }
    config = mvp.with_name("phase5.yaml")
    first = build_mvp_data(config, provider=provider, years=(2024,))
    first_quality = sha256_file(first.quality_report_path)
    second = build_mvp_data(config, provider=provider, years=(2024,))
    assert first.row_counts == {
        "1min": 20,
        "3min": 6,
        "5min": 4,
        "15min": 1,
        "30min": 0,
        "1h": 0,
        "3h": 0,
        "1d": 0,
        "1mo": 0,
    }
    assert first.dataset_version == second.dataset_version
    assert first_quality == sha256_file(second.quality_report_path)
    assert len(requests) == 2
    assert validate_existing_mvp_data(config, allow_partial=True) == first.row_counts
    assert validate_existing_mvp_data(mvp, allow_partial=True) == baseline.row_counts
    assert baseline_hashes == {path: sha256_file(path) for path in baseline_hashes}
    for timeframe, path in first.curated_manifest_paths.items():
        manifest = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
        assert manifest.parameters["partition_basis"] == "utc_open_year"
        assert manifest.parameters["resampling_logic_version"]
        assert manifest.row_count == first.row_counts[timeframe]
    daily = pd.read_parquet(tmp_path / "reports/data/phase5_quality_daily.parquet")
    assert set(daily["timeframe"]) == set(first.row_counts)
    assert daily.groupby("timeframe").size().eq(366).all()
    assert daily.loc[daily["timeframe"].eq("1mo"), "observed_count"].sum() == 0
    assert daily.loc[daily["timeframe"].eq("1mo"), "unknown_gap_count"].sum() == 12
    with pytest.raises(DataBuildError, match="partial"):
        validate_existing_mvp_data(config)

    completion = first.curated_manifest_paths["1min"].parent.parent / "phase5_build.json"
    payload = json.loads(completion.read_text(encoding="utf-8"))
    payload["curated_dataset_versions"]["1mo"] = "sha256:" + "0" * 64
    completion.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataBuildError, match="stale"):
        validate_existing_mvp_data(config, allow_partial=True)
    payload["curated_dataset_versions"]["1mo"] = DatasetManifest.model_validate_json(
        first.curated_manifest_paths["1mo"].read_text(encoding="utf-8")
    ).dataset_version
    payload["reports"] = payload["reports"][4:5]
    completion.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataBuildError, match="required quality report set"):
        validate_existing_mvp_data(config, allow_partial=True)

    instrument_path = config.parent / "instrument_phase5.yaml"
    instrument = yaml.safe_load(instrument_path.read_text(encoding="utf-8"))
    instrument["provider"]["market_hours_policy"] = "unsupported_calendar"
    instrument_path.write_text(yaml.safe_dump(instrument), encoding="utf-8")
    with pytest.raises(DataBuildError, match="observed_source_rows policy"):
        validate_existing_mvp_data(config, allow_partial=True)


def test_hardened_quality_corruption_invalidates_build(tmp_path: Path) -> None:
    mvp = _isolated_config(tmp_path)
    provider, _ = _provider()
    config = mvp.with_name("phase5.yaml")
    result = build_mvp_data(config, provider=provider, years=(2024,))
    assert result.quality_report_path is not None
    result.quality_report_path.write_text("corrupted report", encoding="utf-8")
    with pytest.raises(DataBuildError, match="differs from manifest"):
        validate_existing_mvp_data(config, allow_partial=True)


def test_empty_year_selection_is_rejected_before_acquisition(tmp_path: Path) -> None:
    config = _isolated_config(tmp_path)
    provider, requests = _provider()
    with pytest.raises(DataBuildError, match="at least one"):
        build_mvp_data(config, provider=provider, years=())
    assert not requests


def test_unsupported_timeframe_fails_before_build(tmp_path: Path) -> None:
    config = _isolated_config(tmp_path)
    instrument = config.parent / "instrument.yaml"
    payload = yaml.safe_load(instrument.read_text(encoding="utf-8"))
    payload["data"]["derived_timeframes"].append("2mo")
    instrument.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="unsupported derived timeframe"):
        load_project_config(config)


def test_partial_source_year_spillover_uses_one_utc_scope(tmp_path: Path) -> None:
    config = _isolated_config(tmp_path).with_name("phase5.yaml")
    start = datetime(2020, 12, 31, 18, 45)
    rows = [
        f"{(start + timedelta(minutes=i)):%Y%m%d %H%M%S};1900;1901;1899;1900.5;0" for i in range(20)
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("DAT_ASCII_XAUUSD_M1_2020.csv", "\n".join(rows) + "\n")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, content=buffer.getvalue())
        values = {
            "tk": "0123456789abcdef",
            "date": "2020",
            "datemonth": "2020",
            "platform": "ASCII",
            "timeframe": "M1",
            "fxpair": "XAUUSD",
        }
        html = '<form id="file_down" method="POST" action="/get.php">'
        html += "".join(
            f'<input type="hidden" name="{key}" value="{value}">' for key, value in values.items()
        )
        return httpx.Response(200, text=html + "</form>")

    provider = HistDataM1Provider(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=lambda: datetime(2026, 9, 5, tzinfo=UTC),
    )
    result = build_mvp_data(config, provider=provider, years=(2020,))
    assert result.row_counts["1min"] == 15  # five 2021 UTC minutes remain only in immutable raw
    assert validate_existing_mvp_data(config, allow_partial=True) == result.row_counts
    manifest = DatasetManifest.model_validate_json(
        result.raw_manifest_path.read_text(encoding="utf-8")
    )
    assert manifest.row_count == 15
    assert manifest.period_end_utc == datetime(2021, 1, 1, tzinfo=UTC)
    daily = pd.read_parquet(tmp_path / "reports/data/phase5_quality_daily.parquet")
    assert daily["day_utc"].max() == "2020-12-31"
    assert daily.loc[daily["timeframe"].eq("1min"), "row_count"].sum() == 15
