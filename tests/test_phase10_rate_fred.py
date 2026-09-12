"""DFII10 source authentication, modeled release timing and bundle integrity."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gold_forecasting.phase10.bundle import load_context_data
from gold_forecasting.phase10.contracts import ContextSource, load_source, require_strict_pit
from gold_forecasting.phase10.rate_fred import (
    FRED_RATE_FILENAME,
    RateFredError,
    import_fred_dfii10,
    parse_fred_dfii10_csv,
)


@pytest.fixture
def rate_source() -> ContextSource:
    return load_source("configs/phase10_rate_exploratory.yaml")


def _rows() -> list[tuple[str, str]]:
    return [
        ("2020-01-01", "."),
        ("2020-01-02", "0.10"),
        ("2020-07-02", "0.20"),
        ("2021-01-04", "0.30"),
        ("2022-06-17", "0.40"),
        ("2023-03-10", "0.50"),
        ("2024-12-31", "0.60"),
    ]


def _csv(path: Path, rows: list[tuple[str, str]] | None = None) -> Path:
    path.write_text(
        "observation_date,DFII10\n"
        + "\n".join(f"{stamp},{value}" for stamp, value in (rows or _rows()))
        + "\n",
        encoding="utf-8",
    )
    return path


def test_dfii10_models_next_h15_business_release_with_dst_and_holidays(
    tmp_path: Path, rate_source: ContextSource
) -> None:
    path = _csv(tmp_path / FRED_RATE_FILENAME)
    observations, metadata = parse_fred_dfii10_csv(
        path,
        source=rate_source,
        ingested_at_utc=pd.Timestamp("2026-09-10T00:00:00Z"),
    )
    assert observations.value.tolist() == [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    assert observations.observed_at_utc.tolist()[:2] == [
        pd.Timestamp("2020-01-02T00:00:00Z"),
        pd.Timestamp("2020-07-02T00:00:00Z"),
    ]
    assert observations.available_at_utc.tolist() == [
        pd.Timestamp("2020-01-03T21:15:00Z"),
        pd.Timestamp("2020-07-06T20:15:00Z"),
        pd.Timestamp("2021-01-05T21:15:00Z"),
        pd.Timestamp("2022-06-21T20:15:00Z"),
        pd.Timestamp("2023-03-13T20:15:00Z"),
        pd.Timestamp("2025-01-02T21:15:00Z"),
    ]
    assert metadata["missing_rows_skipped"] == 1
    assert metadata["availability_is_historical_evidence"] is False
    assert metadata["revision_snapshot"] == "latest_downloaded_history_no_vintages"
    assert metadata["year_rows"] == {
        "2020": 2,
        "2021": 1,
        "2022": 1,
        "2023": 1,
        "2024": 1,
    }


def test_rate_source_contract_cannot_claim_strict_pit(
    tmp_path: Path, rate_source: ContextSource
) -> None:
    path = _csv(tmp_path / FRED_RATE_FILENAME)
    strict = rate_source.model_copy(update={"availability_basis": "provider_timestamp"})
    with pytest.raises(RateFredError, match="availability_basis"):
        parse_fred_dfii10_csv(path, source=strict)
    with pytest.raises(ValueError, match="provider release evidence"):
        require_strict_pit(rate_source)


@pytest.mark.parametrize(
    "rows",
    [
        [
            ("2019-12-31", "1"),
            ("2020-01-02", "1"),
            ("2021-01-04", "1"),
            ("2022-01-03", "1"),
            ("2023-01-03", "1"),
            ("2024-01-02", "1"),
        ],
        [
            ("2020-01-02", "nan"),
            ("2021-01-04", "1"),
            ("2022-01-03", "1"),
            ("2023-01-03", "1"),
            ("2024-01-02", "1"),
        ],
        [
            ("2020-01-02", "30"),
            ("2021-01-04", "1"),
            ("2022-01-03", "1"),
            ("2023-01-03", "1"),
            ("2024-01-02", "1"),
        ],
    ],
)
def test_invalid_rate_history_fails_closed(
    tmp_path: Path, rate_source: ContextSource, rows: list[tuple[str, str]]
) -> None:
    with pytest.raises(RateFredError):
        parse_fred_dfii10_csv(_csv(tmp_path / FRED_RATE_FILENAME, rows), source=rate_source)


def test_dfii10_import_partitions_and_authenticates_snapshot(
    tmp_path: Path, rate_source: ContextSource
) -> None:
    source_file = _csv(tmp_path / FRED_RATE_FILENAME)
    output = tmp_path / "rate"
    result = import_fred_dfii10(
        source_file,
        output,
        rate_source,
        ingested_at_utc=pd.Timestamp("2026-09-10T00:00:00Z"),
    )
    assert result["source_id"] == "rate"
    assert result["series_id"] == "DFII10"
    assert result["strict_pit"] is False
    assert result["rows"] == 6
    bundle = json.loads((output / "rate.bundle-set.json").read_text())
    assert bundle["bundles"] == [f"rate-{year}.manifest.json" for year in range(2020, 2025)]
    loaded = load_context_data(output / "rate.bundle-set.json", rate_source)
    assert len(loaded) == 6
    assert loaded.raw_sha256.nunique() == 1
    artifacts = json.loads((output / "source_artifacts.json").read_text())
    assert artifacts["source_file"]["sha256"] == loaded.raw_sha256.iloc[0]
    parquet = output / "rate-2020.parquet"
    parquet.write_bytes(parquet.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        load_context_data(output / "rate.bundle-set.json", rate_source)


def test_dfii10_filename_and_ingestion_time_are_authenticated(
    tmp_path: Path, rate_source: ContextSource
) -> None:
    wrong = _csv(tmp_path / "renamed.csv")
    with pytest.raises(RateFredError, match="named exactly"):
        parse_fred_dfii10_csv(wrong, source=rate_source)
    path = _csv(tmp_path / FRED_RATE_FILENAME)
    with pytest.raises(RateFredError, match="ingested_at_utc precedes"):
        parse_fred_dfii10_csv(
            path,
            source=rate_source,
            ingested_at_utc=pd.Timestamp("2020-01-01T00:00:00Z"),
        )
