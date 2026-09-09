"""Local XAGUSD HistData ingestion stays separate from frozen XAUUSD."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from gold_forecasting.cli import app
from gold_forecasting.phase10.bundle import load_context_data
from gold_forecasting.phase10.contracts import ContextSource
from gold_forecasting.phase10.preflight import inspect_context_source
from gold_forecasting.phase10.silver_histdata import (
    SilverHistDataError,
    import_histdata_silver_archives,
    parse_histdata_xagusd_archive,
)


def _source(**updates: object) -> ContextSource:
    payload: dict[str, object] = {
        "source_id": "silver",
        "description": "HistData XAGUSD modeled-latency research source",
        "source_url": "https://www.histdata.com/f-a-q/data-files-detailed-specification/",
        "market_hours": "OTC quote stream",
        "source_timezone": "Fixed UTC-05:00 without daylight saving time",
        "publication_delay_seconds": 60,
        "stale_after_seconds": 600,
        "availability_basis": "modeled_latency",
        "revision_policy": "append_only",
        "availability_evidence": "Candle close plus modeled 60 second delay",
        "enabled": True,
        **updates,
    }
    return ContextSource.model_validate(payload)


def _zip(
    path: Path,
    *,
    year: int = 2020,
    symbol: str = "XAGUSD",
    rows: list[str] | None = None,
) -> Path:
    lines = rows or [
        "20200102 120000;18.00;18.10;17.90;18.05;0",
        "20200102 120100;18.05;18.20;18.00;18.15;0",
    ]
    member = f"DAT_ASCII_{symbol}_M1_{year}.csv"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, "\n".join(lines) + "\n")
    return path


def test_parse_xagusd_archive_uses_fixed_minus_five_and_modeled_latency(
    tmp_path: Path,
) -> None:
    path = _zip(tmp_path / "silver.zip")
    observations, metadata = parse_histdata_xagusd_archive(
        path,
        year=2020,
        source=_source(),
        ingested_at_utc=pd.Timestamp("2026-09-09T00:00:00Z"),
    )

    assert observations["value"].tolist() == [18.05, 18.15]
    assert observations["observed_at_utc"].tolist() == [
        pd.Timestamp("2020-01-02T17:01:00Z"),
        pd.Timestamp("2020-01-02T17:02:00Z"),
    ]
    assert observations["available_at_utc"].tolist() == [
        pd.Timestamp("2020-01-02T17:02:00Z"),
        pd.Timestamp("2020-01-02T17:03:00Z"),
    ]
    assert metadata["source_symbol"] == "XAGUSD"
    assert metadata["availability_basis"] == "modeled_latency"
    assert metadata["timestamp_semantics"] == "candle_open"
    assert metadata["observation_semantics"] == "candle_close"


def test_xauusd_archive_cannot_masquerade_as_silver(tmp_path: Path) -> None:
    path = _zip(tmp_path / "wrong.zip", symbol="XAUUSD")
    with pytest.raises(SilverHistDataError, match="XAGUSD"):
        parse_histdata_xagusd_archive(
            path,
            year=2020,
            source=_source(),
            ingested_at_utc=pd.Timestamp("2026-09-09T00:00:00Z"),
        )


def test_adapter_rejects_false_provider_timestamp_claim(tmp_path: Path) -> None:
    path = _zip(tmp_path / "silver.zip")
    with pytest.raises(SilverHistDataError, match="modeled_latency"):
        parse_histdata_xagusd_archive(
            path,
            year=2020,
            source=_source(availability_basis="provider_timestamp"),
            ingested_at_utc=pd.Timestamp("2026-09-09T00:00:00Z"),
        )


@pytest.mark.parametrize(
    "rows",
    [
        ["20200102 120000;18.00;17.90;17.80;18.05;0"],
        ["20200102 120000;18.00;18.10;18.06;18.05;0"],
        ["20200102 120000;0;18.10;17.90;18.05;0"],
    ],
)
def test_invalid_ohlc_fails_closed(tmp_path: Path, rows: list[str]) -> None:
    path = _zip(tmp_path / "invalid.zip", rows=rows)
    with pytest.raises(SilverHistDataError, match="OHLC"):
        parse_histdata_xagusd_archive(
            path,
            year=2020,
            source=_source(),
            ingested_at_utc=pd.Timestamp("2026-09-09T00:00:00Z"),
        )


def test_conflicting_duplicate_timestamp_fails_closed(tmp_path: Path) -> None:
    path = _zip(
        tmp_path / "conflict.zip",
        rows=[
            "20200102 120000;18.00;18.10;17.90;18.05;0",
            "20200102 120000;18.00;18.20;17.90;18.15;0",
        ],
    )
    with pytest.raises(SilverHistDataError, match="conflicting candles"):
        parse_histdata_xagusd_archive(
            path,
            year=2020,
            source=_source(),
            ingested_at_utc=pd.Timestamp("2026-09-09T00:00:00Z"),
        )


def test_source_year_is_checked_in_fixed_provider_timezone(tmp_path: Path) -> None:
    path = _zip(
        tmp_path / "wrong-year.zip",
        year=2020,
        rows=["20191231 235900;18.00;18.10;17.90;18.05;0"],
    )
    with pytest.raises(SilverHistDataError, match="source year"):
        parse_histdata_xagusd_archive(
            path,
            year=2020,
            source=_source(),
            ingested_at_utc=pd.Timestamp("2026-09-09T00:00:00Z"),
        )


def test_import_builds_partitioned_bundle_set(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _zip(raw / "HISTDATA_COM_ASCII_XAGUSD_M1_2020.zip")
    output = tmp_path / "context"

    summary = import_histdata_silver_archives(
        raw,
        output,
        _source(),
        years=(2020,),
        ingested_at_utc=pd.Timestamp("2026-09-09T00:00:00Z"),
    )
    loaded = load_context_data(output / "silver.bundle-set.json", _source())

    assert summary["years"] == [2020]
    assert summary["rows"] == 2
    assert summary["strict_pit"] is False
    assert loaded["value"].tolist() == [18.05, 18.15]
    assert (output / "silver-2020.parquet").is_file()
    assert (output / "silver-2020.manifest.json").is_file()
    assert (output / "source_archives.json").is_file()


def test_import_refuses_overwrite(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _zip(raw / "HISTDATA_COM_ASCII_XAGUSD_M1_2020.zip")
    output = tmp_path / "context"
    output.mkdir()

    with pytest.raises(FileExistsError):
        import_histdata_silver_archives(
            raw,
            output,
            _source(),
            years=(2020,),
            ingested_at_utc=pd.Timestamp("2026-09-09T00:00:00Z"),
        )


def test_imported_modeled_bundle_is_exploratory_not_strict_pit(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _zip(raw / "HISTDATA_COM_ASCII_XAGUSD_M1_2020.zip")
    output = tmp_path / "context"
    source_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "phase10_silver_exploratory.yaml"
    )

    import_histdata_silver_archives(
        raw,
        output,
        _source(),
        years=(2020,),
        ingested_at_utc=pd.Timestamp("2026-09-09T00:00:00Z"),
    )
    result = inspect_context_source(
        source_path,
        output / "silver.bundle-set.json",
    )

    assert result["status"] == "ready_for_exploratory_ablation"
    assert result["exploratory_ablation_ready"] is True
    assert result["strict_pit_source_ready"] is False
    assert result["formal_benchmark_ready"] is False
    assert result["formal_blockers"] == ["historical_release_evidence_unproven"]
    assert result["observation_rows"] == 2


def test_cli_imports_local_xagusd_archive(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _zip(raw / "HISTDATA_COM_ASCII_XAGUSD_M1_2020.zip")
    output = tmp_path / "context"
    source_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "phase10_silver_exploratory.yaml"
    )

    result = CliRunner().invoke(
        app,
        [
            "phase10",
            "silver-import",
            "--archive-directory",
            str(raw),
            "--output",
            str(output),
            "--source",
            str(source_path),
            "--years",
            "2020",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output / "silver.bundle-set.json").is_file()
    assert (output / "silver-2020.parquet").is_file()
