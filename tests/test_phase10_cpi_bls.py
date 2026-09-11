"""BLS CPI source authentication, release timing and bundle integrity."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gold_forecasting.phase10.bundle import load_context_data
from gold_forecasting.phase10.contracts import ContextSource, load_source, require_strict_pit
from gold_forecasting.phase10.cpi_bls import (
    BLS_CPI_FILENAME,
    BLS_CPI_SERIES,
    CPI_RELEASE_DATES,
    CpiBlsError,
    import_bls_cpi,
    parse_bls_cpi_file,
)


@pytest.fixture
def cpi_source() -> ContextSource:
    return load_source("configs/phase10_cpi_exploratory.yaml")


def _rows() -> list[tuple[str, int, str, str, str]]:
    rows: list[tuple[str, int, str, str, str]] = []
    index = 0
    for year in range(2020, 2025):
        for month in range(1, 13):
            rows.append((BLS_CPI_SERIES, year, f"M{month:02d}", f"{250 + 0.4 * index:.3f}", ""))
            index += 1
    return rows


def _flat(
    path: Path, rows: list[tuple[str, int, str, str, str]] | None = None
) -> Path:
    payload = ["series_id\tyear\tperiod\tvalue\tfootnote_codes"]
    payload.extend("\t".join(map(str, row)) for row in (rows or _rows()))
    path.write_text("\n".join(payload) + "\n", encoding="utf-8")
    return path


def test_cpi_models_official_release_dates_with_dst(
    tmp_path: Path, cpi_source: ContextSource
) -> None:
    observations, metadata = parse_bls_cpi_file(
        _flat(tmp_path / BLS_CPI_FILENAME),
        source=cpi_source,
        ingested_at_utc=pd.Timestamp("2026-09-11T00:00:00Z"),
    )
    assert len(observations) == 59
    assert observations.observation_id.iloc[0] == "cpi-u-2020-01"
    assert observations.available_at_utc.iloc[0] == pd.Timestamp("2020-02-13T13:30:00Z")
    june_2020 = observations.loc[observations.observation_id.eq("cpi-u-2020-06")].iloc[0]
    assert june_2020.available_at_utc == pd.Timestamp("2020-07-14T12:30:00Z")
    assert observations.observation_id.iloc[-1] == "cpi-u-2024-11"
    assert observations.available_at_utc.iloc[-1] == pd.Timestamp("2024-12-11T13:30:00Z")
    assert "2024-12" not in CPI_RELEASE_DATES
    assert metadata["excluded_after_development_release"] == 1
    assert metadata["availability_is_historical_evidence"] is False
    assert metadata["year_rows"] == {
        "2020": 12,
        "2021": 12,
        "2022": 12,
        "2023": 12,
        "2024": 11,
    }


def test_cpi_source_contract_cannot_claim_strict_pit(
    tmp_path: Path, cpi_source: ContextSource
) -> None:
    path = _flat(tmp_path / BLS_CPI_FILENAME)
    strict = cpi_source.model_copy(update={"availability_basis": "provider_timestamp"})
    with pytest.raises(CpiBlsError, match="availability_basis"):
        parse_bls_cpi_file(path, source=strict)
    with pytest.raises(ValueError, match="provider release evidence"):
        require_strict_pit(cpi_source)


def test_cpi_requires_complete_development_months_and_numeric_values(
    tmp_path: Path, cpi_source: ContextSource
) -> None:
    with pytest.raises(CpiBlsError, match="exactly 60"):
        parse_bls_cpi_file(
            _flat(tmp_path / BLS_CPI_FILENAME, _rows()[:-1]),
            source=cpi_source,
        )
    bad = _rows()
    bad[3] = (*bad[3][:3], "nan", bad[3][4])
    with pytest.raises(CpiBlsError, match="finite positive"):
        parse_bls_cpi_file(
            _flat(tmp_path / BLS_CPI_FILENAME, bad),
            source=cpi_source,
        )


def test_cpi_import_partitions_and_authenticates_snapshot(
    tmp_path: Path, cpi_source: ContextSource
) -> None:
    source_file = _flat(tmp_path / BLS_CPI_FILENAME)
    output = tmp_path / "cpi"
    result = import_bls_cpi(
        source_file,
        output,
        cpi_source,
        ingested_at_utc=pd.Timestamp("2026-09-11T00:00:00Z"),
    )
    assert result["source_id"] == "cpi"
    assert result["series_id"] == BLS_CPI_SERIES
    assert result["strict_pit"] is False
    assert result["rows"] == 59
    bundle = json.loads((output / "cpi.bundle-set.json").read_text())
    assert bundle["bundles"] == [f"cpi-{year}.manifest.json" for year in range(2020, 2025)]
    loaded = load_context_data(output / "cpi.bundle-set.json", cpi_source)
    assert len(loaded) == 59
    assert loaded.raw_sha256.nunique() == 1
    artifacts = json.loads((output / "source_artifacts.json").read_text())
    assert artifacts["source_file"]["sha256"] == loaded.raw_sha256.iloc[0]
    parquet = output / "cpi-2020.parquet"
    parquet.write_bytes(parquet.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        load_context_data(output / "cpi.bundle-set.json", cpi_source)


def test_cpi_filename_and_ingestion_time_are_authenticated(
    tmp_path: Path, cpi_source: ContextSource
) -> None:
    wrong = _flat(tmp_path / "renamed.tsv")
    with pytest.raises(CpiBlsError, match="named exactly"):
        parse_bls_cpi_file(wrong, source=cpi_source)
    path = _flat(tmp_path / BLS_CPI_FILENAME)
    with pytest.raises(CpiBlsError, match="ingested_at_utc precedes"):
        parse_bls_cpi_file(
            path,
            source=cpi_source,
            ingested_at_utc=pd.Timestamp("2020-01-01T00:00:00Z"),
        )
