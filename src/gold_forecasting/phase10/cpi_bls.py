"""BLS CPI-U source authentication and modeled historical release timing.

The source is the unadjusted CPI-U All items index, series CUUR0000SA0, from the
BLS public time-series flat file. Historical release dates are frozen from the
BLS annual release calendars for 2020-2024. The flat file is a current snapshot
and does not preserve the value vintage visible at each historical cutoff, so
these observations remain modeled-latency evidence and can never directly
promote a champion or activate trading.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from gold_forecasting.artifacts import sha256_file, write_json_atomic, write_parquet_atomic
from gold_forecasting.phase10.contracts import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    OBSERVATION_COLUMNS,
    ContextSource,
    validate_observations,
)

BLS_CPI_SERIES = "CUUR0000SA0"
BLS_CPI_FILENAME = "cu.data.1.AllItems"
BLS_CPI_SOURCE_URL = "https://download.bls.gov/pub/time.series/cu/cu.data.1.AllItems"
BLS_CPI_SERIES_URL = "https://download.bls.gov/pub/time.series/cu/cu.series"
BLS_CPI_RELEASE_TIME = "08:30 America/New_York"
BLS_CPI_SCHEDULE_URLS = tuple(f"https://www.bls.gov/schedule/{year}/" for year in range(2020, 2025))
_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_YEARS = (2020, 2021, 2022, 2023, 2024)

# Reference month -> official BLS CPI release date. December 2024 is deliberately
# absent because its January 2025 release is outside the frozen development window.
CPI_RELEASE_DATES: dict[str, str] = {
    "2020-01": "2020-02-13",
    "2020-02": "2020-03-11",
    "2020-03": "2020-04-10",
    "2020-04": "2020-05-12",
    "2020-05": "2020-06-10",
    "2020-06": "2020-07-14",
    "2020-07": "2020-08-12",
    "2020-08": "2020-09-11",
    "2020-09": "2020-10-13",
    "2020-10": "2020-11-12",
    "2020-11": "2020-12-10",
    "2020-12": "2021-01-13",
    "2021-01": "2021-02-10",
    "2021-02": "2021-03-10",
    "2021-03": "2021-04-13",
    "2021-04": "2021-05-12",
    "2021-05": "2021-06-10",
    "2021-06": "2021-07-13",
    "2021-07": "2021-08-11",
    "2021-08": "2021-09-14",
    "2021-09": "2021-10-13",
    "2021-10": "2021-11-10",
    "2021-11": "2021-12-10",
    "2021-12": "2022-01-12",
    "2022-01": "2022-02-10",
    "2022-02": "2022-03-10",
    "2022-03": "2022-04-12",
    "2022-04": "2022-05-11",
    "2022-05": "2022-06-10",
    "2022-06": "2022-07-13",
    "2022-07": "2022-08-10",
    "2022-08": "2022-09-13",
    "2022-09": "2022-10-13",
    "2022-10": "2022-11-10",
    "2022-11": "2022-12-13",
    "2022-12": "2023-01-12",
    "2023-01": "2023-02-14",
    "2023-02": "2023-03-14",
    "2023-03": "2023-04-12",
    "2023-04": "2023-05-10",
    "2023-05": "2023-06-13",
    "2023-06": "2023-07-12",
    "2023-07": "2023-08-10",
    "2023-08": "2023-09-13",
    "2023-09": "2023-10-12",
    "2023-10": "2023-11-14",
    "2023-11": "2023-12-12",
    "2023-12": "2024-01-11",
    "2024-01": "2024-02-13",
    "2024-02": "2024-03-12",
    "2024-03": "2024-04-10",
    "2024-04": "2024-05-15",
    "2024-05": "2024-06-12",
    "2024-06": "2024-07-11",
    "2024-07": "2024-08-14",
    "2024-08": "2024-09-11",
    "2024-09": "2024-10-10",
    "2024-10": "2024-11-13",
    "2024-11": "2024-12-11",
}


class CpiBlsError(ValueError):
    """Raised when the local BLS CPI source cannot be authenticated safely."""


def _as_utc(value: datetime | pd.Timestamp | None, *, name: str) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp(datetime.now(UTC))
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise CpiBlsError(f"{name} must be timezone-aware")
    return result.tz_convert("UTC").as_unit("ns")


def _validate_source(source: ContextSource) -> None:
    expected = {
        "source_id": "cpi",
        "source_url": BLS_CPI_SOURCE_URL,
        "source_timezone": "America/New_York release schedule; DST-aware",
        "publication_delay_seconds": 0,
        "stale_after_seconds": 7776000,
        "availability_basis": "modeled_latency",
        "revision_policy": "append_only",
        "missing_policy": "price_only_fallback",
        "enabled": True,
    }
    for name, expected_value in expected.items():
        if getattr(source, name) != expected_value:
            raise CpiBlsError(f"frozen CPI source mismatch: {name}")


def _read_source(path: Path) -> bytes:
    try:
        if path.is_symlink():
            raise CpiBlsError("BLS CPI source file must not be a symlink")
        if not path.is_file():
            raise CpiBlsError(f"BLS CPI source file is missing: {path}")
        size = path.stat().st_size
        if size <= 0 or size > _MAX_SOURCE_BYTES:
            raise CpiBlsError("BLS CPI source file has an invalid or excessive size")
        with path.open("rb") as stream:
            payload = stream.read(_MAX_SOURCE_BYTES + 1)
    except OSError as exc:
        raise CpiBlsError(f"cannot read BLS CPI source file: {path}") from exc
    if len(payload) > _MAX_SOURCE_BYTES:
        raise CpiBlsError("BLS CPI source file exceeds the configured size limit")
    return payload


def _schedule_digest() -> str:
    payload = json.dumps(CPI_RELEASE_DATES, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _release_timestamp(reference_month: str) -> pd.Timestamp:
    release_date = CPI_RELEASE_DATES[reference_month]
    local = pd.Timestamp(f"{release_date} 08:30:00", tz="America/New_York")
    return local.tz_convert("UTC").as_unit("ns")


def parse_bls_cpi_file(
    source_path: str | Path,
    *,
    source: ContextSource,
    ingested_at_utc: datetime | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Parse the official BLS AllItems flat file into 2020-2024 CPI observations."""
    _validate_source(source)
    supplied = Path(source_path).expanduser()
    if supplied.name != BLS_CPI_FILENAME:
        raise CpiBlsError(f"BLS CPI source must be named exactly {BLS_CPI_FILENAME}")
    payload = _read_source(supplied)
    path = supplied.resolve()
    digest = hashlib.sha256(payload).hexdigest()
    ingested = _as_utc(ingested_at_utc, name="ingested_at_utc")
    try:
        raw = pd.read_csv(
            io.BytesIO(payload),
            sep="\t",
            dtype="string",
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
        )
    except (EmptyDataError, ParserError, UnicodeError) as exc:
        raise CpiBlsError("BLS CPI flat file cannot be parsed") from exc
    raw.columns = [str(name).strip() for name in raw.columns]
    expected_columns = ("series_id", "year", "period", "value", "footnote_codes")
    if raw.empty or tuple(raw.columns) != expected_columns:
        raise CpiBlsError("BLS CPI flat file must contain the canonical five BLS columns")
    for name in expected_columns:
        raw[name] = raw[name].astype("string").str.strip()

    series = raw.loc[raw["series_id"].eq(BLS_CPI_SERIES)].copy()
    if series.empty:
        raise CpiBlsError(f"BLS CPI series {BLS_CPI_SERIES} is missing")
    if not series["year"].str.fullmatch(r"\d{4}").all():
        raise CpiBlsError("BLS CPI years must be four-digit integers")
    monthly = series.loc[series["period"].str.fullmatch(r"M(?:0[1-9]|1[0-2])")].copy()
    monthly_years = pd.to_numeric(monthly["year"], errors="raise").astype("int64")
    development = monthly.loc[monthly_years.between(2020, 2024)].copy()
    if len(development) != 60:
        raise CpiBlsError("CUUR0000SA0 must contain exactly 60 monthly rows for 2020-2024")
    keys = development["year"] + "-" + development["period"].str[1:]
    if keys.duplicated().any() or set(keys) != {
        f"{year}-{month:02d}" for year in _YEARS for month in range(1, 13)
    }:
        raise CpiBlsError("CUUR0000SA0 2020-2024 reference months must be complete and unique")
    try:
        values = pd.to_numeric(development["value"], errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise CpiBlsError(
            "CUUR0000SA0 values must be numeric, finite positive index levels <= 1000"
        ) from exc
    if (
        not np.isfinite(values.to_numpy()).all()
        or values.le(0.0).any()
        or values.gt(1000.0).any()
    ):
        raise CpiBlsError("CUUR0000SA0 values must be finite positive index levels <= 1000")

    development = development.assign(
        reference_month=keys.to_numpy(),
        numeric_value=values.to_numpy(),
    )
    selected = development.loc[development["reference_month"].isin(CPI_RELEASE_DATES)].copy()
    selected = selected.sort_values("reference_month").reset_index(drop=True)
    if selected["reference_month"].tolist() != sorted(CPI_RELEASE_DATES):
        raise CpiBlsError("BLS CPI selected reference months differ from frozen release schedule")
    observed = pd.to_datetime(selected["reference_month"] + "-01", utc=True)
    available = pd.Series(
        [_release_timestamp(key) for key in selected["reference_month"]],
        dtype="datetime64[ns, UTC]",
    )
    if (available >= DEVELOPMENT_END).any():
        raise CpiBlsError("CPI release schedule escapes the 2020-2024 development window")
    if ingested < available.max():
        raise CpiBlsError("ingested_at_utc precedes the final modeled CPI availability")
    if ((observed < DEVELOPMENT_START) | (observed >= DEVELOPMENT_END)).any():
        raise CpiBlsError("CPI observations escape development scope")

    observations = pd.DataFrame(
        {
            "source_id": source.source_id,
            "observation_id": ("cpi-u-" + selected["reference_month"]).to_numpy(),
            "observed_at_utc": observed.array,
            "available_at_utc": available.array,
            "ingested_at_utc": ingested,
            "value": selected["numeric_value"].to_numpy(dtype=np.float64),
            "revision_id": f"sha256:{digest}",
            "source_uri": BLS_CPI_SOURCE_URL,
            "raw_sha256": digest,
        },
        columns=OBSERVATION_COLUMNS,
    )
    observations = validate_observations(observations, source)
    year_rows = {
        str(year): int(observations["observed_at_utc"].dt.year.eq(year).sum()) for year in _YEARS
    }
    metadata: dict[str, object] = {
        "path": str(path),
        "filename": BLS_CPI_FILENAME,
        "sha256": digest,
        "size_bytes": len(payload),
        "series_id": BLS_CPI_SERIES,
        "series_definition_url": BLS_CPI_SERIES_URL,
        "source_url": BLS_CPI_SOURCE_URL,
        "availability_basis": "modeled_latency",
        "availability_is_historical_evidence": False,
        "reference_time_semantics": "reference_month_first_day_00:00_UTC",
        "release_schedule": BLS_CPI_RELEASE_TIME,
        "release_rule": "frozen_official_bls_2020_2024_calendar_dates",
        "schedule_urls": list(BLS_CPI_SCHEDULE_URLS),
        "schedule_sha256": _schedule_digest(),
        "revision_snapshot": "latest_bls_flat_file_no_historical_value_vintages",
        "ingested_at_utc": ingested.isoformat(),
        "raw_rows": len(raw),
        "series_rows": len(series),
        "development_month_rows": len(development),
        "excluded_after_development_release": len(development) - len(selected),
        "rows": len(observations),
        "first_reference_month": selected["reference_month"].iloc[0],
        "last_reference_month": selected["reference_month"].iloc[-1],
        "year_rows": year_rows,
    }
    return observations, metadata


def import_bls_cpi(
    source_path: str | Path,
    output_directory: str | Path,
    source: ContextSource,
    *,
    ingested_at_utc: datetime | pd.Timestamp | None = None,
) -> dict[str, object]:
    """Build authenticated annual Parquet bundles from one official BLS snapshot."""
    _validate_source(source)
    observations, metadata = parse_bls_cpi_file(
        source_path,
        source=source,
        ingested_at_utc=ingested_at_utc,
    )
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    bundle_names: list[str] = []
    total_rows = 0
    for year in _YEARS:
        annual = observations.loc[observations["observed_at_utc"].dt.year.eq(year)].reset_index(
            drop=True
        )
        if annual.empty:
            raise CpiBlsError(f"CPI year {year} contains no usable observations")
        data_name = f"cpi-{year}.parquet"
        manifest_name = f"cpi-{year}.manifest.json"
        data_path = destination / data_name
        write_parquet_atomic(data_path, annual)
        start = annual["observed_at_utc"].min()
        end = annual["observed_at_utc"].max() + pd.Timedelta(1, unit="ns")
        write_json_atomic(
            destination / manifest_name,
            {
                "schema_version": 1,
                "source_id": "cpi",
                "availability_basis": "modeled_latency",
                "observation_start_utc": start.isoformat(),
                "observation_end_utc": end.isoformat(),
                "row_count": len(annual),
                "format": "parquet",
                "file": data_name,
                "sha256": sha256_file(data_path),
            },
        )
        bundle_names.append(manifest_name)
        total_rows += len(annual)
    set_path = destination / "cpi.bundle-set.json"
    write_json_atomic(
        set_path,
        {
            "schema_version": 1,
            "kind": "bundle_set",
            "source_id": "cpi",
            "availability_basis": "modeled_latency",
            "bundles": bundle_names,
        },
    )
    artifacts = destination / "source_artifacts.json"
    write_json_atomic(
        artifacts,
        {
            "schema_version": 1,
            "source_id": "cpi",
            "series_id": BLS_CPI_SERIES,
            "availability_basis": "modeled_latency",
            "availability_is_historical_evidence": False,
            "years": list(_YEARS),
            "source_file": metadata,
        },
    )
    return {
        "source_id": "cpi",
        "series_id": BLS_CPI_SERIES,
        "availability_basis": "modeled_latency",
        "strict_pit": False,
        "years": list(_YEARS),
        "rows": total_rows,
        "bundle_set": str(set_path),
        "bundle_set_sha256": sha256_file(set_path),
        "source_artifacts": str(artifacts),
        "source_artifacts_sha256": sha256_file(artifacts),
    }


__all__ = [
    "BLS_CPI_FILENAME",
    "BLS_CPI_RELEASE_TIME",
    "BLS_CPI_SCHEDULE_URLS",
    "BLS_CPI_SERIES",
    "BLS_CPI_SERIES_URL",
    "BLS_CPI_SOURCE_URL",
    "CPI_RELEASE_DATES",
    "CpiBlsError",
    "import_bls_cpi",
    "parse_bls_cpi_file",
]
