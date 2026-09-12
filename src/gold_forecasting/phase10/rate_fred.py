"""Local FRED DFII10 ingestion for the Phase-10 real-rate hypothesis.

The source is the daily 10-year inflation-indexed Treasury constant-maturity
rate (DFII10), released in H.15 and distributed through FRED. The downloaded
historical snapshot does not preserve original vintages, so availability is
modeled from the published H.15 schedule and is never strict PIT evidence.
"""

from __future__ import annotations

import hashlib
import io
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from pandas.tseries.holiday import USFederalHolidayCalendar

from gold_forecasting.artifacts import sha256_file, write_json_atomic, write_parquet_atomic
from gold_forecasting.phase10.contracts import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    OBSERVATION_COLUMNS,
    ContextSource,
    validate_observations,
)

FRED_RATE_SERIES = "DFII10"
FRED_RATE_FILENAME = "DFII10_2020_2024.csv"
FRED_RATE_SOURCE_URL = "https://fred.stlouisfed.org/series/DFII10"
H15_RELEASE_TIME = "16:15 America/New_York"
_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_YEARS = (2020, 2021, 2022, 2023, 2024)


class RateFredError(ValueError):
    """Raised when the local real-rate source cannot be authenticated safely."""


def _as_utc(value: datetime | pd.Timestamp | None, *, name: str) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp(datetime.now(UTC))
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise RateFredError(f"{name} must be timezone-aware")
    return result.tz_convert("UTC").as_unit("ns")


def _validate_source(source: ContextSource) -> None:
    expected = {
        "source_id": "rate",
        "source_url": FRED_RATE_SOURCE_URL,
        "source_timezone": "America/New_York release schedule; DST-aware",
        "publication_delay_seconds": 0,
        "stale_after_seconds": 432000,
        "availability_basis": "modeled_latency",
        "revision_policy": "append_only",
        "missing_policy": "price_only_fallback",
        "enabled": True,
    }
    for name, expected_value in expected.items():
        if getattr(source, name) != expected_value:
            raise RateFredError(f"frozen DFII10 source mismatch: {name}")


def _read_source(path: Path) -> bytes:
    try:
        if path.is_symlink():
            raise RateFredError("DFII10 source file must not be a symlink")
        if not path.is_file():
            raise RateFredError(f"DFII10 source file is missing: {path}")
        size = path.stat().st_size
        if size <= 0 or size > _MAX_SOURCE_BYTES:
            raise RateFredError("DFII10 source file has an invalid or excessive size")
        with path.open("rb") as stream:
            payload = stream.read(_MAX_SOURCE_BYTES + 1)
    except OSError as exc:
        raise RateFredError(f"cannot read DFII10 source file: {path}") from exc
    if len(payload) > _MAX_SOURCE_BYTES:
        raise RateFredError("DFII10 source file exceeds the configured size limit")
    return payload


def _holiday_dates(start: pd.Timestamp, end: pd.Timestamp) -> set[date]:
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(
        start=(start - pd.Timedelta(days=7)).to_pydatetime(),
        end=(end + pd.Timedelta(days=14)).to_pydatetime(),
    )
    return {stamp.date() for stamp in holidays}


def _next_h15_release(day: date, holidays: set[date]) -> pd.Timestamp:
    candidate = day + timedelta(days=1)
    while candidate.weekday() >= 5 or candidate in holidays:
        candidate += timedelta(days=1)
    local = pd.Timestamp(datetime.combine(candidate, time(16, 15)), tz="America/New_York")
    return local.tz_convert("UTC").as_unit("ns")


def parse_fred_dfii10_csv(
    csv_path: str | Path,
    *,
    source: ContextSource,
    ingested_at_utc: datetime | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Parse one bounded 2020-2024 FRED snapshot into daily rate observations.

    The reference date is represented as 00:00 UTC. It is not an availability
    timestamp. Availability is conservatively modeled as the next U.S. federal
    business day at 16:15 America/New_York, matching the published H.15 schedule.
    Rows whose FRED value is '.' or blank are treated as no new observation and
    are skipped; the previous valid release may therefore remain the latest
    known value until a newer valid rate is released.
    """
    _validate_source(source)
    supplied = Path(csv_path).expanduser()
    if supplied.name != FRED_RATE_FILENAME:
        raise RateFredError(f"DFII10 source must be named exactly {FRED_RATE_FILENAME}")
    payload = _read_source(supplied)
    path = supplied.resolve()
    digest = hashlib.sha256(payload).hexdigest()
    ingested = _as_utc(ingested_at_utc, name="ingested_at_utc")
    try:
        raw = pd.read_csv(
            io.BytesIO(payload),
            dtype="string",
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
        )
    except (EmptyDataError, ParserError, UnicodeError) as exc:
        raise RateFredError("DFII10 CSV cannot be parsed") from exc
    if raw.empty or tuple(raw.columns) not in {
        ("DATE", FRED_RATE_SERIES),
        ("observation_date", FRED_RATE_SERIES),
    }:
        raise RateFredError("DFII10 CSV must contain exactly DATE/observation_date and DFII10")
    date_column = raw.columns[0]
    date_text = raw[date_column].str.strip()
    if not date_text.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise RateFredError("DFII10 dates must use YYYY-MM-DD")
    try:
        dates = pd.to_datetime(date_text, format="%Y-%m-%d", exact=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise RateFredError("DFII10 CSV contains an invalid date") from exc
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise RateFredError("DFII10 reference dates must be unique and increasing")
    if dates.dt.year.lt(2020).any() or dates.dt.year.gt(2024).any():
        raise RateFredError("DFII10 input must be pre-bounded to development years 2020-2024")
    if set(dates.dt.year.unique()) != set(_YEARS):
        raise RateFredError("DFII10 input must contain observations from every year 2020-2024")

    value_text = raw[FRED_RATE_SERIES].str.strip()
    missing = value_text.isin(["", "."])
    numeric_text = value_text.mask(missing, "0")
    try:
        values = pd.to_numeric(numeric_text, errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise RateFredError("DFII10 values must be numeric, '.' or blank") from exc
    observed_values = values.loc[~missing]
    if (
        observed_values.empty
        or not np.isfinite(observed_values.to_numpy()).all()
        or observed_values.lt(-25.0).any()
        or observed_values.gt(25.0).any()
    ):
        raise RateFredError("DFII10 observed yields must be finite percentages in [-25, 25]")

    observed_dates = dates.loc[~missing].reset_index(drop=True)
    values = values.loc[~missing].reset_index(drop=True)
    observed = pd.to_datetime(observed_dates.dt.strftime("%Y-%m-%d"), utc=True)
    holidays = _holiday_dates(observed.min(), observed.max())
    available = pd.Series(
        [_next_h15_release(stamp.date(), holidays) for stamp in observed],
        dtype="datetime64[ns, UTC]",
    )
    if ingested < available.max():
        raise RateFredError("ingested_at_utc precedes modeled H.15 availability")
    if ((observed < DEVELOPMENT_START) | (observed >= DEVELOPMENT_END)).any():
        raise RateFredError("DFII10 observations escape development scope")

    identifiers = observed.dt.strftime("%Y-%m-%d")
    observations = pd.DataFrame(
        {
            "source_id": source.source_id,
            "observation_id": ("dfii10-" + identifiers).to_numpy(),
            "observed_at_utc": observed.array,
            "available_at_utc": available.array,
            "ingested_at_utc": ingested,
            "value": values.to_numpy(dtype=np.float64),
            "revision_id": f"sha256:{digest}",
            "source_uri": FRED_RATE_SOURCE_URL,
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
        "filename": FRED_RATE_FILENAME,
        "sha256": digest,
        "size_bytes": len(payload),
        "series_id": FRED_RATE_SERIES,
        "source_url": FRED_RATE_SOURCE_URL,
        "availability_basis": "modeled_latency",
        "availability_is_historical_evidence": False,
        "reference_time_semantics": "reference_date_00:00_UTC",
        "release_schedule": H15_RELEASE_TIME,
        "release_rule": "next_us_federal_business_day_16:15_America/New_York",
        "revision_snapshot": "latest_downloaded_history_no_vintages",
        "ingested_at_utc": ingested.isoformat(),
        "raw_rows": len(raw),
        "missing_rows_skipped": int(missing.sum()),
        "rows": len(observations),
        "first_reference_date": dates.min().strftime("%Y-%m-%d"),
        "last_reference_date": dates.max().strftime("%Y-%m-%d"),
        "year_rows": year_rows,
    }
    return observations, metadata


def import_fred_dfii10(
    csv_path: str | Path,
    output_directory: str | Path,
    source: ContextSource,
    *,
    ingested_at_utc: datetime | pd.Timestamp | None = None,
) -> dict[str, object]:
    """Build authenticated annual Parquet bundles from one local DFII10 snapshot."""
    _validate_source(source)
    observations, metadata = parse_fred_dfii10_csv(
        csv_path,
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
            raise RateFredError(f"DFII10 year {year} contains no usable observations")
        data_name = f"rate-{year}.parquet"
        manifest_name = f"rate-{year}.manifest.json"
        data_path = destination / data_name
        write_parquet_atomic(data_path, annual)
        start = annual["observed_at_utc"].min()
        end = annual["observed_at_utc"].max() + pd.Timedelta(1, unit="ns")
        write_json_atomic(
            destination / manifest_name,
            {
                "schema_version": 1,
                "source_id": "rate",
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
    set_path = destination / "rate.bundle-set.json"
    write_json_atomic(
        set_path,
        {
            "schema_version": 1,
            "kind": "bundle_set",
            "source_id": "rate",
            "availability_basis": "modeled_latency",
            "bundles": bundle_names,
        },
    )
    artifacts = destination / "source_artifacts.json"
    write_json_atomic(
        artifacts,
        {
            "schema_version": 1,
            "source_id": "rate",
            "series_id": FRED_RATE_SERIES,
            "availability_basis": "modeled_latency",
            "availability_is_historical_evidence": False,
            "years": list(_YEARS),
            "source_file": metadata,
        },
    )
    return {
        "source_id": "rate",
        "series_id": FRED_RATE_SERIES,
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
    "FRED_RATE_FILENAME",
    "FRED_RATE_SERIES",
    "FRED_RATE_SOURCE_URL",
    "H15_RELEASE_TIME",
    "RateFredError",
    "import_fred_dfii10",
    "parse_fred_dfii10_csv",
]
