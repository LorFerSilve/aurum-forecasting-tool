"""Local HistData XAGUSD M1 ingestion for Phase-10 context research.

This adapter is intentionally separate from the frozen XAUUSD ingestion adapter.
It authenticates local annual archives and converts bid closes into provider-neutral
context observations. Historical availability remains a modeled-latency assumption.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

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

HISTDATA_SILVER_SYMBOL = "XAGUSD"
HISTDATA_SILVER_SOURCE = "histdata"
HISTDATA_SILVER_TIMEFRAME = "1min"
HISTDATA_SILVER_SOURCE_OFFSET = "-05:00"
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_MAX_MEMBER_BYTES = 512 * 1024 * 1024
_SOURCE_TIMEZONE = timezone(
    -timedelta(hours=5),
    name=HISTDATA_SILVER_SOURCE_OFFSET,
)
_MEMBER_PATTERN = re.compile(
    r"(?:^|/)DAT_ASCII_XAGUSD_M1_(\d{4})(?:\d{2})?\.csv$",
    flags=re.IGNORECASE,
)
_RAW_COLUMNS = (
    "source_timestamp",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "volume",
)


class SilverHistDataError(ValueError):
    """Raised when XAGUSD source evidence cannot be trusted."""


def _as_utc(value: datetime | pd.Timestamp | None, *, name: str) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp(datetime.now(UTC))
    result = pd.Timestamp(value)
    if result.tzinfo is None or str(result.tz_convert("UTC").tz) != "UTC":
        raise SilverHistDataError(f"{name} must be timezone-aware")
    return result.tz_convert("UTC").as_unit("ns")


def _validate_source(source: ContextSource) -> None:
    if source.source_id != "silver":
        raise SilverHistDataError("HistData XAGUSD adapter requires source_id='silver'")
    if source.availability_basis != "modeled_latency":
        raise SilverHistDataError(
            "HistData XAGUSD has no historical provider release timestamps; "
            "availability_basis must remain modeled_latency"
        )
    if source.revision_policy != "append_only":
        raise SilverHistDataError("HistData annual XAGUSD archives use append_only semantics")
    parsed = urlsplit(source.source_url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "histdata.com",
        "www.histdata.com",
    }:
        raise SilverHistDataError("HistData XAGUSD source_url must use official HTTPS histdata.com")


def _read_archive(path: Path) -> bytes:
    try:
        if path.is_symlink():
            raise SilverHistDataError("silver archive must not be a symlink")
        if not path.is_file():
            raise SilverHistDataError(f"silver archive is missing: {path}")
        size = path.stat().st_size
        if size <= 0 or size > _MAX_ARCHIVE_BYTES:
            raise SilverHistDataError("silver archive has an invalid or excessive size")
        with path.open("rb") as stream:
            payload = stream.read(_MAX_ARCHIVE_BYTES + 1)
    except OSError as exc:
        raise SilverHistDataError(f"cannot read silver archive: {path}") from exc
    if len(payload) > _MAX_ARCHIVE_BYTES:
        raise SilverHistDataError("silver archive exceeds the configured size limit")
    return payload


def _members(zipped: zipfile.ZipFile, year: int) -> list[zipfile.ZipInfo]:
    matches: list[zipfile.ZipInfo] = []
    total_size = 0
    for info in zipped.infolist():
        if info.is_dir():
            continue
        normalized = info.filename.replace("\\", "/")
        match = _MEMBER_PATTERN.search(normalized)
        if match is None:
            continue
        if int(match.group(1)) != year:
            raise SilverHistDataError(
                "XAGUSD archive member period differs from the declared source year"
            )
        if info.flag_bits & 0x1:
            raise SilverHistDataError("XAGUSD archive contains an encrypted CSV member")
        if info.file_size <= 0 or info.file_size > _MAX_MEMBER_BYTES:
            raise SilverHistDataError("XAGUSD archive member has an invalid or excessive size")
        total_size += info.file_size
        if total_size > _MAX_MEMBER_BYTES:
            raise SilverHistDataError("XAGUSD archive expands beyond the safe member-size budget")
        matches.append(info)
    if not matches:
        raise SilverHistDataError(
            "ZIP archive contains no non-empty DAT_ASCII_XAGUSD_M1_<period>.csv member"
        )
    return sorted(matches, key=lambda item: item.filename.casefold())


def _parse_member(
    zipped: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> pd.DataFrame:
    try:
        with zipped.open(info, "r") as stream:
            raw = pd.read_csv(
                stream,
                sep=";",
                header=None,
                dtype="string",
                encoding="utf-8-sig",
                keep_default_na=False,
                na_filter=False,
            )
    except (EmptyDataError, ParserError, UnicodeError) as exc:
        raise SilverHistDataError(
            f"cannot parse Generic ASCII XAGUSD member {info.filename!r}"
        ) from exc
    if raw.empty or raw.shape[1] != len(_RAW_COLUMNS):
        raise SilverHistDataError(
            "XAGUSD CSV must contain non-empty headerless "
            "timestamp;open;high;low;close;volume rows"
        )
    raw.columns = list(_RAW_COLUMNS)
    timestamp_text = raw["source_timestamp"].str.strip()
    if not timestamp_text.str.fullmatch(r"\d{8} \d{6}").all():
        raise SilverHistDataError("XAGUSD source timestamp must use YYYYMMDD HHMMSS")
    try:
        local = pd.to_datetime(
            timestamp_text,
            format="%Y%m%d %H%M%S",
            exact=True,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise SilverHistDataError("XAGUSD archive contains an invalid timestamp") from exc
    if not local.dt.second.eq(0).all():
        raise SilverHistDataError("XAGUSD archive contains a non-minute timestamp")

    result = pd.DataFrame(
        {
            "timestamp_open_utc": local.dt.tz_localize(_SOURCE_TIMEZONE).dt.tz_convert("UTC"),
            "source_member": info.filename,
        }
    )
    for column in ("bid_open", "bid_high", "bid_low", "bid_close", "volume"):
        try:
            values = pd.to_numeric(raw[column].str.strip(), errors="raise").astype("float64")
        except (TypeError, ValueError) as exc:
            raise SilverHistDataError(f"XAGUSD {column} must be numeric") from exc
        if not np.isfinite(values.to_numpy()).all():
            raise SilverHistDataError(f"XAGUSD {column} must be finite")
        result[column] = values

    prices = result[["bid_open", "bid_high", "bid_low", "bid_close"]]
    if prices.le(0).any().any():
        raise SilverHistDataError("XAGUSD OHLC prices must be strictly positive")
    body_high = result[["bid_open", "bid_close"]].max(axis=1)
    body_low = result[["bid_open", "bid_close"]].min(axis=1)
    if (
        result["bid_high"].lt(body_high).any()
        or result["bid_low"].gt(body_low).any()
        or result["bid_high"].lt(result["bid_low"]).any()
    ):
        raise SilverHistDataError("XAGUSD archive violates OHLC invariants")
    return result


def parse_histdata_xagusd_archive(
    archive_path: str | Path,
    *,
    year: int,
    source: ContextSource,
    ingested_at_utc: datetime | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Convert one authenticated annual XAGUSD archive into context observations."""
    _validate_source(source)
    if type(year) is not int or year not in range(2020, 2025):
        raise SilverHistDataError("XAGUSD development archive year must be 2020-2024")
    supplied_path = Path(archive_path).expanduser()
    payload = _read_archive(supplied_path)
    path = supplied_path.resolve()
    digest = hashlib.sha256(payload).hexdigest()
    ingested = _as_utc(ingested_at_utc, name="ingested_at_utc")

    frames: list[pd.DataFrame] = []
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
            members = _members(zipped, year)
            bad_member = zipped.testzip()
            if bad_member is not None:
                raise SilverHistDataError(
                    f"XAGUSD ZIP member failed its CRC check: {bad_member!r}"
                )
            frames = [_parse_member(zipped, info) for info in members]
    except zipfile.BadZipFile as exc:
        raise SilverHistDataError("XAGUSD source file is not an intact ZIP archive") from exc

    candles = pd.concat(frames, ignore_index=True)
    local_year = candles["timestamp_open_utc"].dt.tz_convert(_SOURCE_TIMEZONE).dt.year
    if not local_year.eq(year).all():
        raise SilverHistDataError("XAGUSD archive contains candles outside its source year")
    candles.sort_values(["timestamp_open_utc", "source_member"], kind="stable", inplace=True)
    exact = candles.duplicated(
        [
            "timestamp_open_utc",
            "bid_open",
            "bid_high",
            "bid_low",
            "bid_close",
            "volume",
        ],
        keep="first",
    )
    duplicates_removed = int(exact.sum())
    candles = candles.loc[~exact].copy()
    if candles["timestamp_open_utc"].duplicated().any():
        raise SilverHistDataError("XAGUSD archive contains conflicting candles at one timestamp")

    observed = candles["timestamp_open_utc"] + pd.Timedelta(minutes=1)
    development = observed.ge(DEVELOPMENT_START) & observed.lt(DEVELOPMENT_END)
    candles = candles.loc[development].copy()
    observed = observed.loc[development]
    if candles.empty:
        raise SilverHistDataError("XAGUSD archive has no observations inside development scope")
    available = observed + pd.to_timedelta(source.publication_delay_seconds, unit="s")
    if ingested < available.max():
        raise SilverHistDataError(
            "ingested_at_utc precedes modeled availability for at least one observation"
        )

    identifier = observed.dt.strftime("%Y%m%dT%H%M%SZ")
    source_uri = (
        "https://www.histdata.com/download-free-forex-historical-data/"
        f"?/ascii/1-minute-bar-quotes/xagusd/{year}"
    )
    observations = pd.DataFrame(
        {
            "source_id": source.source_id,
            "observation_id": ("xagusd-m1-" + identifier).to_numpy(),
            "observed_at_utc": observed.array,
            "available_at_utc": available.array,
            "ingested_at_utc": ingested,
            "value": candles["bid_close"].to_numpy(dtype=np.float64),
            "revision_id": f"sha256:{digest}",
            "source_uri": source_uri,
            "raw_sha256": digest,
        },
        columns=OBSERVATION_COLUMNS,
    )
    observations = validate_observations(observations, source)
    metadata: dict[str, object] = {
        "year": year,
        "path": str(path),
        "sha256": digest,
        "size_bytes": len(payload),
        "source_symbol": HISTDATA_SILVER_SYMBOL,
        "source_timezone_offset": HISTDATA_SILVER_SOURCE_OFFSET,
        "source_observes_dst": False,
        "timestamp_semantics": "candle_open",
        "observation_semantics": "candle_close",
        "availability_basis": "modeled_latency",
        "publication_delay_seconds": source.publication_delay_seconds,
        "ingested_at_utc": ingested.isoformat(),
        "source_page_url": source_uri,
        "rows": len(observations),
        "exact_duplicates_removed": duplicates_removed,
        "members": sorted(candles["source_member"].unique().tolist()),
    }
    return observations, metadata


def _find_archive(directory: Path, year: int) -> Path:
    expected = f"HISTDATA_COM_ASCII_XAGUSD_M1_{year}.zip".casefold()
    matches = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.name.casefold() == expected
    ]
    if len(matches) != 1:
        raise SilverHistDataError(
            f"expected exactly one HISTDATA_COM_ASCII_XAGUSD_M1_{year}.zip in {directory}"
        )
    return matches[0]


def import_histdata_silver_archives(
    archive_directory: str | Path,
    output_directory: str | Path,
    source: ContextSource,
    *,
    years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024),
    ingested_at_utc: datetime | pd.Timestamp | None = None,
) -> dict[str, object]:
    """Build annual authenticated Parquet bundles and one bundle-set manifest."""
    _validate_source(source)
    if (
        not years
        or tuple(sorted(set(years))) != years
        or not set(years) <= set(range(2020, 2025))
    ):
        raise SilverHistDataError("years must be a sorted unique subset of 2020-2024")
    supplied_raw = Path(archive_directory).expanduser()
    if supplied_raw.is_symlink():
        raise SilverHistDataError("archive_directory must not be a symlink")
    raw = supplied_raw.resolve(strict=True)
    if not raw.is_dir():
        raise SilverHistDataError("archive_directory must be a real directory")

    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    ingested = _as_utc(ingested_at_utc, name="ingested_at_utc")
    bundle_names: list[str] = []
    archive_records: list[dict[str, object]] = []
    total_rows = 0

    for year in years:
        observations, metadata = parse_histdata_xagusd_archive(
            _find_archive(raw, year),
            year=year,
            source=source,
            ingested_at_utc=ingested,
        )
        data_name = f"silver-{year}.parquet"
        manifest_name = f"silver-{year}.manifest.json"
        data_path = destination / data_name
        write_parquet_atomic(data_path, observations)
        start = observations["observed_at_utc"].min()
        end = observations["observed_at_utc"].max() + pd.Timedelta(1, unit="ns")
        write_json_atomic(
            destination / manifest_name,
            {
                "schema_version": 1,
                "source_id": source.source_id,
                "availability_basis": source.availability_basis,
                "observation_start_utc": start.isoformat(),
                "observation_end_utc": end.isoformat(),
                "row_count": len(observations),
                "format": "parquet",
                "file": data_name,
                "sha256": sha256_file(data_path),
            },
        )
        bundle_names.append(manifest_name)
        archive_records.append(metadata)
        total_rows += len(observations)

    set_path = destination / "silver.bundle-set.json"
    write_json_atomic(
        set_path,
        {
            "schema_version": 1,
            "kind": "bundle_set",
            "source_id": source.source_id,
            "availability_basis": source.availability_basis,
            "bundles": bundle_names,
        },
    )
    source_archives = destination / "source_archives.json"
    write_json_atomic(
        source_archives,
        {
            "schema_version": 1,
            "source_id": source.source_id,
            "source_symbol": HISTDATA_SILVER_SYMBOL,
            "availability_basis": source.availability_basis,
            "availability_is_historical_evidence": False,
            "years": list(years),
            "archives": archive_records,
        },
    )
    return {
        "source_id": source.source_id,
        "source_symbol": HISTDATA_SILVER_SYMBOL,
        "availability_basis": source.availability_basis,
        "strict_pit": False,
        "years": list(years),
        "rows": total_rows,
        "bundle_set": str(set_path),
        "bundle_set_sha256": sha256_file(set_path),
        "source_archives": str(source_archives),
        "source_archives_sha256": sha256_file(source_archives),
    }


__all__ = [
    "HISTDATA_SILVER_SOURCE_OFFSET",
    "HISTDATA_SILVER_SYMBOL",
    "SilverHistDataError",
    "import_histdata_silver_archives",
    "parse_histdata_xagusd_archive",
]
