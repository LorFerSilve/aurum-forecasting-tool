"""HistData Generic ASCII M1 ingestion for the XAU/USD research feed.

HistData's free M1 files contain bid OHLC values and a ``volume`` field.  The
provider documents that volume as unusable, so it is retained for source
fidelity but is explicitly marked ``volume_reliable=False`` and must not be
used as a model feature.

The adapter deliberately ends at a provider-neutral candle table.  A future
broker adapter can produce the same core columns without changing downstream
validation, feature, or model code.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import zipfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Final
from urllib.parse import urljoin, urlsplit

import httpx
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

HISTDATA_BASE_URL: Final = "https://www.histdata.com"
HISTDATA_SOURCE: Final = "histdata"
HISTDATA_SOURCE_SYMBOL: Final = "XAUUSD"
HISTDATA_INSTRUMENT: Final = "XAU_USD"
HISTDATA_TIMEFRAME: Final = "1min"
HISTDATA_PRICE_SIDE: Final = "bid"
HISTDATA_SOURCE_TIMEZONE_OFFSET: Final = "-05:00"
HISTDATA_USER_AGENT: Final = (
    "gold-forecasting/0.1 (personal, non-commercial research; one request per archive)"
)

_SOURCE_TIMEZONE = timezone(-timedelta(hours=5), name=HISTDATA_SOURCE_TIMEZONE_OFFSET)
_RAW_COLUMNS: Final = (
    "source_timestamp",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "volume",
)
_CSV_MEMBER_PATTERN: Final = re.compile(
    r"(?:^|/)DAT_ASCII_XAUUSD_M1_\d{4}(?:\d{2})?\.csv$",
    flags=re.IGNORECASE,
)
_MAX_ARCHIVE_BYTES: Final = 256 * 1024 * 1024


class HistDataError(RuntimeError):
    """Base error for the HistData adapter."""


class HistDataDownloadError(HistDataError):
    """Raised when HistData's public two-step download fails."""


class HistDataArchiveError(HistDataError):
    """Raised when a raw file is not an intact ZIP containing CSV data."""


class HistDataParseError(HistDataError):
    """Raised when an archive does not follow the Generic ASCII M1 schema."""


@dataclass(frozen=True, slots=True)
class HistDataArchive:
    """Identity and provenance of one immutable raw annual archive."""

    path: Path
    sha256: str
    size_bytes: int
    year: int
    source_page_url: str
    download_url: str
    observed_at_utc: datetime
    reused: bool
    source: str = HISTDATA_SOURCE
    source_symbol: str = HISTDATA_SOURCE_SYMBOL
    timeframe: str = HISTDATA_TIMEFRAME

    @property
    def dataset_version(self) -> str:
        """Content-addressed version suitable for manifests and candle rows."""

        return f"sha256:{self.sha256}"

    def metadata(self) -> dict[str, str | int | bool]:
        """Return JSON-serializable source metadata without the one-use token."""

        return {
            "source": self.source,
            "source_symbol": self.source_symbol,
            "timeframe": self.timeframe,
            "year": self.year,
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "dataset_version": self.dataset_version,
            "source_page_url": self.source_page_url,
            "download_url": self.download_url,
            "observed_at_utc": _format_utc(self.observed_at_utc),
            "reused": self.reused,
            "volume_reliable": False,
        }


@dataclass(frozen=True, slots=True)
class HistDataIngestionResult:
    """Raw provenance paired with normalized, fully closed M1 candles."""

    archive: HistDataArchive
    candles: pd.DataFrame


class _DownloadFormParser(HTMLParser):
    """Extract the hidden fields from the public ``/get.php`` form."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.forms: list[tuple[str, dict[str, str]]] = []
        self._action: str | None = None
        self._fields: dict[str, str] = {}

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = {name.lower(): value or "" for name, value in attrs}
        if tag.lower() == "form":
            action = attributes.get("action", "")
            is_download_form = (
                attributes.get("id", "").casefold() == "file_down"
                or attributes.get("name", "").casefold() == "file_down"
            )
            if is_download_form and urlsplit(action).path.rstrip("/").endswith("/get.php"):
                self._action = action
                self._fields = {}
            else:
                self._action = None
                self._fields = {}
            return
        if tag.lower() != "input" or self._action is None:
            return
        name = attributes.get("name")
        if name:
            self._fields[name] = attributes.get("value", "")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "form" and self._action is not None:
            self.forms.append((self._action, self._fields.copy()))
            self._action = None
            self._fields = {}


class HistDataM1Provider:
    """Download and normalize annual public HistData XAUUSD M1 archives.

    No credentials are accepted or sent.  When ``client`` is supplied, its
    lifecycle stays with the caller; this makes the adapter easy to test and to
    integrate into a longer-lived ingestion service.
    """

    def __init__(
        self,
        *,
        base_url: str = HISTDATA_BASE_URL,
        client: httpx.Client | None = None,
        timeout: float | httpx.Timeout = 60.0,
        user_agent: str = HISTDATA_USER_AGENT,
        clock: Callable[[], datetime] | None = None,
        max_archive_bytes: int = _MAX_ARCHIVE_BYTES,
    ) -> None:
        normalized_base = base_url.rstrip("/")
        parsed_base = urlsplit(normalized_base)
        if parsed_base.scheme not in {"http", "https"} or not parsed_base.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if not user_agent.strip():
            raise ValueError("user_agent must identify the research client")
        if max_archive_bytes <= 0:
            raise ValueError("max_archive_bytes must be positive")
        self._base_url = normalized_base
        self._client = client
        self._timeout = timeout
        self._user_agent = user_agent
        self._clock = clock or (lambda: datetime.now(UTC))
        self._max_archive_bytes = max_archive_bytes

    def source_page_url(self, year: int) -> str:
        """Return the annual Generic ASCII M1 selection page."""

        checked_year = _validate_year(year)
        return (
            f"{self._base_url}/download-free-forex-historical-data/"
            f"?/ascii/1-minute-bar-quotes/{HISTDATA_SOURCE_SYMBOL.lower()}/{checked_year}"
        )

    def raw_archive_path(self, raw_directory: str | Path, year: int) -> Path:
        """Return the stable path used for one annual immutable ZIP."""

        checked_year = _validate_year(year)
        filename = f"HISTDATA_COM_ASCII_{HISTDATA_SOURCE_SYMBOL}_M1_{checked_year}.zip"
        return (
            Path(raw_directory).expanduser()
            / HISTDATA_SOURCE
            / HISTDATA_SOURCE_SYMBOL
            / HISTDATA_TIMEFRAME
            / filename
        )

    def download_year(self, year: int, raw_directory: str | Path) -> HistDataArchive:
        """Fetch a year through HistData's GET-token/POST flow exactly once.

        An existing target is validated and returned without any network call.
        Existing bytes are never overwritten, including if they are corrupt.
        """

        checked_year = _validate_year(year)
        target = self.raw_archive_path(raw_directory, checked_year).resolve()
        page_url = self.source_page_url(checked_year)
        default_download_url = f"{self._base_url}/get.php"
        observed_at = _as_utc_datetime(self._clock(), name="clock result")

        if target.exists():
            if not target.is_file():
                raise HistDataArchiveError(f"raw archive target is not a file: {target}")
            _validate_zip_path(target)
            return self._archive_record(
                target,
                checked_year,
                page_url,
                default_download_url,
                observed_at,
                reused=True,
            )

        if self._client is None:
            with httpx.Client(follow_redirects=True, timeout=self._timeout) as client:
                payload, download_url = self._download_payload(client, page_url, checked_year)
        else:
            payload, download_url = self._download_payload(
                self._client,
                page_url,
                checked_year,
            )

        _validate_zip_bytes(payload)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as destination:
                destination.write(payload)
                destination.flush()
                os.fsync(destination.fileno())
        except FileExistsError:
            # Another process won the race.  Its immutable result is acceptable
            # only when it is a valid archive with identical content.
            _validate_zip_path(target)
            if _sha256_path(target) != hashlib.sha256(payload).hexdigest():
                raise HistDataArchiveError(
                    f"an immutable archive with different content already exists: {target}"
                ) from None
        except OSError:
            # Never leave a partial file looking like a valid raw artefact.
            with suppress(OSError):
                target.unlink(missing_ok=True)
            raise

        return self._archive_record(
            target,
            checked_year,
            page_url,
            download_url,
            observed_at,
            reused=False,
        )

    def parse_archive(
        self,
        archive: HistDataArchive | str | Path | bytes,
        *,
        ingested_at_utc: datetime | None = None,
        as_of_utc: datetime | None = None,
    ) -> pd.DataFrame:
        """Parse an archive into provider-neutral, completed UTC candles."""

        return parse_histdata_m1(
            archive,
            ingested_at_utc=ingested_at_utc,
            as_of_utc=as_of_utc,
        )

    def ingest_year(
        self,
        year: int,
        raw_directory: str | Path,
        *,
        ingested_at_utc: datetime | None = None,
        as_of_utc: datetime | None = None,
    ) -> HistDataIngestionResult:
        """Download (or reuse) one raw archive and return its normalized rows."""

        archive = self.download_year(year, raw_directory)
        candles = self.parse_archive(
            archive,
            ingested_at_utc=ingested_at_utc,
            as_of_utc=as_of_utc,
        )
        return HistDataIngestionResult(archive=archive, candles=candles)

    def _download_payload(
        self,
        client: httpx.Client,
        page_url: str,
        year: int,
    ) -> tuple[bytes, str]:
        request_headers = {
            "User-Agent": self._user_agent,
            "Accept": "text/html,application/xhtml+xml",
        }
        try:
            page_response = client.get(page_url, headers=request_headers, timeout=self._timeout)
            page_response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HistDataDownloadError(
                f"could not load HistData download page for {year}: {exc}"
            ) from exc

        action, fields = _extract_download_form(page_response.text)
        _validate_download_fields(fields, year)
        download_url = urljoin(page_url, action)
        if urlsplit(download_url).netloc != urlsplit(self._base_url).netloc:
            raise HistDataDownloadError("HistData download form points to a different host")

        post_headers = {
            "User-Agent": self._user_agent,
            "Accept": "application/zip,application/octet-stream;q=0.9,*/*;q=0.1",
            "Referer": page_url,
        }
        try:
            response = client.post(
                download_url,
                data=fields,
                headers=post_headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HistDataDownloadError(
                f"could not download HistData archive for {year}: {exc}"
            ) from exc

        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError as exc:
                raise HistDataDownloadError("HistData returned an invalid Content-Length") from exc
            if declared_size > self._max_archive_bytes:
                raise HistDataDownloadError(
                    f"HistData archive exceeds the {self._max_archive_bytes}-byte safety limit"
                )
        payload = response.content
        if len(payload) > self._max_archive_bytes:
            raise HistDataDownloadError(
                f"HistData archive exceeds the {self._max_archive_bytes}-byte safety limit"
            )
        return payload, download_url

    @staticmethod
    def _archive_record(
        path: Path,
        year: int,
        page_url: str,
        download_url: str,
        observed_at: datetime,
        *,
        reused: bool,
    ) -> HistDataArchive:
        return HistDataArchive(
            path=path,
            sha256=_sha256_path(path),
            size_bytes=path.stat().st_size,
            year=year,
            source_page_url=page_url,
            download_url=download_url,
            observed_at_utc=observed_at,
            reused=reused,
        )


def download_histdata_year(
    year: int,
    raw_directory: str | Path,
    *,
    client: httpx.Client | None = None,
    base_url: str = HISTDATA_BASE_URL,
) -> HistDataArchive:
    """Convenience wrapper around :class:`HistDataM1Provider`."""

    return HistDataM1Provider(base_url=base_url, client=client).download_year(year, raw_directory)


def parse_histdata_m1(
    archive: HistDataArchive | str | Path | bytes,
    *,
    ingested_at_utc: datetime | None = None,
    as_of_utc: datetime | None = None,
) -> pd.DataFrame:
    """Parse headerless HistData Generic ASCII M1 data into UTC candles.

    HistData timestamps are interpreted using a fixed ``UTC-05:00`` offset,
    including during summer.  Rows whose one-minute close is later than
    ``as_of_utc`` are omitted, so every returned row has ``is_complete=True``.
    """

    payload, raw_sha256, archive_metadata = _read_archive(archive)
    ingestion_time = _as_utc_datetime(
        ingested_at_utc or datetime.now(UTC),
        name="ingested_at_utc",
    )
    completion_cutoff = _as_utc_datetime(
        as_of_utc or ingestion_time,
        name="as_of_utc",
    )

    frames: list[pd.DataFrame] = []
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
            for info in _csv_members(zipped):
                try:
                    with zipped.open(info, "r") as source:
                        raw = pd.read_csv(
                            source,
                            sep=";",
                            header=None,
                            dtype="string",
                            encoding="utf-8-sig",
                            keep_default_na=False,
                            na_filter=False,
                        )
                except (EmptyDataError, ParserError, UnicodeError) as exc:
                    raise HistDataParseError(
                        f"cannot parse Generic ASCII member {info.filename!r}: {exc}"
                    ) from exc
                if raw.empty:
                    raise HistDataParseError(f"CSV member {info.filename!r} contains no rows")
                if raw.shape[1] != len(_RAW_COLUMNS):
                    raise HistDataParseError(
                        f"CSV member {info.filename!r} has {raw.shape[1]} fields; expected 6 "
                        "headerless fields: timestamp; open; high; low; close; volume"
                    )
                raw.columns = list(_RAW_COLUMNS)
                frames.append(_normalize_member(raw, info.filename))
    except zipfile.BadZipFile as exc:
        raise HistDataArchiveError("raw input is not an intact ZIP archive") from exc

    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["timestamp_open_utc", "source_member"], kind="stable", inplace=True)
    exact_columns = [
        "timestamp_open_utc",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "volume",
    ]
    exact_duplicates = result.duplicated(subset=exact_columns, keep="first")
    duplicates_removed = int(exact_duplicates.sum())
    result = result.loc[~exact_duplicates].copy()
    if bool(result.duplicated(subset=["timestamp_open_utc"], keep=False).any()):
        raise HistDataParseError(
            "archive contains conflicting candles for the same source timestamp"
        )
    result["timestamp_close_utc"] = result["timestamp_open_utc"] + pd.Timedelta(minutes=1)
    result = result.loc[result["timestamp_close_utc"] <= pd.Timestamp(completion_cutoff)].copy()

    result.insert(0, "instrument", HISTDATA_INSTRUMENT)
    result.insert(1, "timeframe", HISTDATA_TIMEFRAME)
    result["is_complete"] = True
    result["ingested_at_utc"] = pd.Timestamp(ingestion_time)
    result["source"] = HISTDATA_SOURCE
    result["source_symbol"] = HISTDATA_SOURCE_SYMBOL
    result["price_side"] = HISTDATA_PRICE_SIDE
    result["volume_reliable"] = False
    result["raw_file_hash"] = raw_sha256
    result["dataset_version"] = f"sha256:{raw_sha256}"

    column_order = [
        "instrument",
        "timeframe",
        "timestamp_open_utc",
        "timestamp_close_utc",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "volume",
        "is_complete",
        "ingested_at_utc",
        "source",
        "source_symbol",
        "price_side",
        "volume_reliable",
        "dataset_version",
        "raw_file_hash",
        "source_member",
    ]
    result = result.loc[:, column_order].reset_index(drop=True)
    result.attrs.update(
        {
            "source": HISTDATA_SOURCE,
            "source_symbol": HISTDATA_SOURCE_SYMBOL,
            "timeframe": HISTDATA_TIMEFRAME,
            "timestamp_semantics": "candle_open",
            "source_timezone_offset": HISTDATA_SOURCE_TIMEZONE_OFFSET,
            "source_observes_dst": False,
            "price_side": HISTDATA_PRICE_SIDE,
            "volume_reliable": False,
            "volume_warning": "HistData M1 volume is retained but is not reliable for modelling.",
            "raw_file_hash": raw_sha256,
            "dataset_version": f"sha256:{raw_sha256}",
            "exact_duplicates_removed": duplicates_removed,
            **archive_metadata,
        }
    )
    return result


def _extract_download_form(html: str) -> tuple[str, dict[str, str]]:
    parser = _DownloadFormParser()
    parser.feed(html)
    parser.close()
    required = {"tk", "date", "datemonth", "platform", "timeframe", "fxpair"}
    for action, fields in parser.forms:
        if required.issubset(fields) and fields["tk"].strip():
            return action, {name: fields[name] for name in required}
    raise HistDataDownloadError("HistData page does not contain a usable download token/form")


def _validate_download_fields(fields: dict[str, str], year: int) -> None:
    expected = {
        "date": str(year),
        "datemonth": str(year),
        "platform": "ASCII",
        "timeframe": "M1",
        "fxpair": HISTDATA_SOURCE_SYMBOL,
    }
    for name, expected_value in expected.items():
        if fields.get(name, "").upper() != expected_value.upper():
            raise HistDataDownloadError(
                f"HistData form field {name!r} does not match the requested annual XAUUSD M1 file"
            )


def _validate_year(year: int) -> int:
    if isinstance(year, bool) or not isinstance(year, int):
        raise TypeError("year must be an integer")
    if not 1900 <= year <= 9999:
        raise ValueError("year must be between 1900 and 9999")
    return year


def _validate_zip_bytes(payload: bytes) -> None:
    if not payload:
        raise HistDataArchiveError("HistData returned an empty response instead of a ZIP archive")
    stream = io.BytesIO(payload)
    if not zipfile.is_zipfile(stream):
        raise HistDataArchiveError("HistData response is not a real ZIP archive")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
            _csv_members(zipped)
            bad_member = zipped.testzip()
    except HistDataArchiveError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise HistDataArchiveError("HistData ZIP archive cannot be read safely") from exc
    if bad_member is not None:
        raise HistDataArchiveError(f"HistData ZIP member failed its CRC check: {bad_member!r}")


def _validate_zip_path(path: Path) -> None:
    try:
        if not zipfile.is_zipfile(path):
            raise HistDataArchiveError(f"raw file is not a real ZIP archive: {path}")
        with zipfile.ZipFile(path) as zipped:
            _csv_members(zipped)
            bad_member = zipped.testzip()
    except HistDataArchiveError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise HistDataArchiveError(f"raw ZIP archive cannot be read safely: {path}") from exc
    if bad_member is not None:
        raise HistDataArchiveError(f"raw ZIP member failed its CRC check: {bad_member!r}")


def _csv_members(zipped: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = [
        info
        for info in zipped.infolist()
        if not info.is_dir()
        and _CSV_MEMBER_PATTERN.search(info.filename.replace("\\", "/"))
        and info.file_size > 0
    ]
    if not members:
        raise HistDataArchiveError(
            "ZIP archive contains no non-empty DAT_ASCII_XAUUSD_M1_<period>.csv member"
        )
    encrypted = [info.filename for info in members if info.flag_bits & 0x1]
    if encrypted:
        raise HistDataArchiveError(f"ZIP archive contains encrypted CSV member: {encrypted[0]!r}")
    return sorted(members, key=lambda info: info.filename.casefold())


def _normalize_member(raw: pd.DataFrame, member_name: str) -> pd.DataFrame:
    timestamp_text = raw["source_timestamp"].str.strip()
    if not bool(timestamp_text.str.fullmatch(r"\d{8} \d{6}").all()):
        raise HistDataParseError(
            f"CSV member {member_name!r} contains a timestamp outside YYYYMMDD HHMMSS"
        )
    try:
        source_timestamps = pd.to_datetime(
            timestamp_text,
            format="%Y%m%d %H%M%S",
            exact=True,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise HistDataParseError(f"CSV member {member_name!r} has an invalid timestamp") from exc
    if not bool(source_timestamps.dt.second.eq(0).all()):
        raise HistDataParseError(f"CSV member {member_name!r} contains a non-minute timestamp")

    normalized = pd.DataFrame(
        {
            "timestamp_open_utc": source_timestamps.dt.tz_localize(_SOURCE_TIMEZONE).dt.tz_convert(
                UTC
            ),
            "source_member": member_name,
        }
    )
    for column in ("bid_open", "bid_high", "bid_low", "bid_close", "volume"):
        try:
            values = pd.to_numeric(raw[column].str.strip(), errors="raise").astype("float64")
        except (TypeError, ValueError) as exc:
            raise HistDataParseError(
                f"CSV member {member_name!r} contains a non-numeric {column!r} value"
            ) from exc
        if not bool(np.isfinite(values.to_numpy()).all()):
            raise HistDataParseError(
                f"CSV member {member_name!r} contains a non-finite {column!r} value"
            )
        normalized[column] = values
    return normalized


def _read_archive(
    archive: HistDataArchive | str | Path | bytes,
) -> tuple[bytes, str, dict[str, str | int | bool]]:
    metadata: dict[str, str | int | bool] = {}
    expected_sha256: str | None = None
    if isinstance(archive, HistDataArchive):
        path = archive.path
        expected_sha256 = archive.sha256
        metadata = {
            "archive_path": str(path),
            "archive_year": archive.year,
            "source_page_url": archive.source_page_url,
            "download_url": archive.download_url,
        }
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise HistDataArchiveError(f"cannot read raw archive {path}: {exc}") from exc
    elif isinstance(archive, bytes):
        payload = archive
    else:
        path = Path(archive).expanduser().resolve()
        metadata = {"archive_path": str(path)}
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise HistDataArchiveError(f"cannot read raw archive {path}: {exc}") from exc

    _validate_zip_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise HistDataArchiveError("immutable raw archive content no longer matches its SHA-256")
    return payload, digest, metadata


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise HistDataArchiveError(f"cannot hash raw archive {path}: {exc}") from exc
    return digest.hexdigest()


def _as_utc_datetime(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = [
    "HISTDATA_BASE_URL",
    "HISTDATA_INSTRUMENT",
    "HISTDATA_SOURCE",
    "HISTDATA_SOURCE_SYMBOL",
    "HISTDATA_TIMEFRAME",
    "HISTDATA_USER_AGENT",
    "HistDataArchive",
    "HistDataArchiveError",
    "HistDataDownloadError",
    "HistDataError",
    "HistDataIngestionResult",
    "HistDataM1Provider",
    "HistDataParseError",
    "download_histdata_year",
    "parse_histdata_m1",
]
