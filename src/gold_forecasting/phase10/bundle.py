"""Load an offline context bundle after validating its declared scope and checksum."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gold_forecasting.phase10.contracts import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    OBSERVATION_COLUMNS,
    TIME_COLUMNS,
    ContextSource,
    require_strict_pit,
    validate_observations,
)

_MAX_CSV_BYTES = 256 * 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_UTC_ISO_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|\+00:00)"
)


def _utc_timestamp(value: str, name: str) -> pd.Timestamp:
    if _UTC_ISO_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be an explicit ISO timestamp in UTC (Z or +00:00)")
    try:
        return pd.Timestamp(value).as_unit("ns")
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} contains an invalid UTC timestamp") from exc


class ContextBundleManifest(BaseModel):
    """The CSV checksum is separate from each row's upstream archive provenance."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1]
    source_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    availability_basis: Literal["provider_timestamp", "modeled_latency", "synthetic"]
    observation_start_utc: str
    observation_end_utc: str
    row_count: int = Field(ge=0)
    file: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")

    @field_validator("schema_version", mode="before")
    @classmethod
    def _integer_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @field_validator("observation_start_utc", "observation_end_utc")
    @classmethod
    def _explicit_utc(cls, value: str) -> str:
        _utc_timestamp(value, "manifest observation span")
        return value

    @field_validator("file")
    @classmethod
    def _sibling_csv(cls, value: str) -> str:
        if (
            any(character in value for character in ("/", "\\", ":", "\x00"))
            or Path(value).suffix != ".csv"
            or value.strip() != value
        ):
            raise ValueError("file must be a single sibling .csv filename without traversal")
        return value


def _read_bounded(path: Path, limit: int, name: str) -> bytes:
    try:
        if path.is_symlink():
            raise ValueError(f"{name} must not be a symlink")
        if not path.is_file():
            raise ValueError(f"{name} is missing or is not a regular file: {path}")
        if path.stat().st_size > limit:
            raise ValueError(f"{name} exceeds the {limit}-byte size limit")
        with path.open("rb") as stream:
            payload = stream.read(limit + 1)
    except OSError as exc:
        raise ValueError(f"cannot read {name}: {path}") from exc
    if len(payload) > limit:
        raise ValueError(f"{name} exceeds the {limit}-byte size limit")
    return payload


def load_context_bundle(
    manifest_path: str | Path,
    source: ContextSource,
    *,
    strict_pit: bool = False,
) -> pd.DataFrame:
    """Validate metadata before accessing CSV bytes, then hash and parse one buffer.

    The half-open declared observation span must stay in development 2020-2024.
    Upstream ``raw_sha256`` values remain claims supplied by the exporter: this
    loader verifies the bundled CSV, not the original provider archives.
    """
    path = Path(manifest_path).expanduser()
    manifest = ContextBundleManifest.model_validate_json(
        _read_bounded(path, _MAX_MANIFEST_BYTES, "context manifest")
    )
    if manifest.source_id != source.source_id:
        raise ValueError("bundle source_id differs from source contract")
    if manifest.availability_basis != source.availability_basis:
        raise ValueError("bundle availability_basis differs from source contract")
    start = _utc_timestamp(manifest.observation_start_utc, "observation_start_utc")
    end = _utc_timestamp(manifest.observation_end_utc, "observation_end_utc")
    if not DEVELOPMENT_START <= start < end <= DEVELOPMENT_END:
        raise ValueError("bundle observation span must remain in development 2020-2024")
    if strict_pit:
        require_strict_pit(source)

    csv_path = path.parent / manifest.file
    payload = _read_bounded(csv_path, _MAX_CSV_BYTES, "context CSV")
    if hashlib.sha256(payload).hexdigest() != manifest.sha256.lower():
        raise ValueError("context CSV SHA-256 differs from bundle manifest")
    try:
        frame = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
    except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError("context CSV cannot be parsed") from exc
    if tuple(frame.columns) != OBSERVATION_COLUMNS:
        raise ValueError("context CSV must contain exactly OBSERVATION_COLUMNS in contract order")
    if len(frame) != manifest.row_count:
        raise ValueError("context CSV row_count differs from bundle manifest")
    for name in TIME_COLUMNS:
        if not frame[name].str.fullmatch(_UTC_ISO_PATTERN).all():
            raise ValueError(f"{name} must contain explicit ISO timestamps in UTC (Z or +00:00)")
        try:
            frame[name] = pd.to_datetime(frame[name], format="ISO8601", errors="raise", utc=True)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{name} contains invalid UTC timestamps") from exc
    try:
        frame["value"] = pd.to_numeric(frame["value"].replace("", float("nan")), errors="raise")
    except ValueError as exc:
        raise ValueError("context CSV value must be numeric or empty") from exc
    result = validate_observations(frame, source)
    if (
        (result["observed_at_utc"] < start)
        | (result["observed_at_utc"] >= end)
    ).any():
        raise ValueError("context observation lies outside the declared half-open bundle span")
    return result
