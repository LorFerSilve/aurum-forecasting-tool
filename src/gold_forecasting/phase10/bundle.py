"""Load offline context bundles after validating scope and checksums."""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
_MAX_PARQUET_BYTES = 512 * 1024 * 1024
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


def _safe_sibling(value: str, *, suffixes: tuple[str, ...], name: str) -> str:
    if (
        any(character in value for character in ("/", "\\", ":", "\x00"))
        or value.strip() != value
        or not any(value.endswith(suffix) for suffix in suffixes)
    ):
        expected = " or ".join(suffixes)
        raise ValueError(f"{name} must be a single sibling {expected} filename without traversal")
    return value


class ContextBundleManifest(BaseModel):
    """One authenticated observation partition."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1]
    source_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    availability_basis: Literal["provider_timestamp", "modeled_latency", "synthetic"]
    observation_start_utc: str
    observation_end_utc: str
    row_count: int = Field(ge=0)
    format: Literal["csv", "parquet"] = "csv"
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
    def _sibling_data_file(cls, value: str) -> str:
        return _safe_sibling(
            value,
            suffixes=(".csv", ".parquet"),
            name="file",
        )

    @model_validator(mode="after")
    def _format_matches_suffix(self) -> ContextBundleManifest:
        expected = ".csv" if self.format == "csv" else ".parquet"
        if not self.file.endswith(expected):
            raise ValueError(f"bundle format={self.format!r} requires a {expected} file")
        return self


class ContextBundleSetManifest(BaseModel):
    """Authenticated collection of sibling annual/period bundle manifests."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1]
    kind: Literal["bundle_set"]
    source_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    availability_basis: Literal["provider_timestamp", "modeled_latency", "synthetic"]
    bundles: tuple[str, ...] = Field(min_length=1)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _integer_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @field_validator("bundles")
    @classmethod
    def _safe_bundle_names(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("bundle-set manifest names must be unique")
        return tuple(
            _safe_sibling(
                value,
                suffixes=(".manifest.json",),
                name="bundle",
            )
            for value in values
        )


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


def _validate_manifest_identity(
    source: ContextSource,
    *,
    source_id: str,
    availability_basis: str,
) -> None:
    if source_id != source.source_id:
        raise ValueError("bundle source_id differs from source contract")
    if availability_basis != source.availability_basis:
        raise ValueError("bundle availability_basis differs from source contract")


def _load_csv(payload: bytes) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
    except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError("context CSV cannot be parsed") from exc
    if tuple(frame.columns) != OBSERVATION_COLUMNS:
        raise ValueError("context CSV must contain exactly OBSERVATION_COLUMNS in contract order")
    for name in TIME_COLUMNS:
        if not frame[name].str.fullmatch(_UTC_ISO_PATTERN).all():
            raise ValueError(f"{name} must contain explicit ISO timestamps in UTC (Z or +00:00)")
        try:
            frame[name] = pd.to_datetime(
                frame[name],
                format="ISO8601",
                errors="raise",
                utc=True,
            )
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{name} contains invalid UTC timestamps") from exc
    try:
        frame["value"] = pd.to_numeric(
            frame["value"].replace("", float("nan")),
            errors="raise",
        )
    except ValueError as exc:
        raise ValueError("context CSV value must be numeric or empty") from exc
    return frame


def _load_parquet(payload: bytes) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(io.BytesIO(payload))
    except Exception as exc:
        raise ValueError("context Parquet cannot be parsed") from exc
    if tuple(frame.columns) != OBSERVATION_COLUMNS:
        raise ValueError(
            "context Parquet must contain exactly OBSERVATION_COLUMNS in contract order"
        )
    return frame


def load_context_bundle(
    manifest_path: str | Path,
    source: ContextSource,
    *,
    strict_pit: bool = False,
) -> pd.DataFrame:
    """Validate metadata before accessing authenticated observation bytes."""
    path = Path(manifest_path).expanduser()
    manifest = ContextBundleManifest.model_validate_json(
        _read_bounded(path, _MAX_MANIFEST_BYTES, "context manifest")
    )
    _validate_manifest_identity(
        source,
        source_id=manifest.source_id,
        availability_basis=manifest.availability_basis,
    )
    start = _utc_timestamp(manifest.observation_start_utc, "observation_start_utc")
    end = _utc_timestamp(manifest.observation_end_utc, "observation_end_utc")
    if not DEVELOPMENT_START <= start < end <= DEVELOPMENT_END:
        raise ValueError("bundle observation span must remain in development 2020-2024")
    if strict_pit:
        require_strict_pit(source)

    data_path = path.parent / manifest.file
    limit = _MAX_CSV_BYTES if manifest.format == "csv" else _MAX_PARQUET_BYTES
    payload = _read_bounded(
        data_path,
        limit,
        f"context {manifest.format.upper()}",
    )
    if hashlib.sha256(payload).hexdigest() != manifest.sha256.lower():
        raise ValueError(
            f"context {manifest.format.upper()} SHA-256 differs from bundle manifest"
        )
    frame = _load_csv(payload) if manifest.format == "csv" else _load_parquet(payload)
    if len(frame) != manifest.row_count:
        raise ValueError(
            f"context {manifest.format.upper()} row_count differs from bundle manifest"
        )
    result = validate_observations(frame, source)
    if (
        (result["observed_at_utc"] < start)
        | (result["observed_at_utc"] >= end)
    ).any():
        raise ValueError("context observation lies outside the declared half-open bundle span")
    return result


def load_context_bundle_set(
    manifest_path: str | Path,
    source: ContextSource,
    *,
    strict_pit: bool = False,
) -> pd.DataFrame:
    """Load a sibling partition set and revalidate the concatenated release stream."""
    path = Path(manifest_path).expanduser()
    manifest = ContextBundleSetManifest.model_validate_json(
        _read_bounded(path, _MAX_MANIFEST_BYTES, "context bundle-set manifest")
    )
    _validate_manifest_identity(
        source,
        source_id=manifest.source_id,
        availability_basis=manifest.availability_basis,
    )
    if strict_pit:
        require_strict_pit(source)
    frames = [
        load_context_bundle(path.parent / name, source, strict_pit=False)
        for name in manifest.bundles
    ]
    combined = pd.concat(frames, ignore_index=True)
    return validate_observations(combined, source)


def load_context_data(
    manifest_path: str | Path,
    source: ContextSource,
    *,
    strict_pit: bool = False,
) -> pd.DataFrame:
    """Load either one bundle manifest or a partitioned bundle-set manifest."""
    path = Path(manifest_path).expanduser()
    payload = _read_bounded(path, _MAX_MANIFEST_BYTES, "context manifest")
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("context manifest is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("context manifest must contain a JSON object")
    if decoded.get("kind") == "bundle_set":
        return load_context_bundle_set(path, source, strict_pit=strict_pit)
    return load_context_bundle(path, source, strict_pit=strict_pit)


__all__ = [
    "ContextBundleManifest",
    "ContextBundleSetManifest",
    "load_context_bundle",
    "load_context_bundle_set",
    "load_context_data",
]
