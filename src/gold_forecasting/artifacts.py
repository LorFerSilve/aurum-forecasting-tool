"""Content-addressed manifests and atomic artifact writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

ArtifactLayer = Literal["raw", "curated", "features", "labels", "model_table"]


class ArtifactError(ValueError):
    """Raised when an artifact cannot be represented safely."""


class FileDigest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    dataset_version: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    layer: ArtifactLayer
    source: str = Field(min_length=1)
    instrument: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    period_start_utc: datetime | None
    period_end_utc: datetime | None
    row_count: int = Field(ge=0)
    parameters: dict[str, Any]
    inputs: tuple[FileDigest, ...]
    outputs: tuple[FileDigest, ...]
    created_at_utc: datetime

    @field_validator("period_start_utc", "period_end_utc", "created_at_utc")
    @classmethod
    def timestamps_are_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("manifest timestamps must be timezone-aware UTC")
        return value


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it entirely in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_digest(path: str | Path, *, relative_to: str | Path | None = None) -> FileDigest:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ArtifactError(f"artifact is not a file: {resolved}")
    display_path = (
        resolved.relative_to(Path(relative_to).resolve()).as_posix()
        if relative_to is not None
        else resolved.name
    )
    return FileDigest(
        path=display_path,
        size_bytes=resolved.stat().st_size,
        sha256=sha256_file(resolved),
    )


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"payload is not canonical JSON: {exc}") from exc
    return serialized.encode("utf-8")


def content_version(payload: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 version for canonical JSON content."""

    return f"sha256:{hashlib.sha256(_json_bytes(payload)).hexdigest()}"


def build_dataset_manifest(
    *,
    layer: ArtifactLayer,
    source: str,
    instrument: str,
    timeframe: str,
    row_count: int,
    inputs: Sequence[FileDigest],
    outputs: Sequence[FileDigest],
    parameters: Mapping[str, Any],
    period_start_utc: datetime | None,
    period_end_utc: datetime | None,
    created_at_utc: datetime | None = None,
) -> DatasetManifest:
    """Build a manifest whose version excludes nondeterministic creation time."""

    stable_payload: dict[str, Any] = {
        "schema_version": 1,
        "layer": layer,
        "source": source,
        "instrument": instrument,
        "timeframe": timeframe,
        "period_start_utc": (
            period_start_utc.isoformat() if period_start_utc is not None else None
        ),
        "period_end_utc": period_end_utc.isoformat() if period_end_utc is not None else None,
        "row_count": row_count,
        "parameters": dict(parameters),
        "inputs": [item.model_dump(mode="json") for item in inputs],
        "outputs": [item.model_dump(mode="json") for item in outputs],
    }
    version = content_version(stable_payload)
    return DatasetManifest(
        dataset_version=version,
        layer=layer,
        source=source,
        instrument=instrument,
        timeframe=timeframe,
        period_start_utc=period_start_utc,
        period_end_utc=period_end_utc,
        row_count=row_count,
        parameters=dict(parameters),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        created_at_utc=created_at_utc or datetime.now(UTC),
    )


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON object atomically, never exposing a partial valid artifact."""

    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
            default=_json_default,
        ).encode("utf-8")
        + b"\n"
    )
    write_bytes_atomic(path, encoded)


def write_text_atomic(path: str | Path, text: str) -> None:
    """Write UTF-8 text atomically."""

    write_bytes_atomic(path, text.encode("utf-8"))


def write_parquet_atomic(path: str | Path, frame: pd.DataFrame) -> None:
    """Write a compressed Parquet artifact without exposing a partial file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.stem}.",
            suffix=".parquet.tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_bytes_atomic(path: str | Path, payload: bytes) -> None:
    """Write bytes atomically in the destination directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_manifest_atomic(path: str | Path, manifest: DatasetManifest) -> None:
    write_json_atomic(path, manifest.model_dump(mode="json"))


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


__all__ = [
    "ArtifactError",
    "ArtifactLayer",
    "DatasetManifest",
    "FileDigest",
    "build_dataset_manifest",
    "content_version",
    "file_digest",
    "sha256_file",
    "write_bytes_atomic",
    "write_json_atomic",
    "write_manifest_atomic",
    "write_parquet_atomic",
    "write_text_atomic",
]
