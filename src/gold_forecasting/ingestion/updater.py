"""Fail-closed, idempotent incremental acquisition orchestration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from gold_forecasting.ingestion.contracts import (
    IncrementalM1Provider,
    M1IngestionBatch,
    M1Resource,
    RawArtifact,
    RevisionMode,
)


class IncrementalUpdateError(RuntimeError):
    """Raised when an incremental update violates provenance or scope."""


@dataclass(frozen=True, slots=True)
class IncrementalUpdateResult:
    """Acquired batches in deterministic resource order."""

    resources: tuple[M1Resource, ...]
    batches: tuple[M1IngestionBatch, ...]

    @property
    def artifacts(self) -> tuple[RawArtifact, ...]:
        return tuple(batch.archive for batch in self.batches)

    @property
    def downloaded_count(self) -> int:
        return sum(not artifact.reused for artifact in self.artifacts)

    @property
    def reused_count(self) -> int:
        return sum(artifact.reused for artifact in self.artifacts)


def update_m1_history(
    provider: IncrementalM1Provider,
    raw_directory: str | Path,
    *,
    period_start_utc: datetime,
    period_end_utc: datetime,
    reject_at_or_after_utc: datetime,
    allow_holdout_override: bool = False,
    ingested_at_utc: datetime | None = None,
    max_pages: int = 10_000,
) -> IncrementalUpdateResult:
    """Acquire every provider resource in a guarded half-open interval.

    The function deliberately refuses holdout overrides.  Re-running the same
    interval is idempotent when the adapter advertises and honours
    ``completed_resource_cache``: immutable artifacts are verified and reused,
    never appended or overwritten.
    """

    start = _require_utc(period_start_utc, name="period_start_utc")
    end = _require_utc(period_end_utc, name="period_end_utc")
    guard = _require_utc(reject_at_or_after_utc, name="reject_at_or_after_utc")
    if start >= end:
        raise IncrementalUpdateError("incremental update interval must be non-empty")
    if allow_holdout_override:
        raise IncrementalUpdateError("development holdout override is forbidden")
    if end > guard:
        raise IncrementalUpdateError(
            "incremental update end reaches beyond the development holdout guard"
        )
    if isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages < 1:
        raise ValueError("max_pages must be at least one")
    if ingested_at_utc is not None:
        _require_utc(ingested_at_utc, name="ingested_at_utc")

    resources = _collect_resources(
        provider,
        period_start_utc=start,
        period_end_utc=end,
        max_pages=max_pages,
    )
    _validate_resource_catalog(provider, resources, start=start, end=end, guard=guard)

    raw_root = Path(raw_directory).expanduser().resolve()
    batches: list[M1IngestionBatch] = []
    for resource in resources:
        batch = provider.ingest_resource(
            resource,
            raw_root,
            ingested_at_utc=ingested_at_utc,
            as_of_utc=min(end, guard),
        )
        _validate_batch(
            batch,
            resource=resource,
            revision_mode=provider.capabilities.revision_mode,
            raw_root=raw_root,
            start=start,
            end=end,
            guard=guard,
        )
        batches.append(batch)
    return IncrementalUpdateResult(resources=resources, batches=tuple(batches))


def _collect_resources(
    provider: IncrementalM1Provider,
    *,
    period_start_utc: datetime,
    period_end_utc: datetime,
    max_pages: int,
) -> tuple[M1Resource, ...]:
    resources: list[M1Resource] = []
    seen_cursors: set[str] = set()
    cursor: str | None = None
    for _page_number in range(1, max_pages + 1):
        page = provider.list_resource_page(
            period_start_utc=period_start_utc,
            period_end_utc=period_end_utc,
            cursor=cursor,
        )
        resources.extend(page.resources)
        next_cursor = page.next_cursor
        if provider.capabilities.pagination_mode == "none" and next_cursor is not None:
            raise IncrementalUpdateError(
                "provider advertises no pagination but returned a next cursor"
            )
        if next_cursor is None:
            break
        if next_cursor in seen_cursors:
            raise IncrementalUpdateError("provider resource pagination repeated a cursor")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    else:
        raise IncrementalUpdateError(f"provider resource catalogue exceeded {max_pages} pages")

    ordered = tuple(sorted(resources, key=lambda item: (item.period_start_utc, item.resource_id)))
    ids = [resource.resource_id for resource in ordered]
    if len(ids) != len(set(ids)):
        raise IncrementalUpdateError("provider resource catalogue contains duplicate ids")
    return ordered


def _validate_resource_catalog(
    provider: IncrementalM1Provider,
    resources: tuple[M1Resource, ...],
    *,
    start: datetime,
    end: datetime,
    guard: datetime,
) -> None:
    previous_end: datetime | None = None
    for resource in resources:
        if resource.period_start_utc < start or resource.period_end_utc > end:
            raise IncrementalUpdateError(
                f"provider resource falls outside the requested interval: {resource.resource_id}"
            )
        if resource.period_end_utc > guard:
            raise IncrementalUpdateError(
                f"provider resource reaches the development holdout: {resource.resource_id}"
            )
        if previous_end is not None and resource.period_start_utc < previous_end:
            raise IncrementalUpdateError("provider resource catalogue contains overlapping periods")
        previous_end = resource.period_end_utc

        if provider.capabilities.annual_immutable_resources:
            expected_start = datetime(resource.period_start_utc.year, 1, 1, tzinfo=UTC)
            expected_end = datetime(resource.period_start_utc.year + 1, 1, 1, tzinfo=UTC)
            if (
                resource.period_start_utc != expected_start
                or resource.period_end_utc != expected_end
                or not resource.immutable
            ):
                raise IncrementalUpdateError(
                    "provider advertised annual immutable resources but returned a different shape"
                )


def _validate_batch(
    batch: M1IngestionBatch,
    *,
    resource: M1Resource,
    revision_mode: RevisionMode,
    raw_root: Path,
    start: datetime,
    end: datetime,
    guard: datetime,
) -> None:
    artifact = batch.archive
    try:
        path = artifact.path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise IncrementalUpdateError(f"raw artifact is not readable: {artifact.path}") from exc
    if not path.is_file() or not path.is_relative_to(raw_root):
        raise IncrementalUpdateError(f"raw artifact escaped the configured raw directory: {path}")
    if artifact.size_bytes != path.stat().st_size:
        raise IncrementalUpdateError(f"raw artifact size differs from provenance: {path}")
    actual_hash = _sha256_path(path)
    if artifact.sha256 != actual_hash:
        raise IncrementalUpdateError(f"raw artifact hash differs from provenance: {path}")
    if not artifact.provider_revision.strip():
        raise IncrementalUpdateError("raw artifact provider revision must not be empty")
    if revision_mode == "content_sha256" and artifact.provider_revision != f"sha256:{actual_hash}":
        raise IncrementalUpdateError("raw artifact content revision differs from its hash")
    if artifact.year != resource.period_start_utc.year:
        raise IncrementalUpdateError("raw artifact year differs from requested resource")
    if artifact.timeframe != "1min":
        raise IncrementalUpdateError("raw artifact timeframe must be 1min")
    _require_utc(artifact.observed_at_utc, name="archive.observed_at_utc")
    _require_utc(artifact.ingested_at_utc, name="archive.ingested_at_utc")

    candles = batch.candles
    if not isinstance(candles, pd.DataFrame):
        raise IncrementalUpdateError("ingestion batch candles must be a pandas DataFrame")
    if candles.empty:
        return
    for column in ("timestamp_open_utc", "timestamp_close_utc"):
        if column not in candles:
            raise IncrementalUpdateError(f"ingestion batch lacks {column}")
        timestamps = candles[column]
        if (
            not isinstance(timestamps.dtype, pd.DatetimeTZDtype)
            or str(timestamps.dt.tz) not in {"UTC", "UTC+00:00"}
            or bool(timestamps.isna().any())
        ):
            raise IncrementalUpdateError(f"ingestion batch {column} must contain finite UTC times")
    open_times = candles["timestamp_open_utc"]
    close_times = candles["timestamp_close_utc"]
    if bool((close_times > pd.Timestamp(guard)).any()):
        raise IncrementalUpdateError("ingestion batch contains candles beyond the holdout guard")
    if bool(((open_times < pd.Timestamp(start)) | (close_times > pd.Timestamp(end))).any()):
        raise IncrementalUpdateError(
            "ingestion batch contains candles outside the requested interval"
        )
    if bool(
        (
            (open_times < pd.Timestamp(resource.period_start_utc))
            | (close_times > pd.Timestamp(resource.period_end_utc))
        ).any()
    ):
        raise IncrementalUpdateError(
            "ingestion batch contains candles outside its resource interval"
        )
    if not bool((close_times - open_times).eq(pd.Timedelta(minutes=1)).all()):
        raise IncrementalUpdateError("ingestion batch must contain one-minute candles")
    if not open_times.is_monotonic_increasing or bool(open_times.duplicated().any()):
        raise IncrementalUpdateError(
            "ingestion batch must contain unique ordered candle timestamps"
        )
    for column, expected in (
        ("is_complete", True),
        ("timeframe", artifact.timeframe),
        ("source", artifact.source),
    ):
        if column not in candles or bool(candles[column].isna().any()):
            raise IncrementalUpdateError(f"ingestion batch lacks valid {column}")
        if not bool(candles[column].eq(expected).all()):
            raise IncrementalUpdateError(f"ingestion batch {column} differs from raw provenance")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise IncrementalUpdateError(f"cannot hash raw artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _require_utc(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise IncrementalUpdateError(f"{name} must be timezone-aware UTC")
    return value


__all__ = ["IncrementalUpdateError", "IncrementalUpdateResult", "update_m1_history"]
