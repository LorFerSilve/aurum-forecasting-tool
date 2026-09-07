"""Metadata-only preflight before opening any phase-6 market observations.

This guard limits allowed resources, not their undisclosed contents. The
existing full hash/row validation must still follow it. It never reads a ZIP,
Parquet file, or observation hash; acquisition/creation dates are not market
dates and may legitimately be later than the development window.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from gold_forecasting.artifacts import DatasetManifest
from gold_forecasting.config import ProjectConfig
from gold_forecasting.data_pipeline import (
    DataBuildError,
    _manifest_path,
    _timeframes,
    _validate_manifest_identity,
    _year_output_path,
)

DEVELOPMENT_START = datetime(2020, 1, 1, tzinfo=UTC)
DEVELOPMENT_END = datetime(2025, 1, 1, tzinfo=UTC)
DEVELOPMENT_YEARS = (2020, 2021, 2022, 2023, 2024)


class BenchmarkDataGuardError(ValueError):
    """A resource would violate the frozen phase-6 development-only scope."""


def _canonical_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or path.drive or ".." in path.parts or "\\" in relative:
        raise BenchmarkDataGuardError(f"non-canonical resource path: {relative}")
    candidate = root / path
    if not candidate.is_relative_to(root) or candidate.resolve() != candidate:
        raise BenchmarkDataGuardError(f"resource path is redirected or escapes project: {relative}")
    return candidate


def _load_manifest(
    root: Path, project: ProjectConfig, layer: Literal["raw", "curated"], timeframe: str
) -> DatasetManifest:
    manifest_path = _manifest_path(root, project, layer, timeframe)
    relative = manifest_path.relative_to(root).as_posix()
    path = _canonical_path(root, relative)
    try:
        manifest = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
        _validate_manifest_identity(manifest, config=project, layer=layer, timeframe=timeframe)
    except (OSError, ValueError) as exc:
        raise BenchmarkDataGuardError(f"invalid development manifest: {relative}") from exc
    years = manifest.parameters.get("years")
    if (
        not isinstance(years, list)
        or any(isinstance(year, bool) or not isinstance(year, int) for year in years)
        or tuple(years) != DEVELOPMENT_YEARS
        or manifest.parameters.get("build_scope") != "complete"
    ):
        raise BenchmarkDataGuardError("manifest must declare the complete 2020-2024 years")
    start, end = manifest.period_start_utc, manifest.period_end_utc
    empty_optional = layer == "curated" and timeframe not in {"1min", "3min", "15min"}
    if manifest.row_count == 0:
        if not empty_optional or start is not None or end is not None:
            raise BenchmarkDataGuardError("empty manifest has invalid scope or market bounds")
    elif (
        start is None
        or end is None
        or start < DEVELOPMENT_START
        or start >= end
        or end > DEVELOPMENT_END
    ):
        raise BenchmarkDataGuardError("manifest market bounds expose non-development observations")
    # Validate every referenced path before any downstream hashing. This also
    # rejects links redirecting an apparently allowed 2024 filename elsewhere.
    for digest in (*manifest.inputs, *manifest.outputs):
        _canonical_path(root, digest.path)
    return manifest


def preflight_development_inputs(project: ProjectConfig) -> None:
    """Require the frozen resource set using JSON metadata alone, fail closed.

    Call before ``validate_existing_mvp_data`` (which hashes and reads files),
    not merely before model-table construction. This protocol deliberately
    pins HistData; another provider requires a new reviewed resource policy.
    """
    start = project.splits.splits.train.start
    end = project.splits.splits.test.end
    guard = project.splits.development_guard
    if guard.allow_holdout_override is not False or guard.reject_at_or_after > DEVELOPMENT_END:
        raise BenchmarkDataGuardError("phase-6 holdout boundary cannot move beyond 2025-01-01")
    if start != DEVELOPMENT_START or end != DEVELOPMENT_END or end > guard.reject_at_or_after:
        raise BenchmarkDataGuardError("phase-6 inputs must cover exactly 2020-2024 development")
    provider = project.instrument.provider
    if (provider.id, provider.adapter, provider.source_symbol) != (
        "histdata",
        "histdata_ascii_m1",
        "XAUUSD",
    ):
        raise BenchmarkDataGuardError("phase-6 resource policy is frozen to HistData XAUUSD")
    root = project.config_path.parent.parent.resolve()
    try:
        raw = _load_manifest(root, project, "raw", "1min")
        curated = {tf: _load_manifest(root, project, "curated", tf) for tf in _timeframes(project)}
    except DataBuildError as exc:
        raise BenchmarkDataGuardError(str(exc)) from exc
    raw_directory = _manifest_path(root, project, "raw", "1min").parent
    archives = tuple(
        (raw_directory / f"HISTDATA_COM_ASCII_XAUUSD_M1_{year}.zip").relative_to(root).as_posix()
        for year in DEVELOPMENT_YEARS
    )
    sidecars = tuple(Path(name).with_suffix(".metadata.json").as_posix() for name in archives)
    expected_raw = (*archives, *sidecars)
    if raw.inputs or tuple(item.path for item in raw.outputs) != expected_raw:
        raise BenchmarkDataGuardError("raw manifest references unexpected development resources")
    revisions = raw.parameters.get("provider_revisions")
    if revisions is not None and (
        not isinstance(revisions, dict) or set(revisions) != {Path(name).name for name in archives}
    ):
        raise BenchmarkDataGuardError("raw provider revisions reference unexpected source years")
    for timeframe, manifest in curated.items():
        expected_outputs = tuple(
            _year_output_path(root, project, timeframe, year).relative_to(root).as_posix()
            for year in DEVELOPMENT_YEARS
        )
        if tuple(item.path for item in manifest.outputs) != expected_outputs:
            raise BenchmarkDataGuardError(
                f"curated {timeframe} references unexpected year partitions"
            )
        expected_inputs = raw.outputs if timeframe == "1min" else curated["1min"].outputs
        if manifest.inputs != expected_inputs:
            raise BenchmarkDataGuardError(
                f"curated {timeframe} input lineage differs from development"
            )
    for year, archive, sidecar in zip(DEVELOPMENT_YEARS, archives, sidecars, strict=True):
        try:
            metadata = json.loads(_canonical_path(root, sidecar).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BenchmarkDataGuardError(f"invalid acquisition metadata: {sidecar}") from exc
        expected = {
            "source": provider.id,
            "source_symbol": provider.source_symbol,
            "timeframe": "1min",
            "year": year,
            "archive": Path(archive).name,
        }
        if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in expected.items()):
            raise BenchmarkDataGuardError(
                f"acquisition metadata has unexpected source/year: {sidecar}"
            )


__all__ = ["BenchmarkDataGuardError", "preflight_development_inputs"]
