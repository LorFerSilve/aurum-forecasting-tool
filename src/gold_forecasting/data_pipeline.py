"""Reproducible market-data builds shared by the MVP and hardened data profile."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import pandas as pd

from gold_forecasting.artifacts import (
    DatasetManifest,
    FileDigest,
    build_dataset_manifest,
    file_digest,
    sha256_file,
    write_json_atomic,
    write_manifest_atomic,
    write_parquet_atomic,
    write_text_atomic,
)
from gold_forecasting.config import ProjectConfig, load_project_config
from gold_forecasting.ingestion import (
    IncrementalM1Provider,
    M1IngestionBatch,
    M1Provider,
    ProviderConfigurationError,
    create_m1_provider,
    update_m1_history,
)
from gold_forecasting.resampling import (
    RESAMPLING_LOGIC_VERSION,
    resample_candles,
    resample_timeframes,
)
from gold_forecasting.validation import validate_candles


class DataBuildError(RuntimeError):
    """Raised when the reproducible MVP data build cannot complete."""


@dataclass(frozen=True, slots=True)
class DataBuildResult:
    dataset_version: str
    years: tuple[int, ...]
    row_counts: dict[str, int]
    raw_manifest_path: Path
    curated_manifest_paths: dict[str, Path]
    coverage_report_path: Path
    gap_report_path: Path
    quality_report_path: Path | None = None


RAW_LOGIC_VERSION = "1.0.0"
CURATED_LOGIC_VERSION = "2.0.0"


def _timeframes(config: ProjectConfig) -> tuple[str, ...]:
    return ("1min", *config.instrument.data.derived_timeframes)


def _is_hardened(config: ProjectConfig) -> bool:
    return set(config.instrument.data.derived_timeframes) != {"3min", "15min"}


def _validate_data_policy(config: ProjectConfig) -> None:
    if _is_hardened(config) and (
        config.instrument.provider.market_hours_policy != "observed_source_rows"
    ):
        raise DataBuildError("hardened build requires the documented observed_source_rows policy")


def _quality_output_paths(root: Path) -> list[Path]:
    return [
        root / "reports" / "data" / name
        for name in (
            "phase5_quality_daily.parquet",
            "phase5_quality_gaps.parquet",
            "phase5_quality.json",
            "phase5_quality.md",
            "phase5_coverage.json",
            "phase5_coverage.md",
            "phase5_gaps_1min.parquet",
        )
    ]


_RAW_PROVIDER_PROVENANCE_FIELDS = (
    "adapter",
    "source_timezone_offset",
    "source_observes_dst",
    "timestamp_semantics",
    "price_side",
    "volume_reliable",
)


def _raw_provider_provenance(config: ProjectConfig) -> dict[str, object]:
    provider = config.instrument.provider
    return {
        field_name: getattr(provider, field_name) for field_name in _RAW_PROVIDER_PROVENANCE_FIELDS
    }


def _project_root(config: ProjectConfig) -> Path:
    return config.config_path.parent.parent


def _development_years(config: ProjectConfig) -> tuple[int, ...]:
    start = config.splits.splits.train.start
    end = config.splits.splits.test.end
    if (start.month, start.day, start.hour, start.minute, start.second) != (1, 1, 0, 0, 0):
        raise DataBuildError("the MVP download start must be aligned to a UTC calendar year")
    if (end.month, end.day, end.hour, end.minute, end.second) != (1, 1, 0, 0, 0):
        raise DataBuildError("the MVP download end must be aligned to a UTC calendar year")
    return tuple(range(start.year, end.year))


def _validate_requested_years(config: ProjectConfig, years: tuple[int, ...]) -> None:
    expected = set(_development_years(config))
    if not years:
        raise DataBuildError("at least one development year is required")
    if len(years) != len(set(years)):
        raise DataBuildError("requested years must be unique")
    outside = sorted(set(years) - expected)
    if outside:
        raise DataBuildError(
            f"years outside the development window or inside the holdout are forbidden: {outside}"
        )


def _acquisition_metadata_path(archive_path: Path) -> Path:
    return archive_path.with_suffix(".metadata.json")


def _stable_ingestion_time(result: M1IngestionBatch) -> datetime:
    path = _acquisition_metadata_path(result.archive.path)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["sha256"] != result.archive.sha256:
                raise DataBuildError(f"raw metadata hash does not match archive: {path}")
            timestamp = datetime.fromisoformat(
                payload["first_ingested_at_utc"].replace("Z", "+00:00")
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DataBuildError(f"invalid immutable raw metadata: {path}") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(timestamp):
            raise DataBuildError(f"raw metadata ingestion timestamp is not UTC: {path}")
        return timestamp

    timestamp = result.archive.ingested_at_utc.astimezone(UTC)
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "source": result.archive.source,
            "source_symbol": result.archive.source_symbol,
            "timeframe": result.archive.timeframe,
            "year": result.archive.year,
            "archive": result.archive.path.name,
            "sha256": result.archive.sha256,
            "size_bytes": result.archive.size_bytes,
            "first_ingested_at_utc": timestamp.isoformat().replace("+00:00", "Z"),
            "source_page_url": result.archive.source_page_url,
        },
    )
    return timestamp


def _path_component(value: str, *, field: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise DataBuildError(f"{field} is not a safe path component: {value!r}")
    return value


def _year_output_path(
    root: Path,
    config: ProjectConfig,
    timeframe: str,
    year: int,
) -> Path:
    source = _path_component(config.instrument.provider.id, field="provider.id")
    instrument = _path_component(config.instrument.instrument.id, field="instrument.id")
    return (
        root
        / "data"
        / "curated"
        / source
        / instrument
        / ("phase5" if _is_hardened(config) else "")
        / timeframe
        / f"{'utc_year' if _is_hardened(config) else 'source_year'}={year}"
        / "candles.parquet"
    )


def _manifest_path(
    root: Path,
    config: ProjectConfig,
    layer: str,
    timeframe: str,
) -> Path:
    provider = config.instrument.provider
    source = _path_component(provider.id, field="provider.id")
    symbol = (
        _path_component(provider.source_symbol, field="provider.source_symbol")
        if layer == "raw"
        else _path_component(config.instrument.instrument.id, field="instrument.id")
    )
    base = root / "data" / layer / source / symbol
    if _is_hardened(config) and layer == "curated":
        base = base / "phase5"
    filename = (
        "manifest_phase5.json" if _is_hardened(config) and layer == "raw" else "manifest.json"
    )
    return base / timeframe / filename


def _timestamp_bounds(frames: list[pd.DataFrame]) -> tuple[datetime | None, datetime | None]:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return None, None
    start = min(cast(pd.Timestamp, frame["timestamp_open_utc"].min()) for frame in non_empty)
    end = max(cast(pd.Timestamp, frame["timestamp_close_utc"].max()) for frame in non_empty)
    return start.to_pydatetime(), end.to_pydatetime()


def _write_layer_manifest(
    *,
    path: Path,
    root: Path,
    layer: Literal["raw", "curated", "features", "labels"],
    timeframe: str,
    row_count: int,
    inputs: tuple[FileDigest, ...],
    output_paths: list[Path],
    frames: list[pd.DataFrame],
    years: tuple[int, ...],
    parameters: dict[str, object],
    source: str,
    instrument: str,
) -> DatasetManifest:
    start, end = _timestamp_bounds(frames)
    outputs = tuple(file_digest(item, relative_to=root) for item in output_paths)
    manifest = build_dataset_manifest(
        layer=layer,
        source=source,
        instrument=instrument,
        timeframe=timeframe,
        row_count=row_count,
        inputs=inputs,
        outputs=outputs,
        parameters={"years": list(years), **parameters},
        period_start_utc=start,
        period_end_utc=end,
    )
    write_manifest_atomic(path, manifest)
    return manifest


def _validate_ingestion_batch(
    result: M1IngestionBatch,
    *,
    config: ProjectConfig,
    year: int,
    raw_root: Path,
) -> None:
    archive = result.archive
    provider = config.instrument.provider
    expected = {
        "source": (archive.source, provider.id),
        "source_symbol": (archive.source_symbol, provider.source_symbol),
        "timeframe": (archive.timeframe, config.instrument.data.raw_timeframe),
        "year": (archive.year, year),
    }
    mismatches = [
        f"{name}={observed!r} (expected {wanted!r})"
        for name, (observed, wanted) in expected.items()
        if observed != wanted
    ]
    if mismatches:
        raise DataBuildError(
            "provider batch violates configured contract: " + "; ".join(mismatches)
        )
    try:
        archive.path.resolve(strict=True).relative_to(raw_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise DataBuildError(
            f"provider raw artifact must be stored below {raw_root}: {archive.path}"
        ) from exc

    candles = result.candles
    candle_expectations = {
        "instrument": config.instrument.instrument.id,
        "timeframe": config.instrument.data.raw_timeframe,
        "source": provider.id,
    }
    for column, wanted in candle_expectations.items():
        if column not in candles.columns or set(candles[column].dropna()) != {wanted}:
            raise DataBuildError(f"provider candles must contain only {column}={wanted!r}")


def _coverage_markdown(report: dict[str, object]) -> str:
    timeframes = cast(dict[str, dict[str, object]], report["timeframes"])
    lines = [
        "# MVP-datadekking",
        "",
        f"Datasetversie: `{report['dataset_version']}`",
        "",
        "| Timeframe | Rijen | Eerste candle | Laatste candle |",
        "|---|---:|---|---|",
    ]
    for timeframe, item in timeframes.items():
        lines.append(f"| {timeframe} | {item['rows']} | {item['start']} | {item['end']} |")
    lines.extend(
        [
            "",
            f"Gedetecteerde 1min-gaten: **{report['gap_count_1min']}**.",
            "Gaten zijn gerapporteerd en niet geïnterpoleerd.",
            "",
            "De bron is bid-only. Het volumeveld is niet geschikt als modelinput.",
            "",
        ]
    )
    return "\n".join(lines)


def build_mvp_data(
    config_path: str | Path,
    *,
    provider: M1Provider | None = None,
    years: tuple[int, ...] | None = None,
) -> DataBuildResult:
    """Download, validate, resample, store, and manifest the MVP data."""

    config = load_project_config(config_path)
    selected_years = tuple(sorted(_development_years(config) if years is None else years))
    _validate_requested_years(config, selected_years)
    root = _project_root(config)
    raw_root = root / "data" / "raw"
    active_provider = provider or create_m1_provider(config.instrument.provider)
    source = config.instrument.provider.id
    instrument = config.instrument.instrument.id
    hardened = _is_hardened(config)
    timeframes = _timeframes(config)
    _validate_data_policy(config)

    raw_paths: list[Path] = []
    raw_sidecars: list[Path] = []
    frames_by_timeframe: dict[str, list[pd.DataFrame]] = {key: [] for key in timeframes}
    outputs_by_timeframe: dict[str, list[Path]] = {key: [] for key in frames_by_timeframe}

    for year in selected_years:
        initial = active_provider.ingest_year(year, raw_root)
        _validate_ingestion_batch(initial, config=config, year=year, raw_root=raw_root)
        ingestion_time = _stable_ingestion_time(initial)
        parsed = active_provider.ingest_year(
            year,
            raw_root,
            ingested_at_utc=ingestion_time,
        )
        _validate_ingestion_batch(parsed, config=config, year=year, raw_root=raw_root)
        raw_paths.append(parsed.archive.path)
        raw_sidecars.append(_acquisition_metadata_path(parsed.archive.path))

        validation = validate_candles(
            parsed.candles,
            expected_timeframe="1min",
            duplicate_policy="drop_identical",
        )
        one_minute = validation.candles
        development_start = pd.Timestamp(config.splits.splits.train.start)
        development_end = pd.Timestamp(config.splits.splits.test.end)
        in_development = one_minute["timestamp_open_utc"].between(
            development_start,
            development_end,
            inclusive="left",
        )
        if hardened:
            # A source-local year may spill into the next UTC year. A partial
            # UTC build must apply the same scope to storage, lineage and audits.
            in_development &= one_minute["timestamp_open_utc"].dt.year.isin(selected_years)
        one_minute = one_minute.loc[in_development].reset_index(drop=True)
        if one_minute.empty:
            raise DataBuildError(
                f"HistData source-year archive {year} has no rows in the development window"
            )

        frames_by_timeframe["1min"].append(one_minute)
        if not hardened:
            for timeframe in timeframes[1:]:
                frames_by_timeframe[timeframe].append(resample_candles(one_minute, timeframe))

    # Merge before phase-5 aggregation: a UTC window may cross source-file boundaries.
    combined_one_minute = pd.concat(frames_by_timeframe["1min"], ignore_index=True)
    combined_validation = validate_candles(combined_one_minute, expected_timeframe="1min")
    combined_one_minute = combined_validation.candles
    if hardened:
        derived = resample_timeframes(
            combined_one_minute,
            timeframes[1:],
            closed_through_utc=config.splits.splits.test.end,
        )
        for timeframe in timeframes:
            combined = combined_one_minute if timeframe == "1min" else derived[timeframe]
            frames_by_timeframe[timeframe] = [
                combined.loc[combined["timestamp_open_utc"].dt.year.eq(year)].reset_index(drop=True)
                for year in selected_years
            ]
        del derived
    for timeframe in timeframes:
        for year, frame in zip(selected_years, frames_by_timeframe[timeframe], strict=True):
            output_path = _year_output_path(root, config, timeframe, year)
            write_parquet_atomic(output_path, frame)
            outputs_by_timeframe[timeframe].append(output_path)

    raw_outputs = tuple(file_digest(item, relative_to=root) for item in [*raw_paths, *raw_sidecars])
    raw_manifest_path = _manifest_path(root, config, "raw", "1min")
    raw_start, raw_end = _timestamp_bounds(frames_by_timeframe["1min"])
    raw_manifest = build_dataset_manifest(
        layer="raw",
        source=source,
        instrument=instrument,
        timeframe="1min",
        row_count=sum(len(frame) for frame in frames_by_timeframe["1min"]),
        inputs=(),
        outputs=raw_outputs,
        parameters={
            "years": list(selected_years),
            "build_scope": (
                "complete" if selected_years == _development_years(config) else "partial"
            ),
            **_raw_provider_provenance(config),
            **(
                {
                    "raw_logic_version": RAW_LOGIC_VERSION,
                    "provider_revisions": {
                        item.name: f"sha256:{sha256_file(item)}" for item in raw_paths
                    },
                }
                if hardened
                else {}
            ),
        },
        period_start_utc=raw_start,
        period_end_utc=raw_end,
    )
    write_manifest_atomic(raw_manifest_path, raw_manifest)

    curated_manifests: dict[str, DatasetManifest] = {}
    curated_manifest_paths: dict[str, Path] = {}
    for timeframe in timeframes:
        inputs = (
            raw_outputs
            if timeframe == "1min"
            else tuple(file_digest(item, relative_to=root) for item in outputs_by_timeframe["1min"])
        )
        path = _manifest_path(root, config, "curated", timeframe)
        curated_manifest_paths[timeframe] = path
        curated_manifests[timeframe] = _write_layer_manifest(
            path=path,
            root=root,
            layer="curated",
            timeframe=timeframe,
            row_count=sum(len(frame) for frame in frames_by_timeframe[timeframe]),
            inputs=inputs,
            output_paths=outputs_by_timeframe[timeframe],
            frames=frames_by_timeframe[timeframe],
            years=selected_years,
            parameters={
                "build_scope": (
                    "complete" if selected_years == _development_years(config) else "partial"
                ),
                "complete_windows_only": True,
                "interpolate_gaps": False,
                "alignment_timezone": "UTC",
                **(
                    {
                        "curated_logic_version": CURATED_LOGIC_VERSION,
                        "resampling_logic_version": RESAMPLING_LOGIC_VERSION,
                        "partition_basis": "utc_open_year",
                        "calendar_policy": "observed-source-v1",
                        "completeness_policy": "dense_minutes_unknown_closures_not_certified",
                        "closed_through_utc": config.splits.splits.test.end.isoformat(),
                    }
                    if hardened
                    else {}
                ),
            },
            source=source,
            instrument=instrument,
        )

    gaps = combined_validation.gaps.to_frame()
    prefix = "phase5" if hardened else "mvp"
    gap_report_path = root / "reports" / "data" / f"{prefix}_gaps_1min.parquet"
    write_parquet_atomic(gap_report_path, gaps)

    timeframes_report: dict[str, dict[str, object]] = {}
    row_counts: dict[str, int] = {}
    for timeframe, frames in frames_by_timeframe.items():
        start, end = _timestamp_bounds(frames)
        row_counts[timeframe] = sum(len(frame) for frame in frames)
        timeframes_report[timeframe] = {
            "rows": row_counts[timeframe],
            "start": start.isoformat().replace("+00:00", "Z") if start else None,
            "end": end.isoformat().replace("+00:00", "Z") if end else None,
        }
    coverage: dict[str, object] = {
        "schema_version": 1,
        "instrument": instrument,
        "source": source,
        "years": list(selected_years),
        "dataset_version": curated_manifests["1min"].dataset_version,
        "timeframes": timeframes_report,
        "gap_count_1min": len(gaps),
        "missing_candles_1min": (int(gaps["missing_candles"].sum()) if not gaps.empty else 0),
        "gap_report": gap_report_path.relative_to(root).as_posix(),
    }
    coverage_json_path = root / "reports" / "data" / f"{prefix}_coverage.json"
    write_json_atomic(coverage_json_path, coverage)
    coverage_report_path = root / "reports" / "data" / f"{prefix}_coverage.md"
    write_text_atomic(coverage_report_path, _coverage_markdown(coverage))

    quality_report_path = None
    if hardened:
        from gold_forecasting.data_quality_pipeline import (
            QUALITY_LOGIC_VERSION,
            write_quality_artifacts,
        )

        quality_report_path, quality_outputs = write_quality_artifacts(
            config,
            frames_by_timeframe,
            years=selected_years,
        )
        # Publish last. A failed rebuild cannot authenticate mixed-generation outputs.
        write_json_atomic(
            _completion_path(root, config),
            {
                "schema_version": 1,
                "quality_logic_version": QUALITY_LOGIC_VERSION,
                "raw_dataset_version": raw_manifest.dataset_version,
                "curated_dataset_versions": {
                    key: value.dataset_version for key, value in curated_manifests.items()
                },
                "reports": [
                    file_digest(item, relative_to=root).model_dump(mode="json")
                    for item in [
                        *quality_outputs,
                        coverage_json_path,
                        coverage_report_path,
                        gap_report_path,
                    ]
                ],
            },
        )

    return DataBuildResult(
        dataset_version=curated_manifests["1min"].dataset_version,
        years=selected_years,
        row_counts=row_counts,
        raw_manifest_path=raw_manifest_path,
        curated_manifest_paths=curated_manifest_paths,
        coverage_report_path=coverage_report_path,
        gap_report_path=gap_report_path,
        quality_report_path=quality_report_path,
    )


def _completion_path(root: Path, config: ProjectConfig) -> Path:
    return _manifest_path(root, config, "curated", "1min").parent.parent / "phase5_build.json"


def update_mvp_data(config_path: str | Path) -> DataBuildResult:
    """Acquire missing immutable resources, then rebuild the configured data profile."""
    config = load_project_config(config_path)
    _validate_data_policy(config)
    provider = create_m1_provider(config.instrument.provider)
    if not isinstance(provider, IncrementalM1Provider):
        raise DataBuildError("configured provider does not support incremental acquisition")
    update_m1_history(
        provider,
        _project_root(config) / "data" / "raw",
        period_start_utc=config.splits.splits.train.start,
        period_end_utc=config.splits.splits.test.end,
        reject_at_or_after_utc=config.splits.development_guard.reject_at_or_after,
    )
    return build_mvp_data(config_path, provider=provider)


def _manifest_years(manifest: DatasetManifest) -> tuple[int, ...]:
    raw_years = manifest.parameters.get("years")
    if (
        not isinstance(raw_years, list)
        or not raw_years
        or any(not isinstance(year, int) or isinstance(year, bool) for year in raw_years)
    ):
        raise DataBuildError("manifest parameters.years must be a non-empty integer list")
    years = tuple(raw_years)
    if years != tuple(sorted(set(years))):
        raise DataBuildError("manifest parameters.years must be sorted and unique")
    return years


def _safe_manifest_artifact(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise DataBuildError(
            f"manifest artifact escapes the project root: {relative_path}"
        ) from exc
    return candidate


def _verify_file_digest(root: Path, digest: FileDigest) -> Path:
    path = _safe_manifest_artifact(root, digest.path)
    if not path.is_file():
        raise DataBuildError(f"missing manifest artifact: {path}")
    if path.stat().st_size != digest.size_bytes:
        raise DataBuildError(f"artifact size differs from manifest: {path}")
    if sha256_file(path) != digest.sha256:
        raise DataBuildError(f"artifact hash mismatch: {path}")
    return path


def _validate_manifest_identity(
    manifest: DatasetManifest,
    *,
    config: ProjectConfig,
    layer: Literal["raw", "curated", "features", "labels"],
    timeframe: str,
) -> None:
    expected_values = {
        "layer": (manifest.layer, layer),
        "source": (manifest.source, config.instrument.provider.id),
        "instrument": (manifest.instrument, config.instrument.instrument.id),
        "timeframe": (manifest.timeframe, timeframe),
    }
    mismatches = [
        f"{field}={observed!r} (expected {expected!r})"
        for field, (observed, expected) in expected_values.items()
        if observed != expected
    ]
    if mismatches:
        raise DataBuildError("manifest identity mismatch: " + "; ".join(mismatches))

    rebuilt = build_dataset_manifest(
        layer=manifest.layer,
        source=manifest.source,
        instrument=manifest.instrument,
        timeframe=manifest.timeframe,
        row_count=manifest.row_count,
        inputs=manifest.inputs,
        outputs=manifest.outputs,
        parameters=manifest.parameters,
        period_start_utc=manifest.period_start_utc,
        period_end_utc=manifest.period_end_utc,
        created_at_utc=manifest.created_at_utc,
    )
    if rebuilt.dataset_version != manifest.dataset_version:
        raise DataBuildError("manifest dataset_version does not match its canonical content")


def validate_existing_mvp_data(
    config_path: str | Path,
    *,
    allow_partial: bool = False,
) -> dict[str, int]:
    """Revalidate curated files, hashes, manifests, and development boundaries.

    Normal CLI and model runs require every configured development year.
    ``allow_partial`` exists only for explicit diagnostics and small offline
    integration fixtures; a partial build can never masquerade as the MVP.
    """

    config = load_project_config(config_path)
    _validate_data_policy(config)
    try:
        create_m1_provider(config.instrument.provider)
    except ProviderConfigurationError as exc:
        raise DataBuildError(f"active provider configuration is invalid: {exc}") from exc
    root = _project_root(config)
    counts: dict[str, int] = {}
    manifests: dict[str, DatasetManifest] = {}
    manifest_years: tuple[int, ...] | None = None
    for timeframe in _timeframes(config):
        manifest_path = _manifest_path(root, config, "curated", timeframe)
        if not manifest_path.is_file():
            raise DataBuildError(f"missing curated manifest: {manifest_path}")
        try:
            manifest = DatasetManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise DataBuildError(f"invalid curated manifest: {manifest_path}") from exc
        _validate_manifest_identity(
            manifest,
            config=config,
            layer="curated",
            timeframe=timeframe,
        )
        if _is_hardened(config):
            expected_logic = {
                "curated_logic_version": CURATED_LOGIC_VERSION,
                "resampling_logic_version": RESAMPLING_LOGIC_VERSION,
                "partition_basis": "utc_open_year",
                "calendar_policy": "observed-source-v1",
                "completeness_policy": "dense_minutes_unknown_closures_not_certified",
                "closed_through_utc": config.splits.splits.test.end.isoformat(),
            }
            if any(manifest.parameters.get(key) != value for key, value in expected_logic.items()):
                raise DataBuildError("curated data logic differs from the hardened configuration")
        years = _manifest_years(manifest)
        _validate_requested_years(config, years)
        if not allow_partial and years != _development_years(config):
            raise DataBuildError(
                f"partial data manifest is not a complete MVP build: observed years {years}"
            )
        if manifest.parameters.get("build_scope") != (
            "complete" if years == _development_years(config) else "partial"
        ):
            raise DataBuildError("manifest build_scope is inconsistent with its years")
        if manifest_years is None:
            manifest_years = years
        elif years != manifest_years:
            raise DataBuildError("curated timeframe manifests use different source years")

        expected_outputs = tuple(
            _year_output_path(root, config, timeframe, year).relative_to(root).as_posix()
            for year in years
        )
        observed_outputs = tuple(output.path for output in manifest.outputs)
        if observed_outputs != expected_outputs:
            raise DataBuildError(
                f"manifest output partitions are not the expected {timeframe} year files"
            )

        for input_digest in manifest.inputs:
            _verify_file_digest(root, input_digest)
        frames: list[pd.DataFrame] = []
        for output in manifest.outputs:
            path = _verify_file_digest(root, output)
            frames.append(pd.read_parquet(path))
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        validated = validate_candles(combined, expected_timeframe=timeframe)
        empty_optional = (
            _is_hardened(config)
            and timeframe not in {"1min", "3min", "15min"}
            and validated.candles.empty
        )
        if not empty_optional and set(validated.candles["instrument"]) != {
            config.instrument.instrument.id
        }:
            raise DataBuildError(f"curated {timeframe} instrument differs from config")
        if not empty_optional and set(validated.candles["source"]) != {
            config.instrument.provider.id
        }:
            raise DataBuildError(f"curated {timeframe} source differs from config")
        if len(validated.candles) != manifest.row_count:
            raise DataBuildError(f"row count differs from manifest for {timeframe}")
        actual_start, actual_end = _timestamp_bounds([validated.candles])
        if actual_start != manifest.period_start_utc or actual_end != manifest.period_end_utc:
            raise DataBuildError(f"timestamp bounds differ from manifest for {timeframe}")
        development_start = config.splits.splits.train.start
        development_end = config.splits.splits.test.end
        guard = config.splits.development_guard.reject_at_or_after
        if not empty_optional and (
            actual_start is None
            or actual_end is None
            or actual_start < development_start
            or actual_end > development_end
            or actual_end > guard
        ):
            raise DataBuildError(f"curated {timeframe} data violates development boundaries")
        counts[timeframe] = len(validated.candles)
        manifests[timeframe] = manifest

    raw_manifest_path = _manifest_path(root, config, "raw", "1min")
    if not raw_manifest_path.is_file():
        raise DataBuildError(f"missing raw manifest: {raw_manifest_path}")
    try:
        raw_manifest = DatasetManifest.model_validate_json(
            raw_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise DataBuildError(f"invalid raw manifest: {raw_manifest_path}") from exc
    _validate_manifest_identity(
        raw_manifest,
        config=config,
        layer="raw",
        timeframe="1min",
    )
    raw_years = _manifest_years(raw_manifest)
    _validate_requested_years(config, raw_years)
    expected_raw_scope = "complete" if raw_years == _development_years(config) else "partial"
    if raw_manifest.parameters.get("build_scope") != expected_raw_scope:
        raise DataBuildError("raw manifest build_scope is inconsistent with its years")
    if raw_years != manifest_years:
        raise DataBuildError("raw and curated manifests use different source years")
    expected_provider = _raw_provider_provenance(config)
    provider_mismatches = [
        f"{field_name}={raw_manifest.parameters.get(field_name)!r} (expected {expected!r})"
        for field_name, expected in expected_provider.items()
        if raw_manifest.parameters.get(field_name) != expected
    ]
    if provider_mismatches:
        raise DataBuildError(
            "raw manifest provider provenance mismatch: " + "; ".join(provider_mismatches)
        )
    if raw_manifest.outputs != manifests["1min"].inputs:
        raise DataBuildError("curated 1min lineage does not exactly match raw manifest outputs")
    if raw_manifest.row_count != manifests["1min"].row_count:
        raise DataBuildError("raw and curated 1min row counts differ")

    one_minute_outputs = manifests["1min"].outputs
    for timeframe in _timeframes(config)[1:]:
        if manifests[timeframe].inputs != one_minute_outputs:
            raise DataBuildError(
                f"curated {timeframe} lineage does not exactly match curated 1min outputs"
            )
    if _is_hardened(config):
        from gold_forecasting.data_quality_pipeline import QUALITY_LOGIC_VERSION

        if raw_manifest.parameters.get("raw_logic_version") != RAW_LOGIC_VERSION:
            raise DataBuildError("raw data logic differs from the hardened configuration")
        try:
            completion = json.loads(_completion_path(root, config).read_text(encoding="utf-8"))
            if (
                completion["schema_version"] != 1
                or completion["quality_logic_version"] != QUALITY_LOGIC_VERSION
            ):
                raise DataBuildError("hardened build quality logic is stale")
            if completion["raw_dataset_version"] != raw_manifest.dataset_version:
                raise DataBuildError("hardened build raw version is stale")
            if completion["curated_dataset_versions"] != {
                key: value.dataset_version for key, value in manifests.items()
            }:
                raise DataBuildError("hardened build curated versions are stale")
            reports = [FileDigest.model_validate(item) for item in completion["reports"]]
            if [item.path for item in reports] != [
                item.relative_to(root).as_posix() for item in _quality_output_paths(root)
            ]:
                raise DataBuildError("hardened build lacks the required quality report set")
            for item in reports:
                _verify_file_digest(root, item)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise DataBuildError("missing or invalid hardened build completion record") from exc
    return counts


__all__ = [
    "DataBuildError",
    "DataBuildResult",
    "build_mvp_data",
    "update_mvp_data",
    "validate_existing_mvp_data",
]
