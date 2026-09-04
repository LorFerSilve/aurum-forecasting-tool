"""Reproducible phase-3 feature, label, split, and model-table pipeline."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gold_forecasting.artifacts import (
    ArtifactLayer,
    DatasetManifest,
    FileDigest,
    build_dataset_manifest,
    content_version,
    file_digest,
    sha256_file,
    write_json_atomic,
    write_manifest_atomic,
    write_parquet_atomic,
)
from gold_forecasting.config import ProjectConfig, load_project_config
from gold_forecasting.data_pipeline import validate_existing_mvp_data
from gold_forecasting.datasets.preprocessing import (
    fit_train_preprocessor,
    load_preprocessor,
    sample_id_digest,
    save_preprocessor,
)
from gold_forecasting.features import (
    FeatureCatalog,
    build_feature_catalog,
    build_mvp_features,
)
from gold_forecasting.labels import build_mvp_labels


class DatasetBuildError(RuntimeError):
    """Raised when the phase-3 model table is missing, corrupt, or unsafe."""


MODEL_AUDIT_COLUMNS = (
    "sample_id",
    "instrument",
    "source",
    "prediction_time_utc",
    "feature_available_at_utc",
    "feature_window_start_utc",
    "source_candle_open_utc",
    "entry_time_utc",
    "label_end_time_utc",
    "horizon_minutes",
    "split",
    "data_version_1min",
    "data_version_3min",
    "feature_spec_version",
    "label_spec_version",
    "split_spec_version",
    "source_raw_file_hash",
    "source_dataset_version",
    "entry_raw_file_hash",
    "exit_raw_file_hash",
    "entry_dataset_version",
    "exit_dataset_version",
    "entry_bid_open",
    "exit_bid_open",
    "future_return_bps",
    "target_class",
    "target_class_id",
)
SAMPLE_INDEX_COLUMNS = MODEL_AUDIT_COLUMNS[:-5]


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    dataset_version: str
    row_counts: dict[str, int]
    feature_count: int
    model_table_path: Path
    sample_index_path: Path
    manifest_path: Path
    feature_catalog_path: Path
    preprocessor_path: Path
    preprocessor_metadata_path: Path


def _root(config: ProjectConfig) -> Path:
    return config.config_path.parent.parent


def _base(config: ProjectConfig) -> tuple[str, str]:
    return config.instrument.provider.id, config.instrument.instrument.id


def _curated_manifest_path(root: Path, config: ProjectConfig, timeframe: str) -> Path:
    source, instrument = _base(config)
    return root / "data" / "curated" / source / instrument / timeframe / "manifest.json"


def _load_curated(
    root: Path,
    config: ProjectConfig,
    timeframe: str,
) -> tuple[pd.DataFrame, DatasetManifest]:
    path = _curated_manifest_path(root, config, timeframe)
    try:
        manifest = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetBuildError(f"cannot load curated {timeframe} manifest: {path}") from exc
    frames = [pd.read_parquet(root / output.path) for output in manifest.outputs]
    if not frames:
        raise DatasetBuildError(f"curated {timeframe} manifest contains no outputs")
    return pd.concat(frames, ignore_index=True), manifest


def _timestamp_bounds(
    frame: pd.DataFrame,
    *,
    start_column: str,
    end_column: str,
) -> tuple[datetime | None, datetime | None]:
    if frame.empty:
        return None, None
    start = cast(pd.Timestamp, frame[start_column].min()).to_pydatetime()
    end = cast(pd.Timestamp, frame[end_column].max()).to_pydatetime()
    return start, end


def _label_spec(config: ProjectConfig) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "prediction": config.labels.prediction.model_dump(mode="json"),
        "return": config.labels.return_config.model_dump(mode="json"),
        "classes": config.labels.classes.model_dump(mode="json"),
        "invalid_sample": config.labels.invalid_sample.model_dump(mode="json"),
    }
    payload["label_spec_version"] = content_version(payload)
    return payload


def _split_spec(config: ProjectConfig) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "timezone": config.splits.timezone,
        "boundary_semantics": config.splits.boundary_semantics,
        "splits": config.splits.splits.model_dump(mode="json"),
        "purging": config.splits.purging.model_dump(mode="json"),
        "development_guard": config.splits.development_guard.model_dump(mode="json"),
        "preprocessing": config.splits.preprocessing.model_dump(mode="json"),
    }
    payload["split_spec_version"] = content_version(payload)
    return payload


def _assign_splits(frame: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    split = pd.Series(pd.NA, index=frame.index, dtype="string")
    gap = pd.Timedelta(minutes=config.splits.purging.gap_minutes)
    for name in ("train", "validation", "test"):
        interval = getattr(config.splits.splits, name)
        prediction_end = pd.Timestamp(interval.end) - gap
        belongs = (
            frame["prediction_time_utc"].ge(pd.Timestamp(interval.start))
            & frame["prediction_time_utc"].lt(prediction_end)
            & frame["label_end_time_utc"].lt(pd.Timestamp(interval.end))
        )
        split.loc[belongs] = name
    result = frame.loc[split.notna()].copy()
    result["split"] = split.loc[split.notna()].astype("string")
    if result.empty:
        raise DatasetBuildError("no labeled samples remain after chronological splitting")
    guard = pd.Timestamp(config.splits.development_guard.reject_at_or_after)
    if result["label_end_time_utc"].ge(guard).any():
        raise DatasetBuildError("model table would expose the reserved holdout")
    return result


def _sample_ids(frame: pd.DataFrame) -> pd.Series:
    timestamp = frame["prediction_time_utc"].dt.strftime("%Y%m%dT%H%M%S.%fZ")
    values = (
        frame["instrument"].astype(str)
        + "|"
        + frame["source"].astype(str)
        + "|"
        + timestamp
    )
    return pd.Series(values, index=frame.index, dtype="string")


def _write_derived_manifest(
    *,
    path: Path,
    layer: ArtifactLayer,
    source: str,
    instrument: str,
    timeframe: str,
    frame: pd.DataFrame,
    inputs: Sequence[FileDigest],
    outputs: list[Path],
    parameters: dict[str, object],
    root: Path,
    start_column: str,
    end_column: str,
) -> DatasetManifest:
    start, end = _timestamp_bounds(frame, start_column=start_column, end_column=end_column)
    manifest = build_dataset_manifest(
        layer=layer,
        source=source,
        instrument=instrument,
        timeframe=timeframe,
        row_count=len(frame),
        inputs=inputs,
        outputs=tuple(file_digest(output, relative_to=root) for output in outputs),
        parameters=parameters,
        period_start_utc=start,
        period_end_utc=end,
    )
    write_manifest_atomic(path, manifest)
    return manifest


def build_mvp_dataset(config_path: str | Path) -> DatasetBuildResult:
    """Build and persist the complete leakage-safe phase-3 model table."""

    config = load_project_config(config_path)
    validate_existing_mvp_data(config.config_path)
    root = _root(config)
    source, instrument = _base(config)
    one_minute, manifest_1min = _load_curated(root, config, "1min")
    three_minute, manifest_3min = _load_curated(root, config, "3min")

    feature_result = build_mvp_features(three_minute, config.features)
    candidates = feature_result.features.loc[
        :, ["instrument", "source", "prediction_time_utc"]
    ]
    label_result = build_mvp_labels(one_minute, candidates, config.labels)

    join_key = ["instrument", "source", "prediction_time_utc"]
    joined = feature_result.features.merge(
        label_result.labels,
        on=join_key,
        how="inner",
        validate="one_to_one",
    )
    split = _assign_splits(joined, config)
    split.insert(0, "sample_id", _sample_ids(split))
    if split["sample_id"].duplicated().any():
        raise DatasetBuildError("sample identifiers are not unique")

    label_spec = _label_spec(config)
    label_spec_version = cast(str, label_spec["label_spec_version"])
    split_spec = _split_spec(config)
    split_spec_version = cast(str, split_spec["split_spec_version"])
    split["data_version_1min"] = manifest_1min.dataset_version
    split["data_version_3min"] = manifest_3min.dataset_version
    split["feature_spec_version"] = feature_result.catalog.feature_spec_version
    split["label_spec_version"] = label_spec_version
    split["split_spec_version"] = split_spec_version

    model_table = split.loc[
        :, [*MODEL_AUDIT_COLUMNS, *feature_result.catalog.feature_names]
    ].sort_values("prediction_time_utc", kind="stable").reset_index(drop=True)
    sample_index = model_table.loc[:, list(SAMPLE_INDEX_COLUMNS)].copy()

    feature_root = root / "data" / "features" / source / instrument / "3min"
    label_root = root / "data" / "labels" / source / instrument / "15min"
    table_root = root / "data" / "model_tables" / "mvp"
    preprocessor_root = root / "models" / "preprocessors"
    feature_path = feature_root / "features.parquet"
    catalog_path = feature_root / "feature_catalog.json"
    label_path = label_root / "labels.parquet"
    label_spec_path = label_root / "label_spec.json"
    model_table_path = table_root / "model_table.parquet"
    sample_index_path = table_root / "sample_index.parquet"
    split_spec_path = table_root / "split_spec.json"
    preprocessor_path = preprocessor_root / "mvp.joblib"
    preprocessor_metadata_path = preprocessor_root / "mvp.metadata.json"

    write_parquet_atomic(feature_path, feature_result.features)
    write_json_atomic(catalog_path, feature_result.catalog.model_dump(mode="json"))
    write_parquet_atomic(label_path, label_result.labels)
    write_json_atomic(label_spec_path, label_spec)
    write_parquet_atomic(model_table_path, model_table)
    write_parquet_atomic(sample_index_path, sample_index)
    write_json_atomic(split_spec_path, split_spec)

    fitted = fit_train_preprocessor(model_table, feature_result.catalog.feature_names)
    preprocessor_version = save_preprocessor(
        fitted,
        preprocessor_path,
        preprocessor_metadata_path,
    )

    feature_manifest = _write_derived_manifest(
        path=feature_root / "manifest.json",
        layer="features",
        source=source,
        instrument=instrument,
        timeframe="3min",
        frame=feature_result.features,
        inputs=manifest_3min.outputs,
        outputs=[feature_path, catalog_path],
        parameters={
            "feature_spec_version": feature_result.catalog.feature_spec_version,
            "complete_contiguous_history_only": True,
        },
        root=root,
        start_column="prediction_time_utc",
        end_column="prediction_time_utc",
    )
    label_manifest = _write_derived_manifest(
        path=label_root / "manifest.json",
        layer="labels",
        source=source,
        instrument=instrument,
        timeframe="15min",
        frame=label_result.labels,
        inputs=manifest_1min.outputs,
        outputs=[label_path, label_spec_path],
        parameters={
            "label_spec_version": label_spec_version,
            "dropped_missing_path": label_result.dropped_missing_path,
        },
        root=root,
        start_column="prediction_time_utc",
        end_column="label_end_time_utc",
    )
    manifest_path = table_root / "manifest.json"
    model_manifest = _write_derived_manifest(
        path=manifest_path,
        layer="model_table",
        source=source,
        instrument=instrument,
        timeframe="3min",
        frame=model_table,
        inputs=(*feature_manifest.outputs, *label_manifest.outputs),
        outputs=[
            model_table_path,
            sample_index_path,
            split_spec_path,
            preprocessor_path,
            preprocessor_metadata_path,
        ],
        parameters={
            "feature_spec_version": feature_result.catalog.feature_spec_version,
            "label_spec_version": label_spec_version,
            "split_spec_version": split_spec_version,
            "preprocessor_version": preprocessor_version,
            "feature_names": list(feature_result.catalog.feature_names),
            "purge_gap_minutes": config.splits.purging.gap_minutes,
            "random_shuffle": False,
        },
        root=root,
        start_column="prediction_time_utc",
        end_column="label_end_time_utc",
    )
    counts = {
        name: int(model_table["split"].eq(name).sum())
        for name in ("train", "validation", "test")
    }
    return DatasetBuildResult(
        dataset_version=model_manifest.dataset_version,
        row_counts=counts,
        feature_count=len(feature_result.catalog.feature_names),
        model_table_path=model_table_path,
        sample_index_path=sample_index_path,
        manifest_path=manifest_path,
        feature_catalog_path=catalog_path,
        preprocessor_path=preprocessor_path,
        preprocessor_metadata_path=preprocessor_metadata_path,
    )


def load_feature_catalog(config_path: str | Path) -> FeatureCatalog:
    config = load_project_config(config_path)
    root = _root(config)
    source, instrument = _base(config)
    path = root / "data" / "features" / source / instrument / "3min" / "feature_catalog.json"
    try:
        return FeatureCatalog.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetBuildError(f"cannot load feature catalog: {path}") from exc


def load_mvp_model_table(
    config_path: str | Path,
) -> tuple[pd.DataFrame, DatasetManifest]:
    config = load_project_config(config_path)
    root = _root(config)
    path = root / "data" / "model_tables" / "mvp" / "manifest.json"
    try:
        manifest = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetBuildError(f"cannot load model-table manifest: {path}") from exc
    model_path = root / "data" / "model_tables" / "mvp" / "model_table.parquet"
    if not model_path.is_file():
        raise DatasetBuildError(f"missing model table: {model_path}")
    return pd.read_parquet(model_path), manifest


def _load_manifest_file(path: Path) -> DatasetManifest:
    try:
        manifest = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetBuildError(f"cannot load dataset manifest: {path}") from exc
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
        raise DatasetBuildError(f"manifest version is invalid: {path}")
    return manifest


def _verify_digest(root: Path, digest: FileDigest) -> None:
    path = (root / digest.path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise DatasetBuildError("dataset manifest path escapes project root") from exc
    if (
        not path.is_file()
        or path.stat().st_size != digest.size_bytes
        or sha256_file(path) != digest.sha256
    ):
        raise DatasetBuildError(f"dataset artifact hash mismatch: {path}")


def validate_mvp_dataset(config_path: str | Path) -> dict[str, int]:
    """Validate phase-3 hashes, schemas, current specs, splits, and fit provenance."""

    config = load_project_config(config_path)
    root = _root(config)
    validate_existing_mvp_data(config.config_path)
    source, instrument = _base(config)
    feature_root = root / "data" / "features" / source / instrument / "3min"
    label_root = root / "data" / "labels" / source / instrument / "15min"
    table_root = root / "data" / "model_tables" / "mvp"
    preprocessor_root = root / "models" / "preprocessors"

    manifest = _load_manifest_file(table_root / "manifest.json")
    feature_manifest = _load_manifest_file(feature_root / "manifest.json")
    label_manifest = _load_manifest_file(label_root / "manifest.json")
    identities = (
        (manifest, "model_table", "3min"),
        (feature_manifest, "features", "3min"),
        (label_manifest, "labels", "15min"),
    )
    for candidate, layer, timeframe in identities:
        if (
            candidate.layer != layer
            or candidate.source != source
            or candidate.instrument != instrument
            or candidate.timeframe != timeframe
        ):
            raise DatasetBuildError(f"{layer} manifest identity differs from config")

    curated_1min = _load_manifest_file(_curated_manifest_path(root, config, "1min"))
    curated_3min = _load_manifest_file(_curated_manifest_path(root, config, "3min"))
    if feature_manifest.inputs != curated_3min.outputs:
        raise DatasetBuildError("feature lineage does not match curated 3min data")
    if label_manifest.inputs != curated_1min.outputs:
        raise DatasetBuildError("label lineage does not match curated 1min data")
    if manifest.inputs != (*feature_manifest.outputs, *label_manifest.outputs):
        raise DatasetBuildError("model-table lineage does not match feature and label outputs")

    expected_feature_outputs = (
        (feature_root / "features.parquet").relative_to(root).as_posix(),
        (feature_root / "feature_catalog.json").relative_to(root).as_posix(),
    )
    expected_label_outputs = (
        (label_root / "labels.parquet").relative_to(root).as_posix(),
        (label_root / "label_spec.json").relative_to(root).as_posix(),
    )
    expected_model_outputs = tuple(
        path.relative_to(root).as_posix()
        for path in (
            table_root / "model_table.parquet",
            table_root / "sample_index.parquet",
            table_root / "split_spec.json",
            preprocessor_root / "mvp.joblib",
            preprocessor_root / "mvp.metadata.json",
        )
    )
    if tuple(item.path for item in feature_manifest.outputs) != expected_feature_outputs:
        raise DatasetBuildError("feature manifest has unexpected output paths")
    if tuple(item.path for item in label_manifest.outputs) != expected_label_outputs:
        raise DatasetBuildError("label manifest has unexpected output paths")
    if tuple(item.path for item in manifest.outputs) != expected_model_outputs:
        raise DatasetBuildError("model-table manifest has unexpected output paths")
    for digest in (*manifest.inputs, *manifest.outputs):
        _verify_digest(root, digest)

    table = pd.read_parquet(table_root / "model_table.parquet")
    if len(table) != manifest.row_count or table.empty:
        raise DatasetBuildError("model-table row count differs from manifest")
    catalog = load_feature_catalog(config.config_path)
    current_catalog = build_feature_catalog(config.features)
    if catalog != current_catalog:
        raise DatasetBuildError("stored feature catalog differs from current feature config")
    current_label_spec = _label_spec(config)
    current_label_version = cast(str, current_label_spec["label_spec_version"])
    current_split_spec = _split_spec(config)
    current_split_version = cast(str, current_split_spec["split_spec_version"])
    try:
        stored_label_spec = json.loads(
            (label_root / "label_spec.json").read_text(encoding="utf-8")
        )
        stored_split_spec = json.loads(
            (table_root / "split_spec.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetBuildError("cannot read label or split specification") from exc
    if stored_label_spec != current_label_spec:
        raise DatasetBuildError("stored label specification differs from current config")
    if stored_split_spec != current_split_spec:
        raise DatasetBuildError("stored split specification differs from current config")
    if feature_manifest.parameters.get("feature_spec_version") != catalog.feature_spec_version:
        raise DatasetBuildError("feature manifest differs from current feature config")
    if label_manifest.parameters.get("label_spec_version") != current_label_version:
        raise DatasetBuildError("label manifest differs from current label config")

    required = {
        *MODEL_AUDIT_COLUMNS,
        *catalog.feature_names,
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise DatasetBuildError(f"model table misses required columns: {', '.join(missing)}")
    if list(table.columns) != [*MODEL_AUDIT_COLUMNS, *catalog.feature_names]:
        raise DatasetBuildError("model-table column order differs from the frozen schema")
    invalid_order = (
        table["sample_id"].duplicated().any()
        or not table["prediction_time_utc"].is_monotonic_increasing
    )
    if invalid_order:
        raise DatasetBuildError("model-table sample keys are duplicated or not chronological")
    if not table["feature_available_at_utc"].le(table["prediction_time_utc"]).all():
        raise DatasetBuildError("model table contains a future feature")
    if not (
        table["entry_time_utc"].gt(table["prediction_time_utc"]).all()
        and table["label_end_time_utc"].gt(table["entry_time_utc"]).all()
    ):
        raise DatasetBuildError("model table contains invalid label timing")
    actual_start, actual_end = _timestamp_bounds(
        table,
        start_column="prediction_time_utc",
        end_column="label_end_time_utc",
    )
    if actual_start != manifest.period_start_utc or actual_end != manifest.period_end_utc:
        raise DatasetBuildError("model-table timestamp bounds differ from manifest")
    expected_versions = {
        "feature_spec_version": catalog.feature_spec_version,
        "label_spec_version": current_label_version,
        "split_spec_version": current_split_version,
    }
    for column, expected in expected_versions.items():
        if set(table[column].astype(str)) != {expected}:
            raise DatasetBuildError(f"model-table {column} differs from current config")
        if manifest.parameters.get(column) != expected:
            raise DatasetBuildError(f"model manifest {column} differs from current config")
    data_versions = {
        "data_version_1min": curated_1min.dataset_version,
        "data_version_3min": curated_3min.dataset_version,
    }
    for column, expected in data_versions.items():
        if set(table[column].astype(str)) != {expected}:
            raise DatasetBuildError(f"model-table {column} differs from current curated data")

    original_ids = table["sample_id"].tolist()
    original_splits = table["split"].astype(str).tolist()
    reassigned = _assign_splits(table, config)
    if (
        reassigned["sample_id"].tolist() != original_ids
        or reassigned["split"].astype(str).tolist() != original_splits
    ):
        raise DatasetBuildError("stored split assignment differs from current split config")
    matrix = table.loc[:, list(catalog.feature_names)].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise DatasetBuildError("model table contains non-finite model features")
    observed_splits = set(table["split"])
    if observed_splits != {"train", "validation", "test"}:
        raise DatasetBuildError(f"model table has invalid split coverage: {observed_splits}")

    preprocessor = load_preprocessor(
        preprocessor_root / "mvp.joblib",
        preprocessor_root / "mvp.metadata.json",
    )
    if preprocessor.feature_names != catalog.feature_names:
        raise DatasetBuildError("preprocessor feature order differs from catalog")
    train_rows = table.loc[table["split"].eq("train")]
    if (
        preprocessor.train_row_count != len(train_rows)
        or preprocessor.train_sample_digest != sample_id_digest(train_rows["sample_id"])
        or preprocessor.fit_split != "train"
    ):
        raise DatasetBuildError("preprocessor was not fitted on the current train rows only")
    expected_preprocessor_version = f"sha256:{sha256_file(preprocessor_root / 'mvp.joblib')}"
    if manifest.parameters.get("preprocessor_version") != expected_preprocessor_version:
        raise DatasetBuildError("model manifest has the wrong preprocessor version")

    sample_index = pd.read_parquet(table_root / "sample_index.parquet")
    expected_index = table.loc[:, list(SAMPLE_INDEX_COLUMNS)]
    if list(sample_index.columns) != list(SAMPLE_INDEX_COLUMNS):
        raise DatasetBuildError("sample-index schema differs from the frozen audit schema")
    if not sample_index.equals(expected_index):
        raise DatasetBuildError("sample index does not exactly match model-table audit rows")
    return {
        name: int(table["split"].eq(name).sum())
        for name in ("train", "validation", "test")
    }


__all__ = [
    "DatasetBuildError",
    "DatasetBuildResult",
    "build_mvp_dataset",
    "load_feature_catalog",
    "load_mvp_model_table",
    "validate_mvp_dataset",
]
