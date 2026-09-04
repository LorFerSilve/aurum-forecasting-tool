"""Leakage-safe phase-4 training and model selection orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from gold_forecasting.artifacts import DatasetManifest, sha256_file
from gold_forecasting.config import ProjectConfig, load_project_config
from gold_forecasting.datasets import (
    load_feature_catalog,
    load_mvp_model_table,
    load_preprocessor,
    validate_mvp_dataset,
)
from gold_forecasting.models import (
    LogisticSelectionResult,
    MVPModelBundle,
    build_model_bundle,
    load_model_bundle,
    save_model_bundle,
    select_logistic_candidate,
)
from gold_forecasting.models.logistic import ClassWeight


class TrainingError(RuntimeError):
    """Raised when a model cannot be trained from valid phase-3 artifacts."""


@dataclass(frozen=True, slots=True)
class TrainingResult:
    bundle: MVPModelBundle
    bundle_directory: Path
    selection: LogisticSelectionResult
    train_rows: int
    validation_rows: int


def _single_string_column(frame: pd.DataFrame, column: str) -> str:
    observed = set(frame[column].astype(str))
    if len(observed) != 1:
        raise TrainingError(f"model table must contain one {column}, observed {observed}")
    return str(next(iter(observed)))


def _project_config_snapshot(config: ProjectConfig) -> dict[str, Any]:
    """Return a path-independent snapshot plus hashes of every source YAML."""

    components = {
        "instrument": config.instrument,
        "features": config.features,
        "labels": config.labels,
        "costs": config.costs,
        "splits": config.splits,
        "model": config.model,
        "backtest": config.backtest,
    }
    source_files: dict[str, dict[str, str]] = {
        "root": {
            "filename": config.config_path.name,
            "sha256": sha256_file(config.config_path),
        }
    }
    for name, path in sorted(config.component_paths.items()):
        source_files[name] = {
            "filename": path.name,
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": 1,
        "root": config.root.model_dump(mode="json", by_alias=True),
        "components": {
            name: model.model_dump(mode="json", by_alias=True)
            for name, model in components.items()
        },
        "source_files": source_files,
    }


def _load_current_model_table_manifest(config: ProjectConfig) -> DatasetManifest:
    path = (
        config.config_path.parent.parent
        / "data"
        / "model_tables"
        / "mvp"
        / "manifest.json"
    )
    try:
        return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrainingError(f"cannot read current model-table manifest: {path}") from exc


def _required_manifest_parameter(manifest: DatasetManifest, name: str) -> str:
    value = manifest.parameters.get(name)
    if not isinstance(value, str):
        raise TrainingError(f"model-table manifest has no valid {name}")
    return value


def _assert_bundle_is_current(
    bundle: MVPModelBundle,
    config: ProjectConfig,
    manifest: DatasetManifest,
) -> None:
    catalog = load_feature_catalog(config.config_path)
    expected_feature_names = manifest.parameters.get("feature_names")
    if not isinstance(expected_feature_names, list) or not all(
        isinstance(value, str) for value in expected_feature_names
    ):
        raise TrainingError("model-table manifest has no valid feature_names")
    expected = {
        "release_version": config.model.release_version,
        "data_version": manifest.dataset_version,
        "feature_spec_version": _required_manifest_parameter(
            manifest, "feature_spec_version"
        ),
        "label_spec_version": _required_manifest_parameter(
            manifest, "label_spec_version"
        ),
        "split_spec_version": _required_manifest_parameter(
            manifest, "split_spec_version"
        ),
        "preprocessor_version": _required_manifest_parameter(
            manifest, "preprocessor_version"
        ),
        "seed": config.root.run.seed,
        "feature_names": tuple(expected_feature_names),
        "class_order": tuple(config.model.class_order),
        "config_snapshot": _project_config_snapshot(config),
    }
    stale_fields = [
        name for name, value in expected.items() if getattr(bundle, name) != value
    ]
    if tuple(expected_feature_names) != catalog.feature_names:
        stale_fields.append("feature_catalog")
    if stale_fields:
        raise TrainingError(
            "latest model is incompatible with the current validated project: "
            + ", ".join(stale_fields)
        )


def train_mvp_model(config_path: str | Path) -> TrainingResult:
    """Select logistic settings on validation and keep the final fit train-only."""

    config = load_project_config(config_path)
    validate_mvp_dataset(config.config_path)
    table, table_manifest = load_mvp_model_table(config.config_path)
    catalog = load_feature_catalog(config.config_path)
    root = config.config_path.parent.parent
    preprocessor = load_preprocessor(
        root / "models" / "preprocessors" / "mvp.joblib",
        root / "models" / "preprocessors" / "mvp.metadata.json",
    )
    train = table.loc[table["split"].eq("train")]
    validation = table.loc[table["split"].eq("validation")]
    if train.empty or validation.empty:
        raise TrainingError("model table must contain non-empty train and validation splits")
    x_train = preprocessor.transform(train)
    x_validation = preprocessor.transform(validation)
    class_weights: tuple[ClassWeight, ...] = tuple(
        None if value == "none" else "balanced"
        for value in config.model.logistic.class_weight_options
    )
    selection = select_logistic_candidate(
        x_train,
        train["target_class"],
        x_validation,
        validation["target_class"],
        c_values=tuple(config.model.logistic.c_values),
        class_weight_options=class_weights,
        seed=config.root.run.seed,
        max_iter=config.model.logistic.max_iter,
    )
    preprocessor_version = table_manifest.parameters.get("preprocessor_version")
    if not isinstance(preprocessor_version, str):
        raise TrainingError("model-table manifest has no preprocessor version")
    bundle = build_model_bundle(
        selection,
        preprocessor,
        release_version=config.model.release_version,
        data_version=table_manifest.dataset_version,
        feature_spec_version=_single_string_column(table, "feature_spec_version"),
        label_spec_version=_single_string_column(table, "label_spec_version"),
        split_spec_version=_single_string_column(table, "split_spec_version"),
        preprocessor_version=preprocessor_version,
        config_snapshot=_project_config_snapshot(config),
        seed=config.root.run.seed,
    )
    if bundle.feature_names != catalog.feature_names:
        raise TrainingError("trained bundle does not use the catalog feature order")
    directory = save_model_bundle(bundle, root / "models" / "mvp")
    reloaded = load_model_bundle(directory)
    expected = bundle.predict(validation.iloc[: min(128, len(validation))])
    actual = reloaded.predict(validation.iloc[: min(128, len(validation))])
    if (
        expected.predicted_class != actual.predicted_class
        or not (expected.probabilities == actual.probabilities).all()
    ):
        raise TrainingError("saved model probabilities changed after reload")
    return TrainingResult(
        bundle=reloaded,
        bundle_directory=directory,
        selection=selection,
        train_rows=len(train),
        validation_rows=len(validation),
    )


def load_latest_mvp_model(config_path: str | Path) -> MVPModelBundle:
    """Resolve the latest model and fail closed when current artifacts differ."""

    config = load_project_config(config_path)
    validate_mvp_dataset(config.config_path)
    table_manifest = _load_current_model_table_manifest(config)
    models_root = config.config_path.parent.parent / "models" / "mvp"
    pointer_path = models_root / "latest.json"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"cannot read latest model pointer: {pointer_path}") from exc
    directory_name = pointer.get("directory")
    if not isinstance(directory_name, str) or Path(directory_name).name != directory_name:
        raise TrainingError("latest model pointer contains an unsafe directory")
    directory = models_root / directory_name
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != pointer.get(
        "manifest_sha256"
    ):
        raise TrainingError("latest model manifest hash differs from its pointer")
    bundle = load_model_bundle(directory)
    _assert_bundle_is_current(bundle, config, table_manifest)
    return bundle


__all__ = [
    "TrainingError",
    "TrainingResult",
    "load_latest_mvp_model",
    "train_mvp_model",
]
