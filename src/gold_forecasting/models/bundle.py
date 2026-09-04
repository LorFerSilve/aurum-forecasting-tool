"""Hash-verifiable model bundle shared by evaluation and inference."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from gold_forecasting.artifacts import content_version, sha256_file, write_json_atomic
from gold_forecasting.classification import CLASS_ORDER, PredictionBatch
from gold_forecasting.datasets.preprocessing import TrainOnlyPreprocessor
from gold_forecasting.models.logistic import LogisticSelectionResult, predict_logistic


class ModelBundleError(ValueError):
    """Raised when a model artifact or its metadata is inconsistent."""


@dataclass(slots=True)
class MVPModelBundle:
    """All fitted state and schemas needed for repeatable predictions."""

    estimator: LogisticRegression
    preprocessor: TrainOnlyPreprocessor
    feature_names: tuple[str, ...]
    class_order: tuple[str, ...]
    release_version: str
    model_version: str
    data_version: str
    feature_spec_version: str
    label_spec_version: str
    split_spec_version: str
    preprocessor_version: str
    config_snapshot: dict[str, Any]
    selected_c: float
    selected_class_weight: str | None
    seed: int

    def predict(self, frame: pd.DataFrame) -> PredictionBatch:
        if self.feature_names != self.preprocessor.feature_names:
            raise ModelBundleError("bundle and preprocessor feature orders differ")
        transformed = self.preprocessor.transform(frame)
        return predict_logistic(self.estimator, transformed)


def _stable_model_payload(
    selection: LogisticSelectionResult,
    preprocessor: TrainOnlyPreprocessor,
    *,
    release_version: str,
    data_version: str,
    feature_spec_version: str,
    label_spec_version: str,
    split_spec_version: str,
    preprocessor_version: str,
    config_snapshot: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    estimator = selection.estimator
    return {
        "schema_version": 1,
        "release_version": release_version,
        "data_version": data_version,
        "feature_spec_version": feature_spec_version,
        "label_spec_version": label_spec_version,
        "split_spec_version": split_spec_version,
        "preprocessor_version": preprocessor_version,
        "config_snapshot": config_snapshot,
        "feature_names": list(preprocessor.feature_names),
        "class_order": list(CLASS_ORDER),
        "selected_c": selection.selected_candidate.c,
        "selected_class_weight": selection.selected_candidate.class_weight,
        "seed": seed,
        "train_sample_digest": preprocessor.train_sample_digest,
        "estimator_classes": [str(value) for value in estimator.classes_],
        "coefficients": np.asarray(estimator.coef_, dtype=np.float64).tolist(),
        "intercept": np.asarray(estimator.intercept_, dtype=np.float64).tolist(),
    }


def build_model_bundle(
    selection: LogisticSelectionResult,
    preprocessor: TrainOnlyPreprocessor,
    *,
    release_version: str,
    data_version: str,
    feature_spec_version: str,
    label_spec_version: str,
    split_spec_version: str,
    preprocessor_version: str,
    config_snapshot: dict[str, Any],
    seed: int,
) -> MVPModelBundle:
    """Freeze a selected train-only estimator into a content-versioned bundle."""

    if tuple(str(value) for value in selection.estimator.classes_) != CLASS_ORDER:
        raise ModelBundleError("estimator classes do not match the fixed class order")
    stable = _stable_model_payload(
        selection,
        preprocessor,
        release_version=release_version,
        data_version=data_version,
        feature_spec_version=feature_spec_version,
        label_spec_version=label_spec_version,
        split_spec_version=split_spec_version,
        preprocessor_version=preprocessor_version,
        config_snapshot=config_snapshot,
        seed=seed,
    )
    version_hash = content_version(stable).removeprefix("sha256:")
    return MVPModelBundle(
        estimator=selection.estimator,
        preprocessor=preprocessor,
        feature_names=preprocessor.feature_names,
        class_order=CLASS_ORDER,
        release_version=release_version,
        model_version=f"{release_version}+{version_hash[:12]}",
        data_version=data_version,
        feature_spec_version=feature_spec_version,
        label_spec_version=label_spec_version,
        split_spec_version=split_spec_version,
        preprocessor_version=preprocessor_version,
        config_snapshot=config_snapshot,
        selected_c=selection.selected_candidate.c,
        selected_class_weight=selection.selected_candidate.class_weight,
        seed=seed,
    )


def _atomic_joblib_dump(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        joblib.dump(value, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_model_bundle(bundle: MVPModelBundle, models_root: str | Path) -> Path:
    """Persist the model, schema, and immutable hash metadata atomically."""

    root = Path(models_root)
    directory = root / bundle.model_version
    directory.mkdir(parents=True, exist_ok=True)
    bundle_path = directory / "bundle.joblib"
    schema_path = directory / "feature_schema.json"
    config_path = directory / "config_snapshot.json"
    manifest_path = directory / "manifest.json"
    _atomic_joblib_dump(bundle, bundle_path)
    bundle_hash = sha256_file(bundle_path)
    write_json_atomic(
        schema_path,
        {
            "schema_version": 1,
            "feature_names": list(bundle.feature_names),
            "class_order": list(bundle.class_order),
        },
    )
    write_json_atomic(config_path, bundle.config_snapshot)
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "model_version": bundle.model_version,
            "release_version": bundle.release_version,
            "bundle": bundle_path.name,
            "bundle_sha256": bundle_hash,
            "feature_schema": schema_path.name,
            "feature_schema_sha256": sha256_file(schema_path),
            "config_snapshot": config_path.name,
            "config_snapshot_sha256": sha256_file(config_path),
            "data_version": bundle.data_version,
            "feature_spec_version": bundle.feature_spec_version,
            "label_spec_version": bundle.label_spec_version,
            "split_spec_version": bundle.split_spec_version,
            "preprocessor_version": bundle.preprocessor_version,
            "selected_c": bundle.selected_c,
            "selected_class_weight": bundle.selected_class_weight,
            "seed": bundle.seed,
            "calibration_status": "preliminary",
        },
    )
    write_json_atomic(
        root / "latest.json",
        {
            "schema_version": 1,
            "model_version": bundle.model_version,
            "directory": directory.name,
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    return directory


def _local_artifact_path(
    model_directory: Path,
    value: object,
    *,
    field_name: str,
) -> Path:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ModelBundleError(f"model manifest contains an unsafe {field_name}")
    return model_directory / value


def load_model_bundle(directory: str | Path) -> MVPModelBundle:
    """Verify hashes before deserializing a locally created model bundle."""

    model_directory = Path(directory)
    manifest_path = model_directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBundleError(f"cannot read model manifest: {manifest_path}") from exc
    if manifest.get("schema_version") != 1:
        raise ModelBundleError("model manifest has an unsupported schema version")
    bundle_path = _local_artifact_path(
        model_directory,
        manifest.get("bundle"),
        field_name="bundle path",
    )
    schema_path = _local_artifact_path(
        model_directory,
        manifest.get("feature_schema"),
        field_name="feature schema path",
    )
    config_path = _local_artifact_path(
        model_directory,
        manifest.get("config_snapshot"),
        field_name="config snapshot path",
    )
    if (
        not bundle_path.is_file()
        or sha256_file(bundle_path) != manifest.get("bundle_sha256")
        or not schema_path.is_file()
        or sha256_file(schema_path) != manifest.get("feature_schema_sha256")
        or not config_path.is_file()
        or sha256_file(config_path) != manifest.get("config_snapshot_sha256")
    ):
        raise ModelBundleError("model bundle, feature schema, or config hash does not match")
    loaded = joblib.load(bundle_path)
    if not isinstance(loaded, MVPModelBundle):
        raise ModelBundleError("model bundle contains an unexpected object")
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBundleError("model feature schema is invalid") from exc
    try:
        config_snapshot = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBundleError("model config snapshot is invalid") from exc
    if not isinstance(config_snapshot, dict):
        raise ModelBundleError("model config snapshot must be a JSON object")
    if (
        loaded.model_version != manifest.get("model_version")
        or loaded.release_version != manifest.get("release_version")
        or loaded.data_version != manifest.get("data_version")
        or loaded.feature_spec_version != manifest.get("feature_spec_version")
        or loaded.label_spec_version != manifest.get("label_spec_version")
        or loaded.split_spec_version != manifest.get("split_spec_version")
        or loaded.preprocessor_version != manifest.get("preprocessor_version")
        or loaded.selected_c != manifest.get("selected_c")
        or loaded.selected_class_weight != manifest.get("selected_class_weight")
        or loaded.seed != manifest.get("seed")
        or loaded.config_snapshot != config_snapshot
        or list(loaded.feature_names) != schema.get("feature_names")
        or list(loaded.class_order) != schema.get("class_order")
        or loaded.class_order != CLASS_ORDER
        or schema.get("schema_version") != 1
        or loaded.feature_names != loaded.preprocessor.feature_names
    ):
        raise ModelBundleError("model bundle fields differ from verified metadata")
    return loaded


__all__ = [
    "MVPModelBundle",
    "ModelBundleError",
    "build_model_bundle",
    "load_model_bundle",
    "save_model_bundle",
]
