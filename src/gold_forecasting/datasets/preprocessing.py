"""Train-only preprocessing shared by training, evaluation, and inference."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from gold_forecasting.artifacts import sha256_file, write_json_atomic


class PreprocessingError(ValueError):
    """Raised when the train-only preprocessing contract is violated."""


def sample_id_digest(sample_ids: pd.Series) -> str:
    """Hash ordered sample identifiers for fit-provenance validation."""

    digest = hashlib.sha256()
    for value in sample_ids.astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


@dataclass(slots=True)
class TrainOnlyPreprocessor:
    """Fitted transformer with an immutable feature-order audit trail."""

    pipeline: Any
    feature_names: tuple[str, ...]
    train_row_count: int
    train_sample_digest: str
    fit_split: str = "train"

    def transform(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise PreprocessingError(f"missing model features: {', '.join(missing)}")
        matrix = frame.loc[:, list(self.feature_names)].to_numpy(dtype=np.float64)
        if np.isinf(matrix).any():
            raise PreprocessingError("model features must not contain infinite values")
        transformed = np.asarray(self.pipeline.transform(matrix), dtype=np.float64)
        if transformed.shape != matrix.shape or not np.isfinite(transformed).all():
            raise PreprocessingError("preprocessor produced an invalid feature matrix")
        return transformed


def fit_train_preprocessor(
    model_table: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> TrainOnlyPreprocessor:
    """Fit median imputation and scaling on rows explicitly marked train."""

    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise PreprocessingError("feature_names must be non-empty and unique")
    required = {"split", "sample_id", *feature_names}
    missing = sorted(required.difference(model_table.columns))
    if missing:
        raise PreprocessingError(f"model table misses columns: {', '.join(missing)}")
    train = model_table.loc[model_table["split"].eq("train")]
    if train.empty:
        raise PreprocessingError("model table contains no train rows")
    matrix = train.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
    if np.isinf(matrix).any():
        raise PreprocessingError("train features must not contain infinite values")
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    pipeline.fit(matrix)
    fitted = TrainOnlyPreprocessor(
        pipeline=pipeline,
        feature_names=feature_names,
        train_row_count=len(train),
        train_sample_digest=sample_id_digest(train["sample_id"]),
    )
    fitted.transform(train)
    return fitted


def save_preprocessor(
    preprocessor: TrainOnlyPreprocessor,
    bundle_path: str | Path,
    metadata_path: str | Path,
) -> str:
    """Atomically save a local preprocessor and its hash-verifiable metadata."""

    destination = Path(bundle_path)
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
        joblib.dump(preprocessor, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    digest = sha256_file(destination)
    write_json_atomic(
        metadata_path,
        {
            "schema_version": 1,
            "preprocessor_version": f"sha256:{digest}",
            "bundle_path": destination.name,
            "bundle_sha256": digest,
            "fit_split": preprocessor.fit_split,
            "feature_names": list(preprocessor.feature_names),
            "train_row_count": preprocessor.train_row_count,
            "train_sample_digest": preprocessor.train_sample_digest,
        },
    )
    return f"sha256:{digest}"


def load_preprocessor(
    bundle_path: str | Path,
    metadata_path: str | Path,
) -> TrainOnlyPreprocessor:
    """Hash-check and load a preprocessor created by this local project."""

    bundle = Path(bundle_path)
    metadata_file = Path(metadata_path)
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreprocessingError(f"cannot read preprocessor metadata: {metadata_file}") from exc
    expected_hash = metadata.get("bundle_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise PreprocessingError("preprocessor metadata contains no valid bundle hash")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("bundle_path") != bundle.name
        or metadata.get("fit_split") != "train"
        or metadata.get("preprocessor_version") != f"sha256:{expected_hash}"
    ):
        raise PreprocessingError("preprocessor metadata violates the artifact contract")
    if not bundle.is_file() or sha256_file(bundle) != expected_hash:
        raise PreprocessingError("preprocessor bundle is missing or its hash does not match")
    loaded = joblib.load(bundle)
    if not isinstance(loaded, TrainOnlyPreprocessor):
        raise PreprocessingError("preprocessor bundle contains an unexpected object")
    if (
        list(loaded.feature_names) != metadata.get("feature_names")
        or loaded.fit_split != "train"
        or loaded.train_row_count != metadata.get("train_row_count")
        or loaded.train_sample_digest != metadata.get("train_sample_digest")
    ):
        raise PreprocessingError("preprocessor bundle differs from its metadata")
    return loaded


__all__ = [
    "PreprocessingError",
    "TrainOnlyPreprocessor",
    "fit_train_preprocessor",
    "load_preprocessor",
    "sample_id_digest",
    "save_preprocessor",
]
