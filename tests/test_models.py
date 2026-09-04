from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gold_forecasting import training as training_module
from gold_forecasting.classification import CLASS_ORDER
from gold_forecasting.config import load_project_config
from gold_forecasting.datasets.preprocessing import fit_train_preprocessor
from gold_forecasting.models import (
    ModelBundleError,
    build_model_bundle,
    load_model_bundle,
    save_model_bundle,
    select_logistic_candidate,
)
from gold_forecasting.models.baselines import build_mvp_baseline_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _training_table() -> tuple[pd.DataFrame, tuple[str, ...]]:
    features = ("f1", "f2")
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(12)],
            "split": ["train"] * 9 + ["validation"] * 3,
            "f1": [-3.0, -2.0, -1.0, -0.1, 0.0, 0.1, 1.0, 2.0, 3.0, -2.5, 0.0, 2.5],
            "f2": [0.0, 0.2, -0.1, 1.0, 0.9, 1.1, -0.1, 0.1, 0.0, 0.0, 1.0, 0.0],
            "target_class": [
                "down",
                "down",
                "down",
                "neutral",
                "neutral",
                "neutral",
                "up",
                "up",
                "up",
                "down",
                "neutral",
                "up",
            ],
        }
    )
    return frame, features


def test_all_roadmap_baselines_cover_the_same_rows() -> None:
    features = pd.DataFrame(
        {
            "close_log_return_1_bps": [-1.0, 0.0, 1.0],
            "momentum_5_bps": [-7.0, 6.0, 7.0],
        }
    )

    predictions = build_mvp_baseline_predictions(
        ["down", "neutral", "neutral", "up"],
        features,
    )

    assert set(predictions) == {
        "most_frequent",
        "always_up",
        "always_down",
        "last_candle_direction",
        "momentum",
        "mean_reversion",
    }
    assert {batch.row_count for batch in predictions.values()} == {3}
    assert predictions["most_frequent"].predicted_class == ("neutral",) * 3
    assert predictions["last_candle_direction"].predicted_class == (
        "down",
        "neutral",
        "up",
    )
    assert predictions["momentum"].predicted_class == ("down", "neutral", "up")
    assert predictions["mean_reversion"].predicted_class == ("up", "neutral", "down")


def test_logistic_selection_is_train_only_and_reproducible() -> None:
    table, feature_names = _training_table()
    preprocessor = fit_train_preprocessor(table, feature_names)
    train = table.loc[table["split"].eq("train")]
    validation = table.loc[table["split"].eq("validation")]
    x_train = preprocessor.transform(train)
    x_validation = preprocessor.transform(validation)

    first = select_logistic_candidate(
        x_train,
        train["target_class"],
        x_validation,
        validation["target_class"],
        seed=42,
    )
    second = select_logistic_candidate(
        x_train,
        train["target_class"],
        x_validation,
        validation["target_class"],
        seed=42,
    )

    assert first.train_row_count == 9
    assert first.validation_row_count == 3
    assert len(first.candidate_scores) == 6
    np.testing.assert_array_equal(first.estimator.coef_, second.estimator.coef_)
    assert tuple(str(value) for value in first.estimator.classes_) == CLASS_ORDER


def test_saved_bundle_reloads_with_identical_probabilities_and_hash_guard(
    tmp_path: Path,
) -> None:
    table, feature_names = _training_table()
    preprocessor = fit_train_preprocessor(table, feature_names)
    train = table.loc[table["split"].eq("train")]
    validation = table.loc[table["split"].eq("validation")]
    selection = select_logistic_candidate(
        preprocessor.transform(train),
        train["target_class"],
        preprocessor.transform(validation),
        validation["target_class"],
        seed=42,
    )
    bundle = build_model_bundle(
        selection,
        preprocessor,
        release_version="v0.1.0",
        data_version="sha256:" + "a" * 64,
        feature_spec_version="sha256:" + "b" * 64,
        label_spec_version="sha256:" + "c" * 64,
        split_spec_version="sha256:" + "d" * 64,
        preprocessor_version="sha256:" + "e" * 64,
        config_snapshot={"schema_version": 1, "model": {"release": "v0.1.0"}},
        seed=42,
    )
    directory = save_model_bundle(bundle, tmp_path / "models")

    loaded = load_model_bundle(directory)

    expected = bundle.predict(validation)
    actual = loaded.predict(validation)
    np.testing.assert_array_equal(actual.probabilities, expected.probabilities)
    assert actual.predicted_class == expected.predicted_class
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config_snapshot"] == "config_snapshot.json"
    assert loaded.config_snapshot == {
        "schema_version": 1,
        "model": {"release": "v0.1.0"},
    }
    assert loaded.split_spec_version == "sha256:" + "d" * 64
    assert loaded.preprocessor_version == "sha256:" + "e" * 64

    bundle_path = directory / "bundle.joblib"
    bundle_path.write_bytes(bundle_path.read_bytes() + b"corrupt")
    with pytest.raises(ModelBundleError, match="hash"):
        load_model_bundle(directory)


def test_current_bundle_guard_rejects_stale_validated_artifact_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table, feature_names = _training_table()
    preprocessor = fit_train_preprocessor(table, feature_names)
    train = table.loc[table["split"].eq("train")]
    validation = table.loc[table["split"].eq("validation")]
    config = load_project_config(PROJECT_ROOT / "configs" / "mvp.yaml")
    selection = select_logistic_candidate(
        preprocessor.transform(train),
        train["target_class"],
        preprocessor.transform(validation),
        validation["target_class"],
        seed=config.root.run.seed,
    )
    versions = {
        "data_version": "sha256:" + "a" * 64,
        "feature_spec_version": "sha256:" + "b" * 64,
        "label_spec_version": "sha256:" + "c" * 64,
        "split_spec_version": "sha256:" + "d" * 64,
        "preprocessor_version": "sha256:" + "e" * 64,
    }
    bundle = build_model_bundle(
        selection,
        preprocessor,
        release_version=config.model.release_version,
        config_snapshot=training_module._project_config_snapshot(config),
        seed=config.root.run.seed,
        **versions,
    )
    manifest = SimpleNamespace(
        dataset_version=versions["data_version"],
        parameters={
            "feature_names": list(feature_names),
            **{name: value for name, value in versions.items() if name != "data_version"},
        },
    )
    monkeypatch.setattr(
        training_module,
        "load_feature_catalog",
        lambda _path: SimpleNamespace(feature_names=feature_names),
    )

    training_module._assert_bundle_is_current(bundle, config, manifest)
    bundle.data_version = "sha256:" + "f" * 64

    with pytest.raises(training_module.TrainingError, match="data_version"):
        training_module._assert_bundle_is_current(bundle, config, manifest)
