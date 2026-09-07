from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from gold_forecasting.benchmark.config import BenchmarkConfig, load_benchmark_config
from gold_forecasting.benchmark.models import (
    ALLOWED_FEATURE_NAMES,
    FittedModel,
    ModelSpec,
    candidate_specs,
    class_return_means,
    fit_model,
    prediction_records,
)
from gold_forecasting.classification import CLASS_ORDER, ClassificationContractError
from gold_forecasting.config import load_project_config
from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.features.mvp import build_feature_catalog

FEATURES = ("close_log_return_1_bps", "momentum_5_bps", "candle_range_bps")


def _frame(start: str = "2020-01-02T00:00:00Z", rows: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(20260906)
    classes = np.arange(rows, dtype=np.int64) % 3
    returns = (classes - 1) * 18.0 + rng.normal(0, 1, rows)
    prediction = pd.date_range(start, periods=rows, freq="3min")
    return pd.DataFrame(
        {
            "sample_id": [f"{start}-{index}" for index in range(rows)],
            "instrument": "XAU_USD",
            "source": "synthetic",
            "prediction_time_utc": prediction,
            "entry_time_utc": prediction + pd.Timedelta(minutes=1),
            "label_end_time_utc": prediction + pd.Timedelta(minutes=16),
            "feature_available_at_utc": prediction,
            "target_class_id": classes,
            "target_class": np.asarray(CLASS_ORDER)[classes],
            "arithmetic_return_bps": returns,
            FEATURES[0]: returns / 2 + rng.normal(0, 3, rows),
            FEATURES[1]: rng.normal(0, 4, rows),
            FEATURES[2]: rng.uniform(0, 2, rows),
        }
    )


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(xgb_max_rounds=8, xgb_early_stopping_rounds=2)


def _fit(family: str, train: pd.DataFrame | None = None) -> FittedModel:
    source = _frame() if train is None else train
    spec = ModelSpec(family, 2.0 if family == "xgboost" else 0.1)
    return fit_model(source, FEATURES, spec, _config(), rounds=4 if family == "xgboost" else None)


@pytest.mark.parametrize("family", ["reference", "logistic", "ridge", "xgboost"])
def test_models_return_finite_fixed_order_probabilities_and_train_only_provenance(
    family: str,
) -> None:
    train = _frame()
    test = _frame("2022-01-02T00:00:00Z", 21)
    model = _fit(family, train)
    probabilities, expected = model.predict(test)

    assert probabilities.shape == (len(test), 3)
    assert expected.shape == (len(test),)
    assert np.isfinite(expected).all()
    assert np.isfinite(probabilities).all()
    assert (probabilities > 0).all()
    assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert model.preprocessor.feature_names == FEATURES
    assert model.preprocessor.train_row_count == len(train)
    assert model.preprocessor.train_sample_digest == sample_id_digest(train["sample_id"])
    assert model.preprocessor.fit_split == "train"
    assert_allclose(model.class_returns, class_return_means(train))
    if family != "ridge":
        assert_array_equal(expected, probabilities @ model.class_returns)
        assert tuple(model.estimator.classes_) == (0, 1, 2)
    else:
        residuals = train["arithmetic_return_bps"].to_numpy() - model.estimator.predict(
            model.preprocessor.transform(train)
        )
        assert model.residual_scale == pytest.approx(max(float(np.std(residuals)), 1e-6))


@pytest.mark.parametrize("family", ["reference", "logistic", "ridge", "xgboost"])
def test_model_repeats_are_deterministic_and_preserve_input_frames(family: str) -> None:
    train = _frame()
    test = _frame("2022-01-02T00:00:00Z", 21)
    original_train, original_test = train.copy(deep=True), test.copy(deep=True)
    left = _fit(family, train)
    right = _fit(family, train)
    first, second = left.predict(test), right.predict(test)

    assert_array_equal(first[0], second[0])
    assert_array_equal(first[1], second[1])
    assert_frame_equal(train, original_train)
    assert_frame_equal(test, original_test)


@pytest.mark.parametrize("family", ["logistic", "ridge", "xgboost"])
def test_future_validation_mutation_cannot_fit_scaler_imputer_or_target_statistics(
    family: str,
) -> None:
    train = _frame()
    train.loc[::7, FEATURES[1]] = np.nan
    validation = _frame("2021-04-01T00:00:00Z", 30)
    changed = validation.copy(deep=True)
    changed.loc[:, list(FEATURES)] *= 1_000_000
    changed["arithmetic_return_bps"] *= -1_000
    changed["target_class_id"] = 2 - changed["target_class_id"]
    changed["target_class"] = np.asarray(CLASS_ORDER)[changed["target_class_id"]]
    spec = ModelSpec(family, 2.0 if family == "xgboost" else 0.1)
    left = fit_model(train, FEATURES, spec, _config(), validation=validation)
    right = fit_model(train, FEATURES, spec, _config(), validation=changed)
    left_imputer = left.preprocessor.pipeline.named_steps["imputer"]
    left_scaler = left.preprocessor.pipeline.named_steps["scaler"]
    right_imputer = right.preprocessor.pipeline.named_steps["imputer"]
    right_scaler = right.preprocessor.pipeline.named_steps["scaler"]

    assert_allclose(left_imputer.statistics_, train.loc[:, list(FEATURES)].median().to_numpy())
    assert_array_equal(left_imputer.statistics_, right_imputer.statistics_)
    assert_array_equal(left_scaler.mean_, right_scaler.mean_)
    assert_array_equal(left_scaler.scale_, right_scaler.scale_)
    assert_array_equal(left.class_returns, right.class_returns)
    assert left.preprocessor.train_sample_digest == right.preprocessor.train_sample_digest
    if family != "xgboost":
        # Linear fits never use validation labels or feature values at all.
        assert_array_equal(left.estimator.coef_, right.estimator.coef_)
        assert left.residual_scale == right.residual_scale


def test_prediction_features_ignore_mutated_future_target_columns_and_column_order() -> None:
    model = _fit("logistic")
    test = _frame("2022-01-02T00:00:00Z", 21)
    changed = test.copy(deep=True)
    changed["arithmetic_return_bps"] = np.inf
    changed["target_class_id"] = -100
    changed["target_class"] = "not a feature"
    changed["future_return_bps"] = -1e100
    changed = changed.loc[:, list(reversed(changed.columns))]

    original, mutated = model.predict(test), model.predict(changed)

    assert_array_equal(original[0], mutated[0])
    assert_array_equal(original[1], mutated[1])


def test_xgboost_early_stopping_and_final_refit_use_selected_round_count_only() -> None:
    train = _frame()
    validation = _frame("2021-04-01T00:00:00Z", 30)
    spec = ModelSpec("xgboost", 2.0)
    selected = fit_model(train, FEATURES, spec, _config(), validation=validation)

    assert selected.best_rounds is not None
    assert selected.best_rounds == selected.estimator.best_iteration + 1
    assert 1 <= selected.best_rounds <= _config().xgb_max_rounds
    final = fit_model(train, FEATURES, spec, _config(), rounds=selected.best_rounds)
    assert final.best_rounds == selected.best_rounds
    assert final.estimator.get_params()["n_estimators"] == selected.best_rounds
    assert final.estimator.get_booster().num_boosted_rounds() == selected.best_rounds
    assert final.estimator.get_params()["early_stopping_rounds"] is None
    assert getattr(final.estimator, "evals_result_", None) is None
    assert final.preprocessor.train_row_count == len(train)


@pytest.mark.parametrize("rounds", [None, 0, -1, 9, True, 1.5])
def test_final_xgboost_requires_a_valid_inner_selected_round_count(rounds: Any) -> None:
    with pytest.raises(ValueError, match="round"):
        fit_model(_frame(), FEATURES, ModelSpec("xgboost", 2.0), _config(), rounds=rounds)


def test_explicit_round_count_cannot_be_combined_with_validation_or_another_family() -> None:
    with pytest.raises(ValueError, match="rounds"):
        fit_model(
            _frame(),
            FEATURES,
            ModelSpec("xgboost", 2.0),
            _config(),
            validation=_frame("2021-04-01T00:00:00Z"),
            rounds=3,
        )
    with pytest.raises(ValueError, match="rounds"):
        fit_model(_frame(), FEATURES, ModelSpec("logistic", 0.1), _config(), rounds=3)


def test_classifier_probability_columns_are_restored_using_estimator_class_ids() -> None:
    class ReorderedEstimator:
        classes_ = np.array([2, 0, 1])

        def predict_proba(self, matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            return np.tile([0.6, 0.3, 0.1], (len(matrix), 1))

    model = _fit("logistic")
    model.estimator = ReorderedEstimator()
    probabilities, expected = model.predict(_frame("2022-01-02T00:00:00Z", 3))
    assert_allclose(probabilities, [[0.3, 0.1, 0.6]] * 3)
    assert_allclose(expected, probabilities @ model.class_returns)


def test_prediction_record_class_order_alignment_ties_and_input_immutability() -> None:
    frame = _frame(rows=3)
    before = frame.copy(deep=True)
    matrix = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.5, 0.5, 0.0]])
    expected = np.array([1.0, -2.0, 0.0])
    records = prediction_records(frame, matrix, expected)

    assert records["sample_id"].tolist() == frame["sample_id"].tolist()
    assert records["predicted_class"].tolist() == ["up", "down", "down"]
    assert_array_equal(records[["p_down", "p_neutral", "p_up"]].to_numpy(), matrix)
    assert_array_equal(records["expected_return_bps"].to_numpy(), expected)
    assert_frame_equal(frame, before)


@pytest.mark.parametrize("expected", [np.array([1.0]), np.array([np.nan] * 3)])
def test_prediction_records_reject_misaligned_or_nonfinite_expectations(expected: Any) -> None:
    with pytest.raises(ValueError, match="expected returns"):
        prediction_records(_frame(rows=3), np.full((3, 3), 1 / 3), expected)


def test_prediction_records_reject_bad_probability_column_count() -> None:
    with pytest.raises(ClassificationContractError, match="shape"):
        prediction_records(_frame(rows=3), np.full((3, 2), 0.5), np.zeros(3))


@pytest.mark.parametrize("family", ["logistic", "ridge", "xgboost"])
def test_direct_fitting_and_prediction_cannot_bypass_development_holdout(family: str) -> None:
    holdout = _frame("2025-01-02T00:00:00Z")
    spec = ModelSpec(family, 2.0 if family == "xgboost" else 0.1)
    with pytest.raises(ValueError, match="reserved holdout"):
        fit_model(holdout, FEATURES, spec, _config())
    with pytest.raises(ValueError, match="reserved holdout"):
        fit_model(_frame(), FEATURES, spec, _config(), validation=holdout)
    with pytest.raises(ValueError, match="reserved holdout"):
        _fit(family).predict(holdout)


def test_label_end_and_feature_availability_guards_apply_at_model_boundary() -> None:
    model = _fit("logistic")
    contaminated = _frame()
    contaminated.loc[0, "label_end_time_utc"] = pd.Timestamp("2025-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="reserved holdout"):
        _fit("logistic", contaminated)
    contaminated = _frame()
    contaminated["feature_available_at_utc"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="must not follow"):
        model.predict(contaminated)


def test_validation_cannot_overlap_labels_or_common_embargo() -> None:
    train = _frame()
    validation = _frame("2020-01-02T00:00:00Z")
    with pytest.raises(ValueError, match="overlap validation"):
        fit_model(train, FEATURES, ModelSpec("logistic", 0.1), _config(), validation=validation)
    after = train["prediction_time_utc"].max() + pd.Timedelta(minutes=181)
    validation = _frame(after.isoformat())
    with pytest.raises(ValueError, match="181-minute embargo"):
        fit_model(train, FEATURES, ModelSpec("logistic", 0.1), _config(), validation=validation)


@pytest.mark.parametrize("feature", ["target_class_id", "future_return_bps", "arbitrary_future"])
def test_feature_allowlist_rejects_target_and_unknown_columns(feature: str) -> None:
    train = _frame()
    train[feature] = 0.0
    # Do not overwrite class IDs when checking the allowlist itself.
    if feature == "target_class_id":
        train[feature] = np.arange(len(train), dtype=np.int64) % 3
    with pytest.raises(ValueError, match="causal MVP allowlist"):
        fit_model(train, (feature,), ModelSpec("logistic", 0.1), _config())


def test_allowlist_matches_frozen_feature_catalog() -> None:
    root = Path(__file__).resolve().parents[1]
    features = load_project_config(root / "configs" / "mvp.yaml").features
    assert frozenset(build_feature_catalog(features).feature_names) == ALLOWED_FEATURE_NAMES


def test_bad_training_classes_returns_sample_ids_and_split_markers_fail_closed() -> None:
    train = _frame()
    train.loc[0, "target_class_id"] = 1
    with pytest.raises(ValueError, match="fixed class order"):
        _fit("logistic", train)
    train = _frame()
    train.loc[0, "arithmetic_return_bps"] = np.nan
    with pytest.raises(ValueError, match="returns must be finite"):
        _fit("logistic", train)
    train = _frame()
    train.loc[0, "sample_id"] = train.loc[1, "sample_id"]
    with pytest.raises(ValueError, match="sample IDs"):
        _fit("logistic", train)
    with pytest.raises(ValueError, match="another split"):
        _fit("logistic", _frame().assign(split="test"))
    with pytest.raises(ValueError, match="all three"):
        _fit("logistic", _frame().loc[lambda frame: frame["target_class_id"].ne(2)])


@pytest.mark.parametrize(
    "family, values",
    [
        ("logistic", (0.1, 1.0, 10.0)),
        ("ridge", (0.1, 10.0, 1000.0)),
        ("xgboost", (2.0, 3.0, 4.0)),
    ],
)
def test_candidate_family_budget_and_order_are_frozen(
    family: str, values: tuple[float, ...]
) -> None:
    specs = candidate_specs(family)
    assert tuple(spec.value for spec in specs) == values
    assert tuple(spec.family for spec in specs) == (family,) * 3


@pytest.mark.parametrize(
    "family, value",
    [
        ("logistic", 0.0),
        ("logistic", np.nan),
        ("logistic", 100.0),
        ("xgboost", 2.5),
        ("reference", 1.0),
        ("unknown", 0.1),
        ("logistic", True),
    ],
)
def test_out_of_protocol_model_specifications_are_rejected(family: str, value: float) -> None:
    with pytest.raises(ValueError, match="frozen"):
        ModelSpec(family, value)


@pytest.mark.parametrize(
    "changes",
    [
        {"horizons": ()},
        {"horizons": (15, 3)},
        {"horizons": (3, 3)},
        {"horizons": (1440,)},
        {"horizons": (3.0,)},
        {"test_years": (2025,)},
        {"test_years": (2023, 2022)},
        {"test_years": (2024, 2024)},
        {"test_years": (2024.0,)},
        {"neutral_threshold_bps": 5.0},
        {"neutral_threshold_bps": np.nan},
        {"execution_latency_minutes": 2},
        {"xgb_max_rounds": 121},
        {"xgb_max_rounds": True},
        {"xgb_early_stopping_rounds": 11},
        {"xgb_max_rounds": 2},
        {"minimum_policy_trades": 0},
        {"output_directory": "../reports/test"},
        {"output_directory": "reports/../outside"},
        {"output_directory": "reports"},
        {"output_directory": "/reports/phase6"},
        {"data_config": "../phase5.yaml"},
        {"data_config": ""},
        {"data_config": "not_yaml.txt"},
        {"protocol_version": "phase6-modified"},
        {"allow_holdout_override": True},
    ],
)
def test_benchmark_config_rejects_invalid_or_unfrozen_scope(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(changes)


def test_default_configuration_loads_and_is_immutable() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_benchmark_config(root / "configs" / "benchmark.yaml")
    assert config == BenchmarkConfig()
    assert config.neutral_threshold_bps == 6.0
    with pytest.raises(ValidationError, match="frozen"):
        config.seed = 123
