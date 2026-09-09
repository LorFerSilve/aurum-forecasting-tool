"""Bounded classical challengers with train-only transforms and expectations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.api.types import is_bool_dtype, is_integer_dtype
from scipy.special import ndtr  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression, Ridge  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
from xgboost import XGBClassifier

from gold_forecasting.benchmark.config import BenchmarkConfig
from gold_forecasting.classification import (
    CLASS_ORDER,
    CLASS_TO_INDEX,
    validate_class_labels,
    validate_probability_matrix,
)
from gold_forecasting.datasets.preprocessing import TrainOnlyPreprocessor, fit_train_preprocessor
from gold_forecasting.evaluation.walk_forward import validate_development_frame

_CANDIDATE_VALUES = {
    "reference": (0.1,),
    "logistic": (0.1, 1.0, 10.0),
    "ridge": (0.1, 10.0, 1000.0),
    "xgboost": (2.0, 3.0, 4.0),
}
# Explicitly frozen for phase6-v1; arbitrary columns cannot silently become
# model inputs. New feature families require the next versioned protocol.
ALLOWED_FEATURE_NAMES = frozenset(
    {
        "close_log_return_1_bps",
        "close_log_return_lag_1_bps",
        "close_log_return_lag_2_bps",
        "close_log_return_lag_3_bps",
        "candle_body_bps",
        "candle_range_bps",
        "upper_wick_bps",
        "lower_wick_bps",
        "close_position",
        "momentum_5_bps",
        "momentum_20_bps",
        "realized_volatility_20_bps",
        "distance_to_sma_5_bps",
        "distance_to_sma_20_bps",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
    }
)
_EMBARGO_MINUTES = 181


@dataclass(frozen=True)
class ModelSpec:
    family: str
    value: float

    def __post_init__(self) -> None:
        if (
            self.family not in _CANDIDATE_VALUES
            or isinstance(self.value, bool)
            or self.value not in _CANDIDATE_VALUES[self.family]
        ):
            raise ValueError("model specification is outside the frozen phase6-v1 budget")

    @property
    def name(self) -> str:
        return f"{self.family}-{self.value:g}"


def candidate_specs(family: str) -> tuple[ModelSpec, ...]:
    if family not in _CANDIDATE_VALUES or family == "reference":
        raise ValueError("candidate family must be logistic, ridge or xgboost")
    return tuple(ModelSpec(family, value) for value in _CANDIDATE_VALUES[family])


def _validate_targets(frame: pd.DataFrame, *, name: str) -> None:
    required = {"target_class", "target_class_id", "arithmetic_return_bps", "sample_id"}
    missing = required.difference(frame.columns)
    if frame.empty or missing:
        raise ValueError(f"{name} must be nonempty with target/audit columns: {sorted(missing)}")
    validate_development_frame(frame)
    labels = validate_class_labels(frame["target_class"])
    class_ids = frame["target_class_id"]
    if not is_integer_dtype(class_ids.dtype) or is_bool_dtype(class_ids.dtype):
        raise ValueError(f"{name} target_class_id must contain integer class IDs")
    expected_ids = np.array([CLASS_TO_INDEX[label] for label in labels])
    if class_ids.isna().any() or not np.array_equal(class_ids.to_numpy(), expected_ids):
        raise ValueError(f"{name} target_class_id disagrees with the fixed class order")
    returns = frame["arithmetic_return_bps"].to_numpy(dtype=np.float64)
    if not np.isfinite(returns).all():
        raise ValueError(f"{name} arithmetic returns must be finite")
    sample_ids = frame["sample_id"]
    if (
        sample_ids.duplicated().any()
        or sample_ids.map(lambda value: not isinstance(value, str) or not value.strip()).any()
    ):
        raise ValueError(f"{name} sample IDs must be nonempty unique strings")


def _ordered_probabilities(estimator: Any, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    raw = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
    classes = tuple(estimator.classes_)
    if len(classes) != 3 or set(classes) != {0, 1, 2}:
        raise ValueError("fitted classifier must expose all three integer class IDs")
    if raw.shape != (len(matrix), 3):
        raise ValueError("fitted classifier probability matrix has an invalid shape")
    probabilities = raw[:, [classes.index(index) for index in range(3)]]
    sums = probabilities.sum(axis=1, keepdims=True)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(sums, 1.0, rtol=0.0, atol=1e-6)
    ):
        raise ValueError("fitted classifier produced invalid probabilities")
    # XGBoost's float32 softmax can sum to 1 +/- approximately 1e-7.
    # Correct only that numerical precision before the strict shared contract.
    probabilities /= sums
    return validate_probability_matrix(probabilities, expected_rows=len(matrix))


@dataclass
class FittedModel:
    spec: ModelSpec
    estimator: Any
    preprocessor: TrainOnlyPreprocessor
    class_returns: NDArray[np.float64]
    residual_scale: float
    best_rounds: int | None
    threshold_bps: float

    def predict(self, frame: pd.DataFrame) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # This bundle is a development-research artifact, not a production
        # predictor: it deliberately cannot inspect the final holdout.
        validate_development_frame(frame)
        if frame.empty:
            raise ValueError("prediction requires nonempty development records")
        matrix = self.preprocessor.transform(frame)
        with threadpool_limits(limits=2):
            if self.spec.family == "ridge":
                expected = np.asarray(self.estimator.predict(matrix), dtype=np.float64)
                # Labels use log returns; convert those fixed bounds to arithmetic bps.
                lower = np.expm1(-self.threshold_bps / 10000) * 10000
                upper = np.expm1(self.threshold_bps / 10000) * 10000
                down = ndtr((lower - expected) / self.residual_scale)
                below_up = ndtr((upper - expected) / self.residual_scale)
                probabilities = np.column_stack((down, below_up - down, 1 - below_up))
            else:
                probabilities = _ordered_probabilities(self.estimator, matrix)
        probabilities = np.clip(probabilities, 1e-12, 1.0)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        if self.spec.family != "ridge":
            expected = probabilities @ self.class_returns
        if expected.shape != (len(frame),) or not np.isfinite(expected).all():
            raise ValueError("model produced non-finite or misaligned expected returns")
        return validate_probability_matrix(probabilities, expected_rows=len(frame)), expected


def class_return_means(train: pd.DataFrame) -> NDArray[np.float64]:
    means = train.groupby("target_class")["arithmetic_return_bps"].mean()
    return np.array([float(means.get(label, 0.0)) for label in CLASS_ORDER])


def fit_model(
    train: pd.DataFrame,
    feature_names: tuple[str, ...],
    spec: ModelSpec,
    config: BenchmarkConfig,
    *,
    validation: pd.DataFrame | None = None,
    rounds: int | None = None,
    allowed_feature_names: frozenset[str] | None = None,
    class_weight: Literal["balanced"] | None = "balanced",
) -> FittedModel:
    """Fit development rows with train-only transforms and target statistics.

    Validation is used only for XGBoost early stopping. Its observation times
    must follow training by the frozen common embargo. Fold/calibration-block
    selection remains the caller's responsibility when only train is supplied.
    """
    if class_weight not in (None, "balanced"):
        raise ValueError("class_weight must be None or balanced")
    if spec.family != "logistic" and class_weight != "balanced":
        raise ValueError("class_weight overrides are restricted to logistic challengers")
    _validate_targets(train, name="training")
    if set(train["target_class_id"].unique()) != {0, 1, 2}:
        raise ValueError("training requires nonempty coverage of all three target classes")
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("feature names must be nonempty and unique")
    allowed = ALLOWED_FEATURE_NAMES if allowed_feature_names is None else allowed_feature_names
    if not allowed or any(not isinstance(name, str) or not name for name in allowed):
        raise ValueError("allowed feature catalog must contain non-empty string names")
    if not set(feature_names).issubset(allowed):
        scope = (
            "frozen causal MVP allowlist"
            if allowed_feature_names is None
            else "explicit causal feature catalog"
        )
        raise ValueError(f"feature names must belong to the {scope}")
    if "split" in train and not train["split"].eq("train").all():
        raise ValueError("training input contains rows explicitly marked as another split")
    if validation is not None:
        _validate_targets(validation, name="validation")
        if train["label_end_time_utc"].max() >= validation["prediction_time_utc"].min():
            raise ValueError("training labels overlap validation")
        gap = validation["prediction_time_utc"].min() - train["prediction_time_utc"].max()
        if gap <= pd.Timedelta(minutes=_EMBARGO_MINUTES):
            raise ValueError("training predictions violate the common 181-minute embargo")
        if set(train["sample_id"]).intersection(validation["sample_id"]):
            raise ValueError("training and validation sample IDs overlap")
    if rounds is not None and (
        isinstance(rounds, bool)
        or not isinstance(rounds, int)
        or rounds < 1
        or rounds > config.xgb_max_rounds
        or spec.family != "xgboost"
        or validation is not None
    ):
        raise ValueError("rounds must be an inner-selected final XGBoost count within budget")
    preprocessor = fit_train_preprocessor(train.assign(split="train"), feature_names)
    matrix = preprocessor.transform(train)
    labels = train["target_class_id"].to_numpy(dtype=np.int64)
    returns = train["arithmetic_return_bps"].to_numpy(dtype=np.float64)
    best_rounds: int | None = None
    scale = 1.0
    estimator: Any
    with threadpool_limits(limits=2):
        if spec.family in {"logistic", "reference"}:
            estimator = LogisticRegression(
                C=spec.value, class_weight=class_weight, max_iter=1000, random_state=config.seed
            )
            estimator.fit(matrix, labels)
        elif spec.family == "ridge":
            estimator = Ridge(alpha=spec.value)
            estimator.fit(matrix, returns)
            scale = max(float(np.std(returns - estimator.predict(matrix))), 1e-6)
        elif spec.family == "xgboost":
            early = validation is not None
            if not early and rounds is None:
                raise ValueError("final XGBoost refit requires inner-selected round count")
            estimator = XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                max_depth=int(spec.value),
                learning_rate=0.05,
                n_estimators=config.xgb_max_rounds if early else rounds,
                early_stopping_rounds=config.xgb_early_stopping_rounds if early else None,
                tree_method="hist",
                n_jobs=2,
                random_state=config.seed,
                subsample=1.0,
                colsample_bytree=1.0,
            )
            kwargs: dict[str, Any] = {"verbose": False}
            if validation is not None:
                kwargs["eval_set"] = [
                    (preprocessor.transform(validation), validation["target_class_id"].to_numpy())
                ]
            estimator.fit(matrix, labels, **kwargs)
            best_rounds = int(estimator.best_iteration) + 1 if early else rounds
        else:
            raise ValueError(f"unknown model family {spec.family}")
    return FittedModel(
        spec,
        estimator,
        preprocessor,
        class_return_means(train),
        scale,
        best_rounds,
        config.neutral_threshold_bps,
    )


def prediction_records(
    frame: pd.DataFrame,
    probabilities: NDArray[np.float64],
    expected: NDArray[np.float64],
) -> pd.DataFrame:
    """Preserve identical ordered audit rows for every model and policy."""
    records = frame.copy()
    matrix = validate_probability_matrix(probabilities, expected_rows=len(records))
    if expected.shape != (len(frame),) or not np.isfinite(expected).all():
        raise ValueError("expected returns must be finite and aligned")
    for index, label in enumerate(CLASS_ORDER):
        records[f"p_{label}"] = matrix[:, index]
    records["predicted_class"] = np.array(CLASS_ORDER)[matrix.argmax(axis=1)]
    records["expected_return_bps"] = expected
    return records
