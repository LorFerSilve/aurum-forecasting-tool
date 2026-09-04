"""Train-only multinomial logistic regression with validation-only selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import f1_score, log_loss  # type: ignore[import-untyped]

from gold_forecasting.classification import (
    CLASS_ORDER,
    CLASS_TO_INDEX,
    PredictionBatch,
    validate_class_labels,
)

ClassWeight: TypeAlias = Literal["balanced"] | None


class LogisticTrainingError(ValueError):
    """Raised when a leakage-safe logistic fit cannot be performed."""


@dataclass(frozen=True, slots=True)
class LogisticCandidate:
    c: float
    class_weight: ClassWeight

    def __post_init__(self) -> None:
        if not np.isfinite(self.c) or self.c <= 0:
            raise LogisticTrainingError("candidate C must be finite and positive")
        if self.class_weight not in (None, "balanced"):
            raise LogisticTrainingError("class_weight must be None or 'balanced'")


@dataclass(frozen=True, slots=True)
class CandidateScore:
    candidate: LogisticCandidate
    validation_macro_f1: float
    validation_log_loss: float


@dataclass(frozen=True, slots=True)
class LogisticSelectionResult:
    """Selected train-only estimator and the validation audit trail."""

    estimator: LogisticRegression
    selected_candidate: LogisticCandidate
    candidate_scores: tuple[CandidateScore, ...]
    train_row_count: int
    validation_row_count: int
    feature_count: int


def _numeric_matrix(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise LogisticTrainingError(f"{name} must be a dense numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise LogisticTrainingError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise LogisticTrainingError(f"{name} must contain only finite values")
    return matrix


def _ordered_probabilities(
    estimator: LogisticRegression,
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    raw = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = tuple(str(label) for label in estimator.classes_)
    if set(classes) != set(CLASS_ORDER):
        raise LogisticTrainingError(
            f"fitted estimator classes {classes} do not match required {CLASS_ORDER}"
        )
    ordered = np.empty((len(features), len(CLASS_ORDER)), dtype=np.float64)
    for target_index, label in enumerate(CLASS_ORDER):
        ordered[:, target_index] = raw[:, classes.index(label)]
    return ordered


def predict_logistic(estimator: LogisticRegression, features: ArrayLike) -> PredictionBatch:
    """Predict with a fitted estimator while restoring the fixed class order."""

    matrix = _numeric_matrix(features, name="features")
    probabilities = _ordered_probabilities(estimator, matrix)
    return PredictionBatch.from_probabilities(probabilities)


def select_logistic_candidate(
    train_features: ArrayLike,
    train_labels: object,
    validation_features: ArrayLike,
    validation_labels: object,
    *,
    c_values: tuple[float, ...] = (0.1, 1.0, 10.0),
    class_weight_options: tuple[ClassWeight, ...] = (None, "balanced"),
    seed: int = 0,
    max_iter: int = 2_000,
) -> LogisticSelectionResult:
    """Fit candidates only on train and rank them only on validation.

    Macro-F1 is the primary criterion, log loss the tie-breaker.  An exact
    metric tie prefers no class weighting and then ``C=1``.  The returned
    estimator is the already fitted train-only candidate; validation rows are
    deliberately never folded back into the final MVP fit.
    """

    x_train = _numeric_matrix(train_features, name="train_features")
    x_validation = _numeric_matrix(validation_features, name="validation_features")
    y_train = validate_class_labels(train_labels)
    y_validation = validate_class_labels(validation_labels)
    if x_train.shape[0] != len(y_train):
        raise LogisticTrainingError("train feature and label row counts differ")
    if x_validation.shape[0] != len(y_validation):
        raise LogisticTrainingError("validation feature and label row counts differ")
    if x_train.shape[1] != x_validation.shape[1]:
        raise LogisticTrainingError("train and validation feature counts differ")
    missing_classes = set(CLASS_ORDER).difference(y_train)
    if missing_classes:
        raise LogisticTrainingError(
            f"train labels must contain every MVP class; missing {sorted(missing_classes)}"
        )
    if not c_values or len(c_values) != len(set(c_values)):
        raise LogisticTrainingError("c_values must be non-empty and unique")
    if not class_weight_options or len(class_weight_options) != len(set(class_weight_options)):
        raise LogisticTrainingError("class_weight_options must be non-empty and unique")
    if max_iter <= 0:
        raise LogisticTrainingError("max_iter must be positive")
    if seed < 0:
        raise LogisticTrainingError("seed must be non-negative")

    candidates = tuple(
        LogisticCandidate(c=c, class_weight=class_weight)
        for c in c_values
        for class_weight in class_weight_options
    )
    fitted: list[tuple[CandidateScore, LogisticRegression]] = []
    y_train_array = np.asarray(y_train, dtype=object)
    y_validation_array = np.asarray(y_validation, dtype=object)
    for candidate in candidates:
        estimator = LogisticRegression(
            C=candidate.c,
            class_weight=candidate.class_weight,
            solver="lbfgs",
            max_iter=max_iter,
            random_state=seed,
        )
        estimator.fit(x_train, y_train_array)
        probabilities = _ordered_probabilities(estimator, x_validation)
        predicted = np.asarray(
            [CLASS_ORDER[int(index)] for index in np.argmax(probabilities, axis=1)],
            dtype=object,
        )
        score = CandidateScore(
            candidate=candidate,
            validation_macro_f1=float(
                f1_score(
                    y_validation_array,
                    predicted,
                    labels=list(CLASS_ORDER),
                    average="macro",
                    zero_division=0,
                )
            ),
            validation_log_loss=float(
                log_loss(y_validation_array, probabilities, labels=list(CLASS_ORDER))
            ),
        )
        fitted.append((score, estimator))

    def selection_key(item: tuple[CandidateScore, LogisticRegression]) -> tuple[float, ...]:
        score = item[0]
        return (
            -score.validation_macro_f1,
            score.validation_log_loss,
            0.0 if score.candidate.class_weight is None else 1.0,
            abs(float(np.log10(score.candidate.c))),
            score.candidate.c,
        )

    selected_score, selected_estimator = min(fitted, key=selection_key)
    # A defensive assertion guards against accidental class-order drift in sklearn calls.
    if tuple(CLASS_ORDER[CLASS_TO_INDEX[label]] for label in CLASS_ORDER) != CLASS_ORDER:
        raise LogisticTrainingError("internal class order is inconsistent")
    return LogisticSelectionResult(
        estimator=selected_estimator,
        selected_candidate=selected_score.candidate,
        candidate_scores=tuple(score for score, _ in fitted),
        train_row_count=len(y_train),
        validation_row_count=len(y_validation),
        feature_count=x_train.shape[1],
    )


__all__ = [
    "CandidateScore",
    "ClassWeight",
    "LogisticCandidate",
    "LogisticSelectionResult",
    "LogisticTrainingError",
    "predict_logistic",
    "select_logistic_candidate",
]
