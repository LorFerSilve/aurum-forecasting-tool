"""Strict classification metrics for the three-class research MVP."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)

from gold_forecasting.classification import (
    CLASS_ORDER,
    CLASS_TO_INDEX,
    validate_class_labels,
    validate_probability_matrix,
)


class EvaluationError(ValueError):
    """Raised when inputs cannot be evaluated under the MVP contract."""


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Classification metrics for an eligible set and its issued predictions."""

    eligible_count: int
    issued_count: int
    coverage: float
    accuracy: float | None
    balanced_accuracy: float | None
    macro_f1: float | None
    log_loss: float | None
    multiclass_brier: float | None
    confusion_matrix: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HourlyMetricSummary:
    """Metrics for one UTC prediction hour."""

    hour_utc: int
    metrics: MetricSummary

    def to_dict(self) -> dict[str, Any]:
        return {"hour_utc": self.hour_utc, **self.metrics.to_dict()}


def _validate_issued_mask(mask: ArrayLike | None, *, row_count: int) -> NDArray[np.bool_]:
    if mask is None:
        return np.ones(row_count, dtype=np.bool_)
    values = np.asarray(mask)
    if values.ndim != 1 or len(values) != row_count:
        raise EvaluationError("issued_mask must be one-dimensional and align with all samples")
    if not np.issubdtype(values.dtype, np.bool_):
        raise EvaluationError("issued_mask must contain booleans")
    return values.astype(np.bool_, copy=True)


def _validate_metric_inputs(
    y_true: object,
    probabilities: ArrayLike,
    issued_mask: ArrayLike | None,
) -> tuple[tuple[str, ...], NDArray[np.float64], NDArray[np.bool_]]:
    try:
        labels = validate_class_labels(y_true)
        matrix = validate_probability_matrix(probabilities, expected_rows=len(labels))
    except ValueError as exc:
        raise EvaluationError(str(exc)) from exc
    mask = _validate_issued_mask(issued_mask, row_count=len(labels))
    return labels, matrix, mask


def _empty_confusion_matrix() -> tuple[tuple[int, ...], ...]:
    width = len(CLASS_ORDER)
    return tuple(tuple(0 for _ in range(width)) for _ in range(width))


def _compute_validated_metrics(
    labels: Sequence[str],
    matrix: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> MetricSummary:
    eligible_count = len(labels)
    issued_count = int(mask.sum())
    coverage = issued_count / eligible_count if eligible_count else 0.0
    if issued_count == 0:
        return MetricSummary(
            eligible_count=eligible_count,
            issued_count=0,
            coverage=coverage,
            accuracy=None,
            balanced_accuracy=None,
            macro_f1=None,
            log_loss=None,
            multiclass_brier=None,
            confusion_matrix=_empty_confusion_matrix(),
        )

    true_issued = np.asarray(labels, dtype=object)[mask]
    probability_issued = matrix[mask]
    predicted_issued = np.asarray(
        [CLASS_ORDER[int(index)] for index in np.argmax(probability_issued, axis=1)],
        dtype=object,
    )
    confusion = confusion_matrix(
        true_issued,
        predicted_issued,
        labels=list(CLASS_ORDER),
    ).astype(np.int64)
    supports = confusion.sum(axis=1)
    present = supports > 0
    recalls = np.divide(
        np.diag(confusion),
        supports,
        out=np.zeros(len(CLASS_ORDER), dtype=np.float64),
        where=present,
    )
    balanced_accuracy = float(recalls[present].mean())

    true_indices = np.fromiter(
        (CLASS_TO_INDEX[label] for label in true_issued),
        dtype=np.int64,
        count=issued_count,
    )
    one_hot = np.eye(len(CLASS_ORDER), dtype=np.float64)[true_indices]
    brier = float(np.mean(np.sum(np.square(probability_issued - one_hot), axis=1)))

    return MetricSummary(
        eligible_count=eligible_count,
        issued_count=issued_count,
        coverage=coverage,
        accuracy=float(accuracy_score(true_issued, predicted_issued)),
        balanced_accuracy=balanced_accuracy,
        macro_f1=float(
            f1_score(
                true_issued,
                predicted_issued,
                labels=list(CLASS_ORDER),
                average="macro",
                zero_division=0,
            )
        ),
        log_loss=float(log_loss(true_issued, probability_issued, labels=list(CLASS_ORDER))),
        multiclass_brier=brier,
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in confusion),
    )


def compute_classification_metrics(
    y_true: object,
    probabilities: ArrayLike,
    *,
    issued_mask: ArrayLike | None = None,
) -> MetricSummary:
    """Compute fixed-order metrics on issued predictions and report their coverage.

    Multiclass Brier is the mean squared Euclidean distance between the
    probability vector and the one-hot target.  It therefore ranges from zero
    to two.  Metrics are ``None`` when a model issues no predictions; coverage
    and the all-zero confusion matrix remain reportable.
    """

    labels, matrix, mask = _validate_metric_inputs(y_true, probabilities, issued_mask)
    return _compute_validated_metrics(labels, matrix, mask)


def _utc_hours(values: object, *, expected_rows: int) -> NDArray[np.int64]:
    if isinstance(values, (str, bytes)):
        raise EvaluationError("prediction_times_utc must be a one-dimensional collection")
    if not isinstance(values, Iterable):
        raise EvaluationError("prediction_times_utc must be iterable")
    raw = list(values)
    if len(raw) != expected_rows:
        raise EvaluationError("prediction_times_utc must align with all samples")
    try:
        timestamps = pd.to_datetime(raw, errors="raise", utc=True)
    except (TypeError, ValueError) as exc:
        raise EvaluationError("prediction_times_utc contains an invalid timestamp") from exc
    if not isinstance(timestamps, pd.DatetimeIndex) or np.asarray(timestamps.isna()).any():
        raise EvaluationError("prediction_times_utc contains an invalid timestamp")
    return timestamps.hour.to_numpy(dtype=np.int64)


def compute_metrics_by_utc_hour(
    y_true: object,
    probabilities: ArrayLike,
    prediction_times_utc: object,
    *,
    issued_mask: ArrayLike | None = None,
) -> tuple[HourlyMetricSummary, ...]:
    """Return all 24 UTC-hour rows, including hours with zero eligible samples."""

    labels, matrix, mask = _validate_metric_inputs(y_true, probabilities, issued_mask)
    hours = _utc_hours(prediction_times_utc, expected_rows=len(labels))
    summaries: list[HourlyMetricSummary] = []
    label_array = np.asarray(labels, dtype=object)
    for hour in range(24):
        selected = hours == hour
        hour_labels = tuple(str(value) for value in label_array[selected])
        hour_matrix = matrix[selected]
        hour_mask = mask[selected]
        summaries.append(
            HourlyMetricSummary(
                hour_utc=hour,
                metrics=_compute_validated_metrics(hour_labels, hour_matrix, hour_mask),
            )
        )
    return tuple(summaries)


__all__ = [
    "EvaluationError",
    "HourlyMetricSummary",
    "MetricSummary",
    "compute_classification_metrics",
    "compute_metrics_by_utc_hour",
]
