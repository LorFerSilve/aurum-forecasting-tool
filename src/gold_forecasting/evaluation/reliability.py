"""Confidence reliability diagnostics for preliminary MVP probabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from gold_forecasting.classification import (
    CLASS_ORDER,
    validate_class_labels,
    validate_probability_matrix,
)
from gold_forecasting.evaluation.metrics import EvaluationError

CONFIDENCE_BUCKET_COUNT = 10


@dataclass(frozen=True, slots=True)
class ReliabilityBucket:
    """One left-closed confidence bucket; the final bucket is closed at 1.0."""

    bucket_index: int
    lower_bound: float
    upper_bound: float
    upper_inclusive: bool
    sample_count: int
    fraction_of_issued: float
    mean_confidence: float | None
    empirical_accuracy: float | None
    calibration_gap: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _boolean_mask(mask: ArrayLike | None, *, row_count: int) -> np.ndarray[Any, np.dtype[np.bool_]]:
    if mask is None:
        return np.ones(row_count, dtype=np.bool_)
    values = np.asarray(mask)
    if values.ndim != 1 or len(values) != row_count:
        raise EvaluationError("issued_mask must be one-dimensional and align with all samples")
    if not np.issubdtype(values.dtype, np.bool_):
        raise EvaluationError("issued_mask must contain booleans")
    return values.astype(np.bool_, copy=True)


def build_reliability_table(
    y_true: object,
    probabilities: ArrayLike,
    *,
    issued_mask: ArrayLike | None = None,
    bucket_count: int = CONFIDENCE_BUCKET_COUNT,
) -> tuple[ReliabilityBucket, ...]:
    """Bucket maximum-class confidence into configurable deterministic intervals.

    With the default, intervals are ``[0.0, 0.1)`` through ``[0.9, 1.0]``.
    Clipping the computed index makes an exact confidence of ``1.0`` part of
    the final bucket rather than an out-of-range extra bucket.
    """

    if isinstance(bucket_count, bool) or not isinstance(bucket_count, int):
        raise EvaluationError("bucket_count must be an integer")
    if not 2 <= bucket_count <= 100:
        raise EvaluationError("bucket_count must lie in [2, 100]")
    try:
        labels = validate_class_labels(y_true)
        matrix = validate_probability_matrix(probabilities, expected_rows=len(labels))
    except ValueError as exc:
        raise EvaluationError(str(exc)) from exc
    mask = _boolean_mask(issued_mask, row_count=len(labels))
    issued_count = int(mask.sum())
    confidence = np.max(matrix, axis=1)
    predictions = np.argmax(matrix, axis=1)
    targets = np.fromiter(
        (CLASS_ORDER.index(label) for label in labels),
        dtype=np.int64,
        count=len(labels),
    )
    correct = predictions == targets
    bucket_indices = np.minimum(
        np.floor(confidence * bucket_count).astype(np.int64),
        bucket_count - 1,
    )

    buckets: list[ReliabilityBucket] = []
    for index in range(bucket_count):
        selected = mask & (bucket_indices == index)
        count = int(selected.sum())
        lower = index / bucket_count
        upper = (index + 1) / bucket_count
        mean_confidence = float(confidence[selected].mean()) if count else None
        empirical_accuracy = float(correct[selected].mean()) if count else None
        calibration_gap = (
            empirical_accuracy - mean_confidence
            if mean_confidence is not None and empirical_accuracy is not None
            else None
        )
        buckets.append(
            ReliabilityBucket(
                bucket_index=index,
                lower_bound=lower,
                upper_bound=upper,
                upper_inclusive=index == bucket_count - 1,
                sample_count=count,
                fraction_of_issued=count / issued_count if issued_count else 0.0,
                mean_confidence=mean_confidence,
                empirical_accuracy=empirical_accuracy,
                calibration_gap=calibration_gap,
            )
        )
    return tuple(buckets)


__all__ = ["CONFIDENCE_BUCKET_COUNT", "ReliabilityBucket", "build_reliability_table"]
