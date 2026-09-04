"""Shared, strict contracts for three-class MVP predictions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

ClassLabel: TypeAlias = Literal["down", "neutral", "up"]
CLASS_ORDER: tuple[ClassLabel, ...] = ("down", "neutral", "up")
CLASS_TO_INDEX: dict[ClassLabel, int] = {
    label: index for index, label in enumerate(CLASS_ORDER)
}


class ClassificationContractError(ValueError):
    """Raised when labels or probabilities violate the fixed MVP contract."""


def validate_class_labels(labels: object, *, allow_empty: bool = False) -> tuple[ClassLabel, ...]:
    """Return labels after enforcing the fixed ``down/neutral/up`` vocabulary."""

    if isinstance(labels, (str, bytes)):
        raise ClassificationContractError("labels must be a one-dimensional collection")
    if not isinstance(labels, Iterable):
        raise ClassificationContractError("labels must be iterable")
    raw_labels = list(labels)
    if not raw_labels and not allow_empty:
        raise ClassificationContractError("at least one label is required")

    validated: list[ClassLabel] = []
    for value in raw_labels:
        if not isinstance(value, str) or value not in CLASS_ORDER:
            raise ClassificationContractError(
                f"unknown class label {value!r}; expected one of {CLASS_ORDER}"
            )
        validated.append(value)
    return tuple(validated)


def validate_probability_matrix(
    probabilities: ArrayLike,
    *,
    expected_rows: int | None = None,
    allow_empty: bool = False,
) -> NDArray[np.float64]:
    """Validate and copy an ``n x 3`` probability matrix.

    Columns always follow :data:`CLASS_ORDER`.  A copy is returned so callers
    cannot mutate the input behind a validated prediction object.
    """

    try:
        matrix = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ClassificationContractError("probabilities must be numeric") from exc
    if matrix.ndim != 2 or matrix.shape[1] != len(CLASS_ORDER):
        raise ClassificationContractError(
            f"probabilities must have shape (n, {len(CLASS_ORDER)}) in {CLASS_ORDER} order"
        )
    if matrix.shape[0] == 0 and not allow_empty:
        raise ClassificationContractError("at least one probability row is required")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ClassificationContractError(
            f"probability row count {matrix.shape[0]} does not match expected {expected_rows}"
        )
    if not np.isfinite(matrix).all():
        raise ClassificationContractError("probabilities must all be finite")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ClassificationContractError("probabilities must lie in [0, 1]")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-9):
        raise ClassificationContractError("each probability row must sum to one")
    return matrix.copy()


def labels_from_probabilities(probabilities: ArrayLike) -> tuple[ClassLabel, ...]:
    """Choose classes deterministically, using :data:`CLASS_ORDER` for ties."""

    matrix = validate_probability_matrix(probabilities)
    return tuple(CLASS_ORDER[int(index)] for index in np.argmax(matrix, axis=1))


def one_hot_probabilities(labels: object) -> NDArray[np.float64]:
    """Encode fixed-vocabulary labels as deterministic probabilities."""

    validated = validate_class_labels(labels)
    matrix = np.zeros((len(validated), len(CLASS_ORDER)), dtype=np.float64)
    for row, label in enumerate(validated):
        matrix[row, CLASS_TO_INDEX[label]] = 1.0
    return matrix


@dataclass(frozen=True, slots=True)
class PredictionBatch:
    """Predicted classes and their aligned probability matrix."""

    predicted_class: tuple[ClassLabel, ...]
    probabilities: NDArray[np.float64]

    def __post_init__(self) -> None:
        labels = validate_class_labels(self.predicted_class)
        matrix = validate_probability_matrix(self.probabilities, expected_rows=len(labels))
        inferred = tuple(CLASS_ORDER[int(index)] for index in np.argmax(matrix, axis=1))
        if labels != inferred:
            raise ClassificationContractError(
                "predicted_class must equal the probability argmax in fixed class order"
            )
        matrix.setflags(write=False)
        object.__setattr__(self, "predicted_class", labels)
        object.__setattr__(self, "probabilities", matrix)

    @classmethod
    def from_probabilities(cls, probabilities: ArrayLike) -> PredictionBatch:
        matrix = validate_probability_matrix(probabilities)
        labels = tuple(CLASS_ORDER[int(index)] for index in np.argmax(matrix, axis=1))
        return cls(predicted_class=labels, probabilities=matrix)

    @property
    def row_count(self) -> int:
        return len(self.predicted_class)


__all__ = [
    "CLASS_ORDER",
    "CLASS_TO_INDEX",
    "ClassLabel",
    "ClassificationContractError",
    "PredictionBatch",
    "labels_from_probabilities",
    "one_hot_probabilities",
    "validate_class_labels",
    "validate_probability_matrix",
]
