"""Deterministic baseline classifiers for the research MVP."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_forecasting.classification import (
    CLASS_ORDER,
    ClassLabel,
    PredictionBatch,
    one_hot_probabilities,
    validate_class_labels,
)


class BaselineError(ValueError):
    """Raised when a baseline cannot make a valid prediction."""


@dataclass(frozen=True, slots=True)
class MostFrequentClassBaseline:
    """Predict the most common training class, with fixed-order tie breaking."""

    predicted_class: ClassLabel

    @classmethod
    def fit(cls, train_labels: object) -> MostFrequentClassBaseline:
        labels = validate_class_labels(train_labels)
        counts = Counter(labels)
        winner = max(CLASS_ORDER, key=lambda label: (counts[label], -CLASS_ORDER.index(label)))
        return cls(predicted_class=winner)

    def predict(self, row_count: int) -> PredictionBatch:
        if row_count <= 0:
            raise BaselineError("row_count must be positive")
        labels = (self.predicted_class,) * row_count
        return PredictionBatch(labels, one_hot_probabilities(labels))


@dataclass(frozen=True, slots=True)
class ConstantClassBaseline:
    """Always predict one configured class."""

    predicted_class: ClassLabel

    def __post_init__(self) -> None:
        validate_class_labels((self.predicted_class,))

    def predict(self, row_count: int) -> PredictionBatch:
        if row_count <= 0:
            raise BaselineError("row_count must be positive")
        labels = (self.predicted_class,) * row_count
        return PredictionBatch(labels, one_hot_probabilities(labels))


@dataclass(frozen=True, slots=True)
class DirectionRuleBaseline:
    """Map a signed feature to down/neutral/up, optionally reversing direction."""

    name: str
    feature_name: str
    threshold_bps: float
    reverse: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise BaselineError("baseline name must not be empty")
        if not self.feature_name.strip():
            raise BaselineError("feature_name must not be empty")
        if not np.isfinite(self.threshold_bps) or self.threshold_bps < 0:
            raise BaselineError("threshold_bps must be a finite non-negative number")

    def predict(self, features: pd.DataFrame) -> PredictionBatch:
        if self.feature_name not in features.columns:
            raise BaselineError(f"missing baseline feature: {self.feature_name}")
        if features.empty:
            raise BaselineError("at least one feature row is required")
        try:
            values = features[self.feature_name].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise BaselineError(f"baseline feature {self.feature_name!r} must be numeric") from exc
        if not np.isfinite(values).all():
            raise BaselineError(f"baseline feature {self.feature_name!r} must be finite")

        labels: list[ClassLabel] = []
        for value in values:
            if value > self.threshold_bps:
                label: ClassLabel = "up"
            elif value < -self.threshold_bps:
                label = "down"
            else:
                label = "neutral"
            if self.reverse:
                reverse_labels: dict[ClassLabel, ClassLabel] = {
                    "down": "up",
                    "neutral": "neutral",
                    "up": "down",
                }
                label = reverse_labels[label]
            labels.append(label)
        frozen_labels = tuple(labels)
        return PredictionBatch(frozen_labels, one_hot_probabilities(frozen_labels))


def build_mvp_baseline_predictions(
    train_labels: object,
    test_features: pd.DataFrame,
    *,
    last_return_feature: str = "close_log_return_1_bps",
    momentum_feature: str = "momentum_5_bps",
    momentum_threshold_bps: float = 6.0,
) -> dict[str, PredictionBatch]:
    """Generate every roadmap baseline on the same ordered feature rows."""

    if test_features.empty:
        raise BaselineError("at least one test row is required")
    row_count = len(test_features)
    models = {
        "most_frequent": MostFrequentClassBaseline.fit(train_labels).predict(row_count),
        "always_up": ConstantClassBaseline("up").predict(row_count),
        "always_down": ConstantClassBaseline("down").predict(row_count),
        "last_candle_direction": DirectionRuleBaseline(
            name="last_candle_direction",
            feature_name=last_return_feature,
            threshold_bps=0.0,
        ).predict(test_features),
        "momentum": DirectionRuleBaseline(
            name="momentum",
            feature_name=momentum_feature,
            threshold_bps=momentum_threshold_bps,
        ).predict(test_features),
        "mean_reversion": DirectionRuleBaseline(
            name="mean_reversion",
            feature_name=momentum_feature,
            threshold_bps=momentum_threshold_bps,
            reverse=True,
        ).predict(test_features),
    }
    if {batch.row_count for batch in models.values()} != {row_count}:
        raise BaselineError("baseline predictions do not cover identical rows")
    return models


__all__ = [
    "BaselineError",
    "ConstantClassBaseline",
    "DirectionRuleBaseline",
    "MostFrequentClassBaseline",
    "build_mvp_baseline_predictions",
]
