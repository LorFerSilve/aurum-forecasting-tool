"""Executable-price label construction for the research MVP."""

from gold_forecasting.labels.mvp import (
    LABEL_COLUMNS,
    LabelBuildError,
    LabelBuildResult,
    build_mvp_labels,
    classify_future_returns,
)

__all__ = [
    "LABEL_COLUMNS",
    "LabelBuildError",
    "LabelBuildResult",
    "build_mvp_labels",
    "classify_future_returns",
]
