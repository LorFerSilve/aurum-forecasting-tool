"""Frozen phase-6 expanding-window evaluation with a fail-closed holdout guard.

All intervals are half-open UTC intervals. For an outer test year Y, the last
quarter of Y-1 is a *reserved* calibration block, not a training or selection
block. Inner validation uses the second and third quarters of Y-1. The final
model can fit the development history preceding that calibration block.

Apply ``select_block(..., purge=True)`` to every training block with the SAME
gap for every horizon: maximum configured horizon plus execution latency.
Training prediction times must precede ``end - gap`` and their labels must
mature strictly before ``end``. In this strictly forward-only design the gap
is on the training side of each boundary; there are no post-validation training
observations needing a second embargo. Validation, calibration, and test use
``purge=False``: their labels must still mature strictly inside the block.

The 2024 observations were already inspected during MVP development. These
folds therefore provide development evidence, not a new pristine holdout.
Nothing here permits access to the reserved 2025-or-later observations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from statistics import mean, median

import pandas as pd

FINAL_HOLDOUT_START = pd.Timestamp("2025-01-01T00:00:00Z")
DEFAULT_TEST_YEARS = (2022, 2023, 2024)
_REQUIRED_TIME_COLUMNS = ("prediction_time_utc", "label_end_time_utc")
_ADDITIONAL_TIME_COLUMNS = frozenset(
    {"feature_available_at_utc", "feature_window_start_utc", "source_candle_open_utc"}
)


class WalkForwardError(ValueError):
    """Raised for a malformed schedule, unsafe observations, or invalid metrics."""


def _utc_timestamp(value: pd.Timestamp | datetime | str, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise WalkForwardError(f"{name} must be a timezone-aware UTC timestamp") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None or str(timestamp.tzinfo) != "UTC":
        raise WalkForwardError(f"{name} must be a timezone-aware UTC timestamp")
    return timestamp


def _development_guard(value: pd.Timestamp | datetime | str) -> pd.Timestamp:
    guard = _utc_timestamp(value, name="guard_start")
    if guard > FINAL_HOLDOUT_START:
        raise WalkForwardError("guard_start cannot expose the reserved 2025-or-later holdout")
    return guard


@dataclass(frozen=True, slots=True)
class TimeBlock:
    """A nonempty half-open interval whose boundaries are explicitly UTC."""

    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        start = _utc_timestamp(self.start, name="block start")
        end = _utc_timestamp(self.end, name="block end")
        if start >= end:
            raise WalkForwardError("block start must precede block end")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def as_record(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


@dataclass(frozen=True, slots=True)
class InnerFold:
    """One expanding training block and the immediately following validation."""

    name: str
    train: TimeBlock
    validation: TimeBlock

    def __post_init__(self) -> None:
        if not self.name:
            raise WalkForwardError("inner fold name must not be empty")
        if self.train.end > self.validation.start:
            raise WalkForwardError("inner training and validation blocks overlap")

    def as_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "train": self.train.as_record(),
            "validation": self.validation.as_record(),
        }


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One outer development fold, including a reserved calibration block."""

    name: str
    train: TimeBlock
    calibration: TimeBlock
    test: TimeBlock
    inner_folds: tuple[InnerFold, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.inner_folds:
            raise WalkForwardError("outer folds require a name and at least one inner fold")
        if self.train.end > self.calibration.start or self.calibration.end > self.test.start:
            raise WalkForwardError("outer training, calibration, and test blocks overlap")
        if self.test.end > FINAL_HOLDOUT_START:
            raise WalkForwardError("outer fold would expose the reserved holdout")
        previous_validation_end: pd.Timestamp | None = None
        inner_names: set[str] = set()
        for inner in self.inner_folds:
            if inner.name in inner_names:
                raise WalkForwardError("inner fold names must be unique")
            inner_names.add(inner.name)
            if inner.train.start != self.train.start or inner.validation.end > self.train.end:
                raise WalkForwardError("inner folds must remain inside the outer training history")
            if (
                previous_validation_end is not None
                and inner.validation.start < previous_validation_end
            ):
                raise WalkForwardError("inner validation blocks must be ordered and disjoint")
            previous_validation_end = inner.validation.end

    def as_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "train": self.train.as_record(),
            "calibration": self.calibration.as_record(),
            "calibration_usage": "reserved_not_used_for_fit_or_selection",
            "test": self.test.as_record(),
            "inner_folds": [inner.as_record() for inner in self.inner_folds],
        }


def _boundary(year: int, month: int = 1) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=1, tz="UTC")


def make_walk_forward_folds(
    test_years: Sequence[int] = DEFAULT_TEST_YEARS,
    *,
    start_year: int = 2020,
) -> tuple[WalkForwardFold, ...]:
    """Build the frozen calendar schedule without consulting model outcomes."""

    years = tuple(test_years)
    if isinstance(start_year, bool) or not isinstance(start_year, int):
        raise WalkForwardError("start_year must be an integer year")
    if not years or any(isinstance(year, bool) or not isinstance(year, int) for year in years):
        raise WalkForwardError("test_years must contain integer years")
    if years != tuple(sorted(set(years))):
        raise WalkForwardError("test_years must be unique and increasing")
    if start_year < 1970 or years[0] <= start_year or years[-1] >= 2025:
        raise WalkForwardError("test years must follow development start and precede the holdout")
    start = _boundary(start_year)
    folds: list[WalkForwardFold] = []
    for year in years:
        april = _boundary(year - 1, 4)
        july = _boundary(year - 1, 7)
        october = _boundary(year - 1, 10)
        january = _boundary(year)
        inner = (
            InnerFold(f"{year}_inner_q2", TimeBlock(start, april), TimeBlock(april, july)),
            InnerFold(f"{year}_inner_q3", TimeBlock(start, july), TimeBlock(july, october)),
        )
        folds.append(
            WalkForwardFold(
                name=f"test_{year}",
                train=TimeBlock(start, october),
                calibration=TimeBlock(october, january),
                test=TimeBlock(january, _boundary(year + 1)),
                inner_folds=inner,
            )
        )
    return tuple(folds)


def required_gap_minutes(horizons_minutes: Sequence[int], *, latency_minutes: int = 1) -> int:
    """Return the common conservative gap, including execution latency."""

    if (
        isinstance(latency_minutes, bool)
        or not isinstance(latency_minutes, int)
        or latency_minutes < 0
    ):
        raise WalkForwardError("latency_minutes must be a nonnegative integer")
    if not horizons_minutes or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in horizons_minutes
    ):
        raise WalkForwardError("horizons_minutes must contain positive integers")
    return max(horizons_minutes) + latency_minutes


def validate_development_frame(
    frame: pd.DataFrame,
    *,
    guard_start: pd.Timestamp | datetime | str = "2025-01-01T00:00:00Z",
) -> None:
    """Reject unsafe *input* observations, even when a later filter hides them.

    Market, prediction, label, and feature timestamps must already be UTC;
    local/naive timestamps are never silently reinterpreted. Ingestion and
    artifact creation timestamps are deliberately not market-observation
    timestamps and need not predate the holdout boundary.
    """

    guard = _development_guard(guard_start)
    if frame.columns.duplicated().any():
        raise WalkForwardError("evaluation frame contains duplicate columns")
    missing = set(_REQUIRED_TIME_COLUMNS).difference(frame.columns)
    if missing:
        raise WalkForwardError(f"missing evaluation timestamp columns: {sorted(missing)}")
    timestamp_columns = [
        column
        for column in frame.columns
        if str(column).endswith("_time_utc") or column in _ADDITIONAL_TIME_COLUMNS
    ]
    for column in timestamp_columns:
        values = frame[column]
        if not isinstance(values.dtype, pd.DatetimeTZDtype) or str(values.dt.tz) != "UTC":
            raise WalkForwardError(f"{column} must contain timezone-aware UTC datetimes")
        if values.isna().any():
            raise WalkForwardError(f"{column} contains missing timestamps")
        if values.ge(guard).any():
            raise WalkForwardError(f"{column} would expose the reserved holdout")
    prediction = frame["prediction_time_utc"]
    label_end = frame["label_end_time_utc"]
    if label_end.le(prediction).any():
        raise WalkForwardError("labels must end strictly after prediction time")
    if "entry_time_utc" in frame:
        entry = frame["entry_time_utc"]
        if entry.le(prediction).any() or label_end.le(entry).any():
            raise WalkForwardError("entry must follow prediction and precede label end")
    for column in _ADDITIONAL_TIME_COLUMNS.intersection(frame.columns):
        if frame[column].gt(prediction).any():
            raise WalkForwardError(f"{column} must not follow prediction time")
    if {"feature_window_start_utc", "feature_available_at_utc"}.issubset(frame.columns) and (
        frame["feature_window_start_utc"].gt(frame["feature_available_at_utc"]).any()
    ):
        raise WalkForwardError("feature window starts after feature availability")


def select_block(
    frame: pd.DataFrame,
    block: TimeBlock,
    *,
    gap_minutes: int = 31,
    guard_start: pd.Timestamp | datetime | str = "2025-01-01T00:00:00Z",
    purge: bool = True,
) -> pd.DataFrame:
    """Select fully matured observations; apply the common gap only to training.

    This function preserves input ordering and indices and returns a copy.
    It never repairs, truncates, or downloads observations from the holdout.
    """

    if isinstance(gap_minutes, bool) or not isinstance(gap_minutes, int) or gap_minutes < 0:
        raise WalkForwardError("gap_minutes must be a nonnegative integer")
    if not isinstance(purge, bool):
        raise WalkForwardError("purge must be a boolean")
    guard = _development_guard(guard_start)
    if block.end > guard:
        raise WalkForwardError("evaluation block would expose the reserved holdout")
    validate_development_frame(frame, guard_start=guard)
    prediction_end = block.end - pd.Timedelta(minutes=gap_minutes if purge else 0)
    selected = (
        frame["prediction_time_utc"].ge(block.start)
        & frame["prediction_time_utc"].lt(prediction_end)
        & frame["label_end_time_utc"].lt(block.end)
    )
    return frame.loc[selected].copy()


def summarize_fold_metrics(
    metrics: Sequence[Mapping[str, float | int | None]],
    *,
    higher_is_better: Mapping[str, bool],
) -> dict[str, dict[str, float | int | bool | None]]:
    """Aggregate folds with equal weight and an explicit worst-case direction.

    Null metrics remain visible through observed/missing counts. If any fold
    lacks a metric, its all-fold worst value is unknown, not the worst of a
    conveniently selected subset. Means and medians use available values.
    """

    if not metrics or not higher_is_better:
        raise WalkForwardError("fold metrics and metric directions must not be empty")
    summary: dict[str, dict[str, float | int | bool | None]] = {}
    for name, higher in higher_is_better.items():
        if not isinstance(higher, bool):
            raise WalkForwardError(f"metric direction for {name} must be boolean")
        values: list[float] = []
        for fold in metrics:
            if name not in fold:
                raise WalkForwardError(f"fold is missing metric: {name}")
            value = fold[name]
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise WalkForwardError(f"metric {name} must be finite numeric or null")
            values.append(float(value))
        missing = len(metrics) - len(values)
        summary[name] = {
            "fold_count": len(metrics),
            "observed_fold_count": len(values),
            "missing_fold_count": missing,
            "higher_is_better": higher,
            "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "worst": (min(values) if higher else max(values)) if values and not missing else None,
        }
    return summary
