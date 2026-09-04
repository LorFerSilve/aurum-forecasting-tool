"""Leakage-separated 15-minute labels from exact future one-minute opens."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_forecasting.classification import CLASS_TO_INDEX
from gold_forecasting.config import LabelsConfig
from gold_forecasting.validation import CandleValidationError, validate_candles

LABEL_COLUMNS = (
    "instrument",
    "source",
    "prediction_time_utc",
    "entry_time_utc",
    "label_end_time_utc",
    "horizon_minutes",
    "entry_bid_open",
    "exit_bid_open",
    "future_return_bps",
    "target_class",
    "target_class_id",
    "entry_raw_file_hash",
    "exit_raw_file_hash",
    "entry_dataset_version",
    "exit_dataset_version",
)


class LabelBuildError(ValueError):
    """Raised when the executable-price label contract cannot be satisfied."""


@dataclass(frozen=True, slots=True)
class LabelBuildResult:
    labels: pd.DataFrame
    candidate_count: int
    output_row_count: int
    dropped_missing_path: int


def classify_future_returns(
    returns_bps: pd.Series,
    *,
    threshold_bps: float,
) -> pd.Series:
    """Apply strict outer thresholds and an inclusive neutral interval."""

    if not np.isfinite(threshold_bps) or threshold_bps <= 0:
        raise LabelBuildError("threshold_bps must be finite and positive")
    numeric = pd.to_numeric(returns_bps, errors="coerce").astype("float64")
    if not np.isfinite(numeric.to_numpy()).all():
        raise LabelBuildError("future returns must contain only finite values")
    labels = np.select(
        [numeric.lt(-threshold_bps), numeric.gt(threshold_bps)],
        ["down", "up"],
        default="neutral",
    )
    return pd.Series(labels, index=returns_bps.index, dtype="string")


def _normalize_prediction_times(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"instrument", "source", "prediction_time_utc"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise LabelBuildError(f"prediction candidates miss columns: {', '.join(missing)}")
    candidates = predictions.loc[:, sorted(required)].copy()
    normalized: list[pd.Timestamp] = []
    for index, raw in candidates["prediction_time_utc"].items():
        try:
            timestamp = pd.Timestamp(raw)
        except (TypeError, ValueError) as exc:
            raise LabelBuildError(f"invalid prediction timestamp at index {index!r}") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise LabelBuildError("prediction timestamps must be timezone-aware")
        normalized.append(timestamp.tz_convert("UTC"))
    candidates["prediction_time_utc"] = pd.DatetimeIndex(
        normalized, dtype="datetime64[ns, UTC]"
    )
    for column in ("instrument", "source"):
        if candidates[column].map(lambda value: not isinstance(value, str) or not value).any():
            raise LabelBuildError(f"prediction {column} values must be non-empty strings")
    aligned = candidates["prediction_time_utc"].eq(
        candidates["prediction_time_utc"].dt.floor("3min")
    )
    if not aligned.all():
        raise LabelBuildError("prediction timestamps must align to the three-minute UTC grid")
    key = ["instrument", "source", "prediction_time_utc"]
    if candidates.duplicated(subset=key).any():
        raise LabelBuildError("prediction candidates contain duplicate keys")
    return candidates.sort_values(key, kind="stable").reset_index(drop=True)


def _one_minute_segments(candles: pd.DataFrame) -> pd.DataFrame:
    frame = candles.sort_values(
        ["instrument", "source", "timestamp_open_utc"], kind="stable"
    ).reset_index(drop=True)
    group_changed = (
        frame["instrument"].ne(frame["instrument"].shift())
        | frame["source"].ne(frame["source"].shift())
    )
    gap = frame["timestamp_open_utc"].diff().ne(pd.Timedelta(minutes=1))
    frame["_segment_id"] = (group_changed | gap).cumsum().astype("int64")
    frame["_segment_position"] = frame.groupby("_segment_id", sort=False).cumcount()
    return frame


def build_mvp_labels(
    candles_1min: pd.DataFrame,
    prediction_candidates: pd.DataFrame,
    config: LabelsConfig,
) -> LabelBuildResult:
    """Label predictions using entry at ``t+1min`` and exit at ``t+16min``.

    Exact timestamp joins and segment positions prove that every minute from
    entry through exit exists. No positional future shift or interpolation is
    used, and no future value enters the feature table.
    """

    try:
        validated = validate_candles(candles_1min, expected_timeframe="1min").candles
    except CandleValidationError as exc:
        raise LabelBuildError(f"invalid 1min label input: {exc}") from exc
    candidates = _normalize_prediction_times(prediction_candidates)
    if candidates.empty:
        raise LabelBuildError("at least one prediction candidate is required")

    latency = config.prediction.decision_latency_minutes
    horizon = config.prediction.primary_horizon_minutes
    candidates["entry_time_utc"] = candidates["prediction_time_utc"] + pd.to_timedelta(
        latency, unit="min"
    )
    candidates["label_end_time_utc"] = candidates["entry_time_utc"] + pd.to_timedelta(
        horizon, unit="min"
    )

    market = _one_minute_segments(validated)
    lookup_columns = [
        "instrument",
        "source",
        "timestamp_open_utc",
        "bid_open",
        "raw_file_hash",
        "dataset_version",
        "_segment_id",
        "_segment_position",
    ]
    entry = market.loc[:, lookup_columns].rename(
        columns={
            "timestamp_open_utc": "entry_time_utc",
            "bid_open": "entry_bid_open",
            "raw_file_hash": "entry_raw_file_hash",
            "dataset_version": "entry_dataset_version",
            "_segment_id": "_entry_segment",
            "_segment_position": "_entry_position",
        }
    )
    exit_prices = market.loc[:, lookup_columns].rename(
        columns={
            "timestamp_open_utc": "label_end_time_utc",
            "bid_open": "exit_bid_open",
            "raw_file_hash": "exit_raw_file_hash",
            "dataset_version": "exit_dataset_version",
            "_segment_id": "_exit_segment",
            "_segment_position": "_exit_position",
        }
    )
    joined = candidates.merge(
        entry,
        on=["instrument", "source", "entry_time_utc"],
        how="left",
        validate="one_to_one",
    ).merge(
        exit_prices,
        on=["instrument", "source", "label_end_time_utc"],
        how="left",
        validate="one_to_one",
    )
    complete_path = (
        joined["_entry_segment"].notna()
        & joined["_entry_segment"].eq(joined["_exit_segment"])
        & joined["_exit_position"].sub(joined["_entry_position"]).eq(horizon)
    )
    joined = joined.loc[complete_path].copy()
    if not joined.empty:
        joined["future_return_bps"] = 10_000.0 * np.log(
            joined["exit_bid_open"] / joined["entry_bid_open"]
        )
        joined["target_class"] = classify_future_returns(
            joined["future_return_bps"],
            threshold_bps=config.classes.total_threshold_bps,
        )
        joined["target_class_id"] = joined["target_class"].map(CLASS_TO_INDEX).astype("int8")
    else:
        joined["future_return_bps"] = pd.Series(dtype="float64")
        joined["target_class"] = pd.Series(dtype="string")
        joined["target_class_id"] = pd.Series(dtype="int8")
    joined["horizon_minutes"] = horizon

    labels = joined.loc[:, list(LABEL_COLUMNS)].sort_values(
        ["instrument", "source", "prediction_time_utc"], kind="stable"
    ).reset_index(drop=True)
    if not labels.empty:
        if not labels["entry_time_utc"].gt(labels["prediction_time_utc"]).all():
            raise LabelBuildError("entry time must be strictly after prediction time")
        if not labels["label_end_time_utc"].gt(labels["entry_time_utc"]).all():
            raise LabelBuildError("label end must be strictly after entry time")
        numeric_return = labels["future_return_bps"]
        if not np.isfinite(numeric_return.to_numpy(dtype=np.float64)).all():
            raise LabelBuildError("label calculation produced non-finite returns")

    return LabelBuildResult(
        labels=labels,
        candidate_count=len(candidates),
        output_row_count=len(labels),
        dropped_missing_path=len(candidates) - len(labels),
    )


__all__ = [
    "LABEL_COLUMNS",
    "LabelBuildError",
    "LabelBuildResult",
    "build_mvp_labels",
    "classify_future_returns",
]
