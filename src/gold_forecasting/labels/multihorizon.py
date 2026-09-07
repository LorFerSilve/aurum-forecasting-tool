"""Versioned, exact-execution multi-horizon labels for development research.

Only observed, complete one-minute bid candles are used. Range covers candles
in ``[entry, exit)``; realized volatility covers the ``horizon_minutes``
open-to-open returns from entry through the exit open. These future quantities
are targets, never features. The reserved 2025-and-later data cannot be supplied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from gold_forecasting.classification import CLASS_TO_INDEX
from gold_forecasting.labels.mvp import (
    LABEL_COLUMNS,
    LabelBuildError,
    LabelBuildResult,
    _normalize_prediction_times,
    _one_minute_segments,
    classify_future_returns,
)
from gold_forecasting.validation import CandleValidationError, validate_candles

DEVELOPMENT_END_EXCLUSIVE_UTC = pd.Timestamp("2025-01-01T00:00:00Z")
MULTIHORIZON_LABEL_COLUMNS = (
    *LABEL_COLUMNS,
    "label_spec_version",
    "execution_latency_minutes",
    "neutral_threshold_bps",
    "arithmetic_return_bps",
    "future_range_bps",
    "future_realized_vol_bps",
)
_TARGET_NUMERIC_COLUMNS = (
    "future_return_bps",
    "arithmetic_return_bps",
    "future_range_bps",
    "future_realized_vol_bps",
)


def _validate_parameters(
    horizon_minutes: int,
    execution_latency_minutes: int,
    neutral_threshold_bps: float,
    label_spec_version: str,
) -> None:
    for name, value in (
        ("horizon_minutes", horizon_minutes),
        ("execution_latency_minutes", execution_latency_minutes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise LabelBuildError(f"{name} must be a positive integer")
    if (
        isinstance(neutral_threshold_bps, bool)
        or not np.isfinite(neutral_threshold_bps)
        or neutral_threshold_bps <= 0
    ):
        raise LabelBuildError("neutral_threshold_bps must be finite and positive")
    if not isinstance(label_spec_version, str) or not label_spec_version.strip():
        raise LabelBuildError("label_spec_version must be a non-empty string")


def _validate_optional_ask_open(candles: pd.DataFrame) -> bool:
    if "ask_open" not in candles:
        return False
    asks = candles["ask_open"]
    if not is_numeric_dtype(asks.dtype) or is_bool_dtype(asks.dtype):
        raise LabelBuildError("ask_open must be numeric when supplied")
    values = asks.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all() or not (values > 0).all():
        raise LabelBuildError("ask_open must contain finite, strictly positive prices")
    if not asks.ge(candles["bid_open"]).all():
        raise LabelBuildError("ask_open must be greater than or equal to bid_open")
    return True


def _add_path_targets(market: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Vectorized forward windows, used only after the complete-segment check.

    Rolling operations may cross adjacent segments in this intermediate table.
    Such rows are never emitted: the exact entry/exit segment proof below must
    establish that all H+1 opens belong to one uninterrupted instrument/source.
    """

    market["_future_high"] = (
        market["bid_high"].rolling(horizon, min_periods=horizon).max().shift(1 - horizon)
    )
    market["_future_low"] = (
        market["bid_low"].rolling(horizon, min_periods=horizon).min().shift(1 - horizon)
    )
    log_returns = np.log(market["bid_open"]).diff() * 10_000.0
    # Returning zero at a new segment avoids a spurious cross-market jump in
    # intermediate windows. Valid emitted windows exclude that transition.
    log_returns = log_returns.where(market["_segment_position"].ne(0), 0.0)
    market["_future_squared_returns"] = (
        log_returns.pow(2).rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    )
    return market


def build_horizon_labels(
    candles_1min: pd.DataFrame,
    prediction_candidates: pd.DataFrame,
    *,
    horizon_minutes: int,
    execution_latency_minutes: int = 1,
    neutral_threshold_bps: float = 6.0,
    label_spec_version: str = "v0.2",
) -> LabelBuildResult:
    """Build continuous and three-class targets with exact observed execution.

    The neutral zone is supplied explicitly (normally assumed round-trip costs
    plus the protocol's fixed margin). Classification uses the log return,
    preserving the MVP semantics; arithmetic returns are separately available
    for fixed-notional P&L accounting. Ask opens are not invented: both optional
    output ask columns exist only when validated input ``ask_open`` is present.

    Actual input candle opens or prediction times at/after 2025-01-01 UTC raise.
    Candidates whose requested exit reaches that boundary, or whose complete
    path is unavailable, are counted as ``dropped_missing_path``. There is no
    holdout override. The input tables are not modified.
    """

    _validate_parameters(
        horizon_minutes, execution_latency_minutes, neutral_threshold_bps, label_spec_version
    )
    try:
        validated = validate_candles(candles_1min, expected_timeframe="1min").candles
    except CandleValidationError as exc:
        raise LabelBuildError(f"invalid 1min label input: {exc}") from exc
    candidates = _normalize_prediction_times(prediction_candidates)
    if candidates.empty:
        raise LabelBuildError("at least one prediction candidate is required")
    if validated["timestamp_open_utc"].ge(DEVELOPMENT_END_EXCLUSIVE_UTC).any():
        raise LabelBuildError("1min label input would expose the reserved holdout")
    if candidates["prediction_time_utc"].ge(DEVELOPMENT_END_EXCLUSIVE_UTC).any():
        raise LabelBuildError("prediction candidates would expose the reserved holdout")
    has_asks = _validate_optional_ask_open(validated)

    try:
        candidates["entry_time_utc"] = candidates["prediction_time_utc"] + pd.Timedelta(
            minutes=execution_latency_minutes
        )
        candidates["label_end_time_utc"] = candidates["entry_time_utc"] + pd.Timedelta(
            minutes=horizon_minutes
        )
    except (OverflowError, ValueError) as exc:
        raise LabelBuildError("label timing exceeds supported timestamp bounds") from exc

    market = _add_path_targets(_one_minute_segments(validated), horizon_minutes)
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
    if has_asks:
        lookup_columns.append("ask_open")
    entry = market.loc[
        :, [*lookup_columns, "_future_high", "_future_low", "_future_squared_returns"]
    ].rename(
        columns={
            "timestamp_open_utc": "entry_time_utc",
            "bid_open": "entry_bid_open",
            "ask_open": "entry_ask_open",
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
            "ask_open": "exit_ask_open",
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
        & joined["_exit_position"].sub(joined["_entry_position"]).eq(horizon_minutes)
        & joined["label_end_time_utc"].lt(DEVELOPMENT_END_EXCLUSIVE_UTC)
    )
    joined = joined.loc[complete_path].copy()
    if not joined.empty:
        ratio = joined["exit_bid_open"] / joined["entry_bid_open"]
        joined["future_return_bps"] = 10_000.0 * np.log(ratio)
        joined["arithmetic_return_bps"] = 10_000.0 * (ratio - 1.0)
        joined["future_range_bps"] = (
            (joined["_future_high"] - joined["_future_low"]) / joined["entry_bid_open"]
        ) * 10_000.0
        joined["future_realized_vol_bps"] = np.sqrt(
            joined["_future_squared_returns"].clip(lower=0.0)
        )
        numeric = joined.loc[:, list(_TARGET_NUMERIC_COLUMNS)].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise LabelBuildError("label calculation produced non-finite targets")
        joined["target_class"] = classify_future_returns(
            joined["future_return_bps"], threshold_bps=neutral_threshold_bps
        )
        joined["target_class_id"] = joined["target_class"].map(CLASS_TO_INDEX).astype("int8")
    else:
        for column in _TARGET_NUMERIC_COLUMNS:
            joined[column] = pd.Series(dtype="float64")
        joined["target_class"] = pd.Series(dtype="string")
        joined["target_class_id"] = pd.Series(dtype="int8")
    joined["horizon_minutes"] = horizon_minutes
    joined["label_spec_version"] = label_spec_version
    joined["execution_latency_minutes"] = execution_latency_minutes
    joined["neutral_threshold_bps"] = neutral_threshold_bps
    columns = list(MULTIHORIZON_LABEL_COLUMNS)
    if has_asks:
        columns.extend(("entry_ask_open", "exit_ask_open"))
    labels = (
        joined.loc[:, columns]
        .sort_values(["instrument", "source", "prediction_time_utc"], kind="stable")
        .reset_index(drop=True)
    )
    return LabelBuildResult(
        labels=labels,
        candidate_count=len(candidates),
        output_row_count=len(labels),
        dropped_missing_path=len(candidates) - len(labels),
    )


__all__ = [
    "DEVELOPMENT_END_EXCLUSIVE_UTC",
    "MULTIHORIZON_LABEL_COLUMNS",
    "build_horizon_labels",
]
