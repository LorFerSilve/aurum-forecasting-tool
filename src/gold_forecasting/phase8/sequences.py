"""Causal fixed-length candle sequences for the phase-8 neural challenger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from gold_forecasting.validation import CandleValidationError, validate_candles

SEQUENCE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "log_return_1_bps",
    "body_bps",
    "range_bps",
    "upper_wick_bps",
    "lower_wick_bps",
    "close_position",
)
_TIMEFRAME_MINUTES: Final[dict[str, int]] = {
    "1min": 1,
    "3min": 3,
    "15min": 15,
    "1h": 60,
}


class Phase8SequenceError(ValueError):
    """Raised when sequence construction cannot satisfy the causal time contract."""


@dataclass(frozen=True, slots=True)
class SequenceFrame:
    """Dense sequence tensor plus explicit per-sample availability."""

    timeframe: str
    values: np.ndarray
    available: np.ndarray
    source_close_utc: pd.Series
    window_start_utc: pd.Series


@dataclass(frozen=True, slots=True)
class Phase8SequenceBuildResult:
    prediction_times: pd.Series
    by_timeframe: dict[str, SequenceFrame]
    diagnostics: dict[str, object]


def _segment_ids(frame: pd.DataFrame, *, minutes: int) -> np.ndarray:
    changed = (
        frame["instrument"].ne(frame["instrument"].shift())
        | frame["source"].ne(frame["source"].shift())
    )
    broken = frame["timestamp_open_utc"].diff().ne(pd.Timedelta(minutes=minutes))
    return (changed | broken).cumsum().to_numpy(dtype=np.int64)


def _candle_features(frame: pd.DataFrame, segment: np.ndarray) -> np.ndarray:
    open_price = frame["bid_open"].to_numpy(dtype=np.float64)
    high = frame["bid_high"].to_numpy(dtype=np.float64)
    low = frame["bid_low"].to_numpy(dtype=np.float64)
    close = frame["bid_close"].to_numpy(dtype=np.float64)
    previous = np.roll(close, 1)
    previous[0] = np.nan
    segment_start = np.empty(len(frame), dtype=bool)
    segment_start[0] = True
    segment_start[1:] = segment[1:] != segment[:-1]
    previous[segment_start] = np.nan
    log_return = 10_000.0 * np.log(close / previous)
    body = 10_000.0 * (close - open_price) / open_price
    candle_range = high - low
    range_bps = 10_000.0 * candle_range / open_price
    upper = 10_000.0 * (high - np.maximum(open_price, close)) / open_price
    lower = 10_000.0 * (np.minimum(open_price, close) - low) / open_price
    close_position = np.divide(
        close - low,
        candle_range,
        out=np.full_like(close, 0.5),
        where=candle_range != 0.0,
    )
    return np.column_stack(
        (log_return, body, range_bps, upper, lower, close_position)
    ).astype(np.float32)


def build_timeframe_sequences(
    candles: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    *,
    timeframe: str,
    sequence_length: int,
) -> SequenceFrame:
    """Align only fully closed contiguous source sequences at or before prediction time."""

    if timeframe not in _TIMEFRAME_MINUTES:
        raise Phase8SequenceError(f"unsupported phase-8 timeframe: {timeframe}")
    if sequence_length < 2:
        raise Phase8SequenceError("sequence_length must be at least two candles")
    try:
        source = validate_candles(candles, expected_timeframe=timeframe).candles
    except CandleValidationError as exc:
        raise Phase8SequenceError(f"invalid {timeframe} source: {exc}") from exc
    if source.empty or prediction_rows.empty:
        raise Phase8SequenceError("source candles and prediction rows must be nonempty")
    required = {"instrument", "source", "prediction_time_utc"}
    if not required.issubset(prediction_rows.columns):
        raise Phase8SequenceError("prediction rows miss the alignment key")

    source = source.sort_values(
        ["instrument", "source", "timestamp_open_utc"], kind="stable"
    ).reset_index(drop=True)
    predictions = prediction_rows.sort_values(
        ["instrument", "source", "prediction_time_utc"], kind="stable"
    ).reset_index(drop=True)
    if source[["instrument", "source"]].drop_duplicates().shape[0] != 1:
        raise Phase8SequenceError("phase-8-v1 expects one instrument/source stream")
    stream = source[["instrument", "source"]].iloc[0]
    if not (
        predictions["instrument"].eq(stream["instrument"]).all()
        and predictions["source"].eq(stream["source"]).all()
    ):
        raise Phase8SequenceError("prediction rows must match the source instrument/source")

    minutes = _TIMEFRAME_MINUTES[timeframe]
    segment = _segment_ids(source, minutes=minutes)
    features = _candle_features(source, segment)
    close_times = source["timestamp_close_utc"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(
        dtype="datetime64[ns]"
    )
    prediction_times = (
        predictions["prediction_time_utc"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    last = np.searchsorted(close_times, prediction_times, side="right") - 1
    valid_last = last >= 0
    safe_last = np.clip(last, 0, len(source) - 1)
    age_minutes = (
        prediction_times - close_times[safe_last]
    ).astype("timedelta64[s]").astype(np.float64) / 60.0
    valid = valid_last & (age_minutes >= 0.0) & (age_minutes <= float(minutes))
    start = safe_last - sequence_length + 1
    valid &= start >= 1
    safe_start = np.clip(start, 0, len(source) - 1)
    valid &= segment[safe_start] == segment[safe_last]

    offsets = np.arange(sequence_length - 1, -1, -1, dtype=np.int64)
    indices = safe_last[:, None] - offsets[None, :]
    safe_indices = np.clip(indices, 0, len(source) - 1)
    values = features[safe_indices].copy()
    finite = np.isfinite(values).all(axis=(1, 2))
    valid &= finite
    values[~valid] = np.nan

    available = valid.astype(bool)
    source_close = pd.Series(pd.NaT, index=predictions.index, dtype="datetime64[ns, UTC]")
    window_start = pd.Series(pd.NaT, index=predictions.index, dtype="datetime64[ns, UTC]")
    if available.any():
        rows = np.flatnonzero(available)
        source_close.iloc[rows] = source.loc[
            safe_last[rows], "timestamp_close_utc"
        ].to_numpy()
        window_start.iloc[rows] = source.loc[
            safe_start[rows], "timestamp_open_utc"
        ].to_numpy()
    if source_close.dropna().gt(
        predictions.loc[source_close.notna(), "prediction_time_utc"]
    ).any():
        raise Phase8SequenceError("aligned source candle closes after prediction time")
    return SequenceFrame(
        timeframe=timeframe,
        values=values.astype(np.float32, copy=False),
        available=available,
        source_close_utc=source_close,
        window_start_utc=window_start,
    )


def build_phase8_sequences(
    candles_by_timeframe: dict[str, pd.DataFrame],
    prediction_rows: pd.DataFrame,
    sequence_lengths: dict[str, int],
) -> Phase8SequenceBuildResult:
    """Build every configured timeframe against one immutable prediction index."""

    ordered = prediction_rows.sort_values(
        ["instrument", "source", "prediction_time_utc"], kind="stable"
    ).reset_index(drop=True)
    frames: dict[str, SequenceFrame] = {}
    timeframe_diagnostics: dict[str, Any] = {}
    for timeframe, sequence_length in sequence_lengths.items():
        if timeframe not in candles_by_timeframe:
            raise Phase8SequenceError(f"missing curated source for {timeframe}")
        frame = build_timeframe_sequences(
            candles_by_timeframe[timeframe],
            ordered,
            timeframe=timeframe,
            sequence_length=sequence_length,
        )
        frames[timeframe] = frame
        timeframe_diagnostics[timeframe] = {
            "sequence_length": sequence_length,
            "available_rows": int(frame.available.sum()),
            "available_fraction": float(frame.available.mean()),
            "feature_count": len(SEQUENCE_FEATURE_NAMES),
        }
    diagnostics: dict[str, object] = {
        "rows": len(ordered),
        "timeframes": timeframe_diagnostics,
    }
    return Phase8SequenceBuildResult(
        prediction_times=ordered["prediction_time_utc"].copy(),
        by_timeframe=frames,
        diagnostics=diagnostics,
    )


def subset_phase8_sequences(
    built: Phase8SequenceBuildResult,
    row_indices: np.ndarray,
) -> Phase8SequenceBuildResult:
    """Return an aligned subset for one horizon model table."""

    rows = np.asarray(row_indices, dtype=np.int64)
    if (
        rows.ndim != 1
        or len(rows) == 0
        or (rows < 0).any()
        or (rows >= len(built.prediction_times)).any()
    ):
        raise Phase8SequenceError("sequence subset rows are invalid")
    frames: dict[str, SequenceFrame] = {}
    for name, frame in built.by_timeframe.items():
        frames[name] = SequenceFrame(
            timeframe=name,
            values=frame.values[rows],
            available=frame.available[rows],
            source_close_utc=frame.source_close_utc.iloc[rows].reset_index(drop=True),
            window_start_utc=frame.window_start_utc.iloc[rows].reset_index(drop=True),
        )
    return Phase8SequenceBuildResult(
        prediction_times=built.prediction_times.iloc[rows].reset_index(drop=True),
        by_timeframe=frames,
        diagnostics={"rows": len(rows), "subset_of_rows": len(built.prediction_times)},
    )


__all__ = [
    "Phase8SequenceBuildResult",
    "Phase8SequenceError",
    "SEQUENCE_FEATURE_NAMES",
    "SequenceFrame",
    "build_phase8_sequences",
    "build_timeframe_sequences",
    "subset_phase8_sequences",
]
