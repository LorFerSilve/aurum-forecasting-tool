"""Exact five-candle 3min path targets for phase 9.

The path starts with the first 3min candle opening exactly at prediction time.
Five contiguous future candles therefore cover [t, t+15min]. These targets are
future-only labels and are never model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gold_forecasting.labels.multihorizon import DEVELOPMENT_END_EXCLUSIVE_UTC
from gold_forecasting.labels.mvp import LabelBuildError, _normalize_prediction_times
from gold_forecasting.validation import CandleValidationError, validate_candles

PATH_STEPS: Final[int] = 5
PATH_COMPONENTS: Final[tuple[str, ...]] = (
    "gap_log_bps",
    "body_log_bps",
    "upper_wick_log_bps",
    "lower_wick_log_bps",
)
OHLC_COMPONENTS: Final[tuple[str, ...]] = ("open", "high", "low", "close")


class FuturePathError(LabelBuildError):
    """Raised when a five-candle future path cannot satisfy its exact contract."""


@dataclass(frozen=True, slots=True)
class FuturePathBuildResult:
    labels: pd.DataFrame
    candidate_count: int
    output_row_count: int
    dropped_missing_path: int


def _segments(candles: pd.DataFrame) -> pd.DataFrame:
    frame = candles.sort_values(
        ["instrument", "source", "timestamp_open_utc"],
        kind="stable",
    ).reset_index(drop=True)
    changed = (
        frame["instrument"].ne(frame["instrument"].shift())
        | frame["source"].ne(frame["source"].shift())
    )
    gap = frame["timestamp_open_utc"].diff().ne(pd.Timedelta(minutes=3))
    frame["_segment_id"] = (changed | gap).cumsum().astype("int64")
    frame["_market_row"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _representation_from_ohlc(
    previous_close: NDArray[np.float64],
    open_price: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
) -> NDArray[np.float64]:
    gap = 10_000.0 * np.log(open_price / previous_close)
    body = 10_000.0 * np.log(close / open_price)
    upper = 10_000.0 * np.log(high / np.maximum(open_price, close))
    lower = 10_000.0 * np.log(np.minimum(open_price, close) / low)
    result = np.column_stack((gap, body, upper, lower)).astype(np.float64)
    if not np.isfinite(result).all():
        raise FuturePathError("future-path representation became non-finite")
    if (result[:, 2:] < -1e-9).any():
        raise FuturePathError("future-path wick representation became negative")
    result[:, 2:] = np.maximum(result[:, 2:], 0.0)
    return result


def reconstruct_path(
    anchor_close: np.ndarray,
    representation: np.ndarray,
    *,
    clip_log_bps: float = 5_000.0,
) -> NDArray[np.float64]:
    """Reconstruct positive, structurally valid OHLC from log-bps components."""

    anchor = np.asarray(anchor_close, dtype=np.float64)
    values = np.asarray(representation, dtype=np.float64)
    if anchor.ndim != 1 or values.ndim != 3 or values.shape != (
        len(anchor),
        PATH_STEPS,
        len(PATH_COMPONENTS),
    ):
        raise FuturePathError(
            "reconstruction requires anchor [rows] and representation [rows,5,4]"
        )
    if (
        not np.isfinite(anchor).all()
        or not (anchor > 0.0).all()
        or not np.isfinite(values).all()
        or not np.isfinite(clip_log_bps)
        or clip_log_bps <= 0.0
    ):
        raise FuturePathError("reconstruction inputs must be finite and positive where required")

    output = np.empty((len(anchor), PATH_STEPS, 4), dtype=np.float64)
    previous = anchor.copy()
    for step in range(PATH_STEPS):
        gap = np.clip(values[:, step, 0], -clip_log_bps, clip_log_bps)
        body = np.clip(values[:, step, 1], -clip_log_bps, clip_log_bps)
        upper = np.clip(values[:, step, 2], 0.0, clip_log_bps)
        lower = np.clip(values[:, step, 3], 0.0, clip_log_bps)
        open_price = previous * np.exp(gap / 10_000.0)
        close = open_price * np.exp(body / 10_000.0)
        high = np.maximum(open_price, close) * np.exp(upper / 10_000.0)
        low = np.minimum(open_price, close) * np.exp(-lower / 10_000.0)
        output[:, step, 0] = open_price
        output[:, step, 1] = high
        output[:, step, 2] = low
        output[:, step, 3] = close
        previous = close
    if not np.isfinite(output).all() or not (output > 0.0).all():
        raise FuturePathError("reconstructed path contains invalid prices")
    if (
        (output[:, :, 1] < np.maximum(output[:, :, 0], output[:, :, 3])).any()
        or (output[:, :, 2] > np.minimum(output[:, :, 0], output[:, :, 3])).any()
        or (output[:, :, 1] < output[:, :, 2]).any()
    ):
        raise FuturePathError("reconstructed path violates OHLC invariants")
    return output


def aggregate_path_ohlc(path_ohlc: np.ndarray) -> NDArray[np.float64]:
    values = np.asarray(path_ohlc, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (PATH_STEPS, 4):
        raise FuturePathError("path OHLC must have shape [rows,5,4]")
    result = np.column_stack(
        (
            values[:, 0, 0],
            values[:, :, 1].max(axis=1),
            values[:, :, 2].min(axis=1),
            values[:, -1, 3],
        )
    )
    return result.astype(np.float64)


def build_future_path_targets(
    candles_3min: pd.DataFrame,
    prediction_candidates: pd.DataFrame,
) -> FuturePathBuildResult:
    """Build five exact contiguous future 3min candles after each prediction."""

    try:
        validated = validate_candles(
            candles_3min,
            expected_timeframe="3min",
        ).candles
    except CandleValidationError as exc:
        raise FuturePathError(f"invalid 3min path input: {exc}") from exc
    candidates = _normalize_prediction_times(prediction_candidates)
    if candidates.empty:
        raise FuturePathError("at least one prediction candidate is required")
    if validated["timestamp_open_utc"].ge(DEVELOPMENT_END_EXCLUSIVE_UTC).any():
        raise FuturePathError("3min path input would expose the reserved holdout")
    if candidates["prediction_time_utc"].ge(DEVELOPMENT_END_EXCLUSIVE_UTC).any():
        raise FuturePathError("prediction candidates would expose the reserved holdout")

    market = _segments(validated)
    anchor = market.loc[
        :,
        [
            "instrument",
            "source",
            "timestamp_close_utc",
            "bid_close",
            "_segment_id",
            "_market_row",
        ],
    ].rename(
        columns={
            "timestamp_close_utc": "prediction_time_utc",
            "bid_close": "path_anchor_close",
            "_segment_id": "_anchor_segment",
            "_market_row": "_anchor_row",
        }
    )
    joined = candidates.merge(
        anchor,
        on=["instrument", "source", "prediction_time_utc"],
        how="left",
        validate="one_to_one",
    )

    anchor_present = joined["_anchor_row"].notna().to_numpy()
    anchor_rows = joined["_anchor_row"].fillna(0).to_numpy(dtype=np.int64)
    future_indices = anchor_rows[:, None] + np.arange(1, PATH_STEPS + 1)
    in_bounds = future_indices[:, -1] < len(market)
    safe = np.clip(future_indices, 0, max(len(market) - 1, 0))
    segment = market["_segment_id"].to_numpy(dtype=np.int64)
    same_segment = np.ones(len(joined), dtype=bool)
    if len(market):
        same_segment = (
            segment[safe]
            == joined["_anchor_segment"].fillna(-1).to_numpy(dtype=np.int64)[:, None]
        ).all(axis=1)
    expected_open = np.column_stack(
        [
            joined["prediction_time_utc"]
            + pd.Timedelta(minutes=3 * step)
            for step in range(PATH_STEPS)
        ]
    )
    actual_open = np.empty((len(joined), PATH_STEPS), dtype=object)
    for step in range(PATH_STEPS):
        actual_open[:, step] = market.iloc[safe[:, step]][
            "timestamp_open_utc"
        ].to_numpy()
    exact_timestamps = np.ones(len(joined), dtype=bool)
    for step in range(PATH_STEPS):
        exact_timestamps &= pd.Series(actual_open[:, step]).reset_index(drop=True).eq(
            pd.Series(expected_open[:, step]).reset_index(drop=True)
        ).to_numpy()
    path_end = joined["prediction_time_utc"] + pd.Timedelta(minutes=15)
    eligible = (
        anchor_present
        & in_bounds
        & same_segment
        & exact_timestamps
        & path_end.lt(DEVELOPMENT_END_EXCLUSIVE_UTC).to_numpy()
    )

    rows = np.flatnonzero(eligible)
    labels = joined.iloc[rows][
        ["instrument", "source", "prediction_time_utc", "path_anchor_close"]
    ].copy()
    labels["path_end_time_utc"] = path_end.iloc[rows].to_numpy()

    if len(rows):
        selected = safe[rows]
        path_ohlc = np.empty((len(rows), PATH_STEPS, 4), dtype=np.float64)
        previous = labels["path_anchor_close"].to_numpy(dtype=np.float64)
        representation = np.empty(
            (len(rows), PATH_STEPS, len(PATH_COMPONENTS)),
            dtype=np.float64,
        )
        for step in range(PATH_STEPS):
            frame = market.iloc[selected[:, step]]
            open_price = frame["bid_open"].to_numpy(dtype=np.float64)
            high = frame["bid_high"].to_numpy(dtype=np.float64)
            low = frame["bid_low"].to_numpy(dtype=np.float64)
            close = frame["bid_close"].to_numpy(dtype=np.float64)
            path_ohlc[:, step] = np.column_stack((open_price, high, low, close))
            representation[:, step] = _representation_from_ohlc(
                previous,
                open_price,
                high,
                low,
                close,
            )
            labels[f"path_step_{step + 1}_open_time_utc"] = frame[
                "timestamp_open_utc"
            ].to_numpy()
            labels[f"path_step_{step + 1}_close_time_utc"] = frame[
                "timestamp_close_utc"
            ].to_numpy()
            for component_index, component in enumerate(OHLC_COMPONENTS):
                labels[f"path_step_{step + 1}_bid_{component}"] = path_ohlc[
                    :, step, component_index
                ]
            for component_index, component in enumerate(PATH_COMPONENTS):
                labels[f"path_step_{step + 1}_{component}"] = representation[
                    :, step, component_index
                ]
            previous = close

        aggregate = aggregate_path_ohlc(path_ohlc)
        aggregate_representation = _representation_from_ohlc(
            labels["path_anchor_close"].to_numpy(dtype=np.float64),
            aggregate[:, 0],
            aggregate[:, 1],
            aggregate[:, 2],
            aggregate[:, 3],
        )
        for component_index, component in enumerate(OHLC_COMPONENTS):
            labels[f"path_15m_bid_{component}"] = aggregate[:, component_index]
        for component_index, component in enumerate(PATH_COMPONENTS):
            labels[f"path_15m_{component}"] = aggregate_representation[:, component_index]
        labels["path_15m_close_return_log_bps"] = (
            10_000.0
            * np.log(
                aggregate[:, 3]
                / labels["path_anchor_close"].to_numpy(dtype=np.float64)
            )
        )

        rebuilt = reconstruct_path(
            labels["path_anchor_close"].to_numpy(dtype=np.float64),
            representation,
        )
        if not np.allclose(rebuilt, path_ohlc, rtol=0.0, atol=1e-9):
            raise FuturePathError("future-path representation does not reconstruct exactly")

    labels = labels.sort_values(
        ["instrument", "source", "prediction_time_utc"],
        kind="stable",
    ).reset_index(drop=True)
    return FuturePathBuildResult(
        labels=labels,
        candidate_count=len(candidates),
        output_row_count=len(labels),
        dropped_missing_path=len(candidates) - len(labels),
    )


__all__ = [
    "OHLC_COMPONENTS",
    "PATH_COMPONENTS",
    "PATH_STEPS",
    "FuturePathBuildResult",
    "FuturePathError",
    "aggregate_path_ohlc",
    "build_future_path_targets",
    "reconstruct_path",
]
