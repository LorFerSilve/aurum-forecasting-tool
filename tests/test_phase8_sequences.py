from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold_forecasting.phase8.sequences import (
    SEQUENCE_FEATURE_NAMES,
    build_timeframe_sequences,
)

_MINUTES = {"1min": 1, "3min": 3, "15min": 15, "1h": 60}


def _candles(timeframe: str, *, periods: int = 180) -> pd.DataFrame:
    minutes = _MINUTES[timeframe]
    opens = pd.date_range("2024-01-02T00:00:00Z", periods=periods, freq=f"{minutes}min")
    row = np.arange(periods, dtype=np.float64)
    bid_open = pd.Series(2000.0 + row * 0.05 + np.sin(row / 9.0) * 0.2)
    bid_close = bid_open + np.sin(row / 4.0) * 0.08
    return pd.DataFrame(
        {
            "timestamp_open_utc": opens,
            "timestamp_close_utc": opens + pd.Timedelta(minutes=minutes),
            "bid_open": bid_open,
            "bid_high": np.maximum(bid_open, bid_close) + 0.1,
            "bid_low": np.minimum(bid_open, bid_close) - 0.1,
            "bid_close": bid_close,
            "instrument": "XAU_USD",
            "timeframe": timeframe,
            "is_complete": True,
            "source": "fixture",
            "raw_file_hash": "a" * 64,
            "ingested_at_utc": pd.Timestamp("2026-01-01T00:00:00Z"),
            "dataset_version": "sha256:" + "b" * 64,
        }
    )


def _predictions(
    candles: pd.DataFrame,
    start: int = 30,
    stop: int = 120,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument": "XAU_USD",
            "source": "fixture",
            "prediction_time_utc": candles.loc[
                start:stop, "timestamp_close_utc"
            ].to_numpy(),
        }
    )


def test_sequences_have_fixed_shape_and_oldest_to_newest_causal_order() -> None:
    candles = _candles("3min")
    predictions = _predictions(candles)
    result = build_timeframe_sequences(
        candles,
        predictions,
        timeframe="3min",
        sequence_length=10,
    )

    assert result.values.shape == (
        len(predictions),
        10,
        len(SEQUENCE_FEATURE_NAMES),
    )
    assert result.available.all()
    first_prediction = predictions.iloc[0]["prediction_time_utc"]
    last_source = candles.index[
        candles["timestamp_close_utc"].eq(first_prediction)
    ].item()
    expected_first_return = 10_000.0 * np.log(
        candles.loc[last_source - 9, "bid_close"]
        / candles.loc[last_source - 10, "bid_close"]
    )
    assert result.values[0, 0, 0] == pytest.approx(
        expected_first_return,
        rel=1e-5,
    )
    assert result.source_close_utc.le(
        predictions["prediction_time_utc"]
    ).all()


def test_future_price_mutation_does_not_change_older_sequences() -> None:
    candles = _candles("3min")
    predictions = _predictions(candles)
    original = build_timeframe_sequences(
        candles,
        predictions,
        timeframe="3min",
        sequence_length=10,
    )
    cutoff = predictions.iloc[
        len(predictions) // 2
    ]["prediction_time_utc"]
    changed = candles.copy(deep=True)
    future = changed["timestamp_close_utc"].gt(cutoff)
    changed.loc[
        future,
        ["bid_open", "bid_high", "bid_low", "bid_close"],
    ] *= 3.0
    rebuilt = build_timeframe_sequences(
        changed,
        predictions,
        timeframe="3min",
        sequence_length=10,
    )
    mask = predictions["prediction_time_utc"].le(cutoff).to_numpy()
    np.testing.assert_array_equal(
        rebuilt.available[mask],
        original.available[mask],
    )
    np.testing.assert_allclose(
        rebuilt.values[mask],
        original.values[mask],
        equal_nan=True,
    )


def test_gap_marks_sequence_unavailable_without_removing_prediction_rows() -> None:
    candles = _candles("15min")
    predictions = _predictions(candles, start=20, stop=80)
    baseline = build_timeframe_sequences(
        candles,
        predictions,
        timeframe="15min",
        sequence_length=6,
    )
    changed = candles.drop(index=45).reset_index(drop=True)
    sparse = build_timeframe_sequences(
        changed,
        predictions,
        timeframe="15min",
        sequence_length=6,
    )
    assert len(sparse.available) == len(baseline.available)
    assert sparse.available.sum() < baseline.available.sum()
    assert np.isnan(sparse.values[~sparse.available]).all()
