from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold_forecasting.phase9.targets import (
    PATH_COMPONENTS,
    FuturePathError,
    aggregate_path_ohlc,
    build_future_path_targets,
    reconstruct_path,
)


def _candles(*, periods: int = 80) -> pd.DataFrame:
    opens = pd.date_range(
        "2024-01-02T00:00:00Z",
        periods=periods,
        freq="3min",
    )
    row = np.arange(periods, dtype=np.float64)
    bid_open = pd.Series(2000.0 + row * 0.25)
    bid_close = bid_open + np.sin(row / 3.0) * 0.18
    return pd.DataFrame(
        {
            "timestamp_open_utc": opens,
            "timestamp_close_utc": opens + pd.Timedelta(minutes=3),
            "bid_open": bid_open,
            "bid_high": np.maximum(bid_open, bid_close) + 0.12,
            "bid_low": np.minimum(bid_open, bid_close) - 0.10,
            "bid_close": bid_close,
            "instrument": "XAU_USD",
            "timeframe": "3min",
            "is_complete": True,
            "source": "fixture",
            "raw_file_hash": "a" * 64,
            "ingested_at_utc": pd.Timestamp("2026-01-01T00:00:00Z"),
            "dataset_version": "sha256:" + "b" * 64,
        }
    )


def _predictions(candles: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument": "XAU_USD",
            "source": "fixture",
            "prediction_time_utc": candles.loc[
                indices,
                "timestamp_close_utc",
            ].to_numpy(),
        }
    )


def _representation(labels: pd.DataFrame) -> np.ndarray:
    result = np.empty((len(labels), 5, len(PATH_COMPONENTS)), dtype=np.float64)
    for step in range(5):
        for component_index, component in enumerate(PATH_COMPONENTS):
            result[:, step, component_index] = labels[
                f"path_step_{step + 1}_{component}"
            ].to_numpy(dtype=np.float64)
    return result


def _ohlc(labels: pd.DataFrame) -> np.ndarray:
    result = np.empty((len(labels), 5, 4), dtype=np.float64)
    for step in range(5):
        for component_index, component in enumerate(("open", "high", "low", "close")):
            result[:, step, component_index] = labels[
                f"path_step_{step + 1}_bid_{component}"
            ].to_numpy(dtype=np.float64)
    return result


def test_future_path_uses_exact_next_five_closed_three_minute_candles() -> None:
    candles = _candles()
    predictions = _predictions(candles, [10, 20, 30])

    result = build_future_path_targets(candles, predictions)

    assert result.output_row_count == 3
    assert result.dropped_missing_path == 0
    labels = result.labels
    assert isinstance(labels["path_end_time_utc"].dtype, pd.DatetimeTZDtype)
    assert str(labels["path_end_time_utc"].dt.tz) == "UTC"
    for step in range(1, 6):
        assert isinstance(
            labels[f"path_step_{step}_open_time_utc"].dtype,
            pd.DatetimeTZDtype,
        )
        assert isinstance(
            labels[f"path_step_{step}_close_time_utc"].dtype,
            pd.DatetimeTZDtype,
        )
        assert str(labels[f"path_step_{step}_open_time_utc"].dt.tz) == "UTC"
        assert str(labels[f"path_step_{step}_close_time_utc"].dt.tz) == "UTC"
    first = labels.iloc[0]
    assert first["path_step_1_open_time_utc"] == candles.loc[
        11, "timestamp_open_utc"
    ]
    assert first["path_step_5_close_time_utc"] == candles.loc[
        15, "timestamp_close_utc"
    ]
    assert first["path_end_time_utc"] == first["prediction_time_utc"] + pd.Timedelta(
        minutes=15
    )


def test_path_representation_reconstructs_true_ohlc_and_aggregate() -> None:
    candles = _candles()
    result = build_future_path_targets(
        candles,
        _predictions(candles, [10, 20, 30]),
    )
    labels = result.labels
    representation = _representation(labels)
    actual = _ohlc(labels)

    rebuilt = reconstruct_path(
        labels["path_anchor_close"].to_numpy(dtype=np.float64),
        representation,
    )

    np.testing.assert_allclose(rebuilt, actual, rtol=0.0, atol=1e-9)
    aggregate = aggregate_path_ohlc(rebuilt)
    np.testing.assert_allclose(
        aggregate[:, 0],
        labels["path_15m_bid_open"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        aggregate[:, 3],
        labels["path_15m_bid_close"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-9,
    )


def test_missing_future_candle_drops_affected_path_instead_of_bridging_gap() -> None:
    candles = _candles()
    predictions = _predictions(candles, [10, 20])
    baseline = build_future_path_targets(candles, predictions)

    changed = candles.drop(index=13).reset_index(drop=True)
    sparse = build_future_path_targets(changed, predictions)

    assert baseline.output_row_count == 2
    assert sparse.output_row_count == 1
    assert sparse.dropped_missing_path == 1


def test_reconstruction_guarantees_positive_ohlc_for_extreme_predictions() -> None:
    anchor = np.array([2000.0, 2100.0])
    representation = np.array(
        [
            [[-20_000.0, 20_000.0, -5.0, -10.0]] * 5,
            [[20_000.0, -20_000.0, 20_000.0, 20_000.0]] * 5,
        ],
        dtype=np.float64,
    )

    path = reconstruct_path(anchor, representation)

    assert np.isfinite(path).all()
    assert (path > 0.0).all()
    assert (path[:, :, 1] >= np.maximum(path[:, :, 0], path[:, :, 3])).all()
    assert (path[:, :, 2] <= np.minimum(path[:, :, 0], path[:, :, 3])).all()


def test_reserved_holdout_candle_input_is_rejected() -> None:
    candles = _candles()
    extra = candles.tail(1).copy()
    extra["timestamp_open_utc"] = pd.Timestamp("2025-01-01T00:00:00Z")
    extra["timestamp_close_utc"] = pd.Timestamp("2025-01-01T00:03:00Z")
    changed = pd.concat([candles, extra], ignore_index=True)

    with pytest.raises(FuturePathError, match="holdout"):
        build_future_path_targets(
            changed,
            _predictions(candles, [10]),
        )
