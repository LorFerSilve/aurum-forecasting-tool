from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from gold_forecasting.phase9.dataset import build_phase9_dataset


def _candles(
    timeframe: str,
    *,
    periods: int,
) -> pd.DataFrame:
    minutes = 1 if timeframe == "1min" else 3
    opens = pd.date_range(
        "2024-01-02T00:00:00Z",
        periods=periods,
        freq=f"{minutes}min",
    )
    row = np.arange(periods, dtype=np.float64)
    bid_open = pd.Series(2000.0 + row * 0.04)
    bid_close = bid_open + np.sin(row / 5.0) * 0.03
    return pd.DataFrame(
        {
            "timestamp_open_utc": opens,
            "timestamp_close_utc": (
                opens
                + pd.Timedelta(minutes=minutes)
            ),
            "bid_open": bid_open,
            "bid_high": np.maximum(
                bid_open,
                bid_close,
            ) + 0.05,
            "bid_low": np.minimum(
                bid_open,
                bid_close,
            ) - 0.05,
            "bid_close": bid_close,
            "instrument": "XAU_USD",
            "timeframe": timeframe,
            "is_complete": True,
            "source": "fixture",
            "raw_file_hash": "a" * 64,
            "ingested_at_utc": pd.Timestamp(
                "2026-01-01T00:00:00Z"
            ),
            "dataset_version": (
                "sha256:" + "b" * 64
            ),
        }
    )


def _predictions() -> pd.DataFrame:
    times = pd.date_range(
        "2024-01-02T01:00:00Z",
        periods=8,
        freq="15min",
    )
    return pd.DataFrame(
        {
            "instrument": "XAU_USD",
            "source": "fixture",
            "prediction_time_utc": times,
        }
    )


def test_common_phase9_dataset_preserves_both_timing_contracts() -> None:
    predictions = _predictions()
    result = build_phase9_dataset(
        _candles("1min", periods=300),
        _candles("3min", periods=100),
        predictions,
    )

    table = result.table
    assert len(table) == len(predictions)
    assert table["path_end_time_utc"].eq(
        table["prediction_time_utc"]
        + pd.Timedelta(minutes=15)
    ).all()
    assert table["entry_time_utc"].eq(
        table["prediction_time_utc"]
        + pd.Timedelta(minutes=1)
    ).all()
    assert table["label_end_time_utc"].eq(
        table["prediction_time_utc"]
        + pd.Timedelta(minutes=16)
    ).all()
    first = table.iloc[0]
    expected = hashlib.sha256(
        (
            f"{first['instrument']}|{first['source']}|"
            f"{first['prediction_time_utc'].isoformat()}|15"
        ).encode()
    ).hexdigest()
    assert first["sample_id"] == expected
    assert str(result.diagnostics["sample_digest"]).startswith(
        "sha256:"
    )


def test_three_minute_gap_reduces_common_universe_without_bridging() -> None:
    predictions = _predictions()
    one_minute = _candles(
        "1min",
        periods=300,
    )
    three_minute = _candles(
        "3min",
        periods=100,
    )
    baseline = build_phase9_dataset(
        one_minute,
        three_minute,
        predictions,
    )
    changed = three_minute.drop(
        index=23
    ).reset_index(drop=True)
    sparse = build_phase9_dataset(
        one_minute,
        changed,
        predictions,
    )

    assert len(sparse.table) < len(baseline.table)
    assert (
        sparse.diagnostics["path_eligible"]
        < baseline.diagnostics["path_eligible"]
    )
