from __future__ import annotations

import pandas as pd
import pytest

from gold_forecasting.validation import CandleValidationError, detect_gaps, validate_candles


def _derived_candles(
    opens: list[str],
    *,
    timeframe: str,
    closes: list[str],
) -> pd.DataFrame:
    count = len(opens)
    base = pd.Series(range(count), dtype=float) + 2_000.0
    return pd.DataFrame(
        {
            "timestamp_open_utc": pd.to_datetime(opens, utc=True),
            "timestamp_close_utc": pd.to_datetime(closes, utc=True),
            "bid_open": base,
            "bid_high": base + 1.0,
            "bid_low": base - 1.0,
            "bid_close": base + 0.25,
            "instrument": "XAU_USD",
            "timeframe": timeframe,
            "is_complete": True,
            "source": "fixture",
            "raw_file_hash": "a" * 64,
            "ingested_at_utc": pd.Timestamp("2025-01-01T00:00:00Z"),
            "dataset_version": "sha256:" + "b" * 64,
        }
    )


@pytest.mark.parametrize(
    ("timeframe", "minutes"),
    [("5min", 5), ("30min", 30), ("1h", 60), ("3h", 180), ("1d", 1_440)],
)
def test_phase5_fixed_timeframes_use_the_utc_grid(
    timeframe: str,
    minutes: int,
) -> None:
    open_time = pd.Timestamp("2024-01-02T00:00:00Z")
    close_time = open_time + pd.Timedelta(minutes=minutes)
    candles = _derived_candles(
        [open_time.isoformat()],
        timeframe=timeframe,
        closes=[close_time.isoformat()],
    )

    result = validate_candles(candles, expected_timeframe=timeframe)

    assert len(result.candles) == 1


def test_calendar_month_close_and_gap_count_are_calendar_aware() -> None:
    candles = _derived_candles(
        ["2024-01-01T00:00:00Z", "2024-03-01T00:00:00Z"],
        timeframe="1mo",
        closes=["2024-02-01T00:00:00Z", "2024-04-01T00:00:00Z"],
    )

    result = validate_candles(candles, expected_timeframe="1mo")
    gaps = detect_gaps(result.candles)

    assert gaps.gap_count == 1
    assert gaps.missing_candles == 1
    assert gaps.gaps[0].expected_next_open_utc == pd.Timestamp("2024-02-01T00:00:00Z")


def test_calendar_month_rejects_non_month_boundary_and_fixed_duration() -> None:
    misaligned = _derived_candles(
        ["2024-01-02T00:00:00Z"],
        timeframe="1mo",
        closes=["2024-02-01T00:00:00Z"],
    )
    with pytest.raises(CandleValidationError, match="UTC calendar"):
        validate_candles(misaligned, expected_timeframe="1mo")

    thirty_days = _derived_candles(
        ["2024-02-01T00:00:00Z"],
        timeframe="1mo",
        closes=["2024-03-02T00:00:00Z"],
    )
    with pytest.raises(CandleValidationError, match="plus the timeframe"):
        validate_candles(thirty_days, expected_timeframe="1mo")


def test_calendar_month_rejects_sub_microsecond_boundary_shift() -> None:
    shifted = _derived_candles(
        ["2024-01-01T00:00:00.000000001Z"],
        timeframe="1mo",
        closes=["2024-02-01T00:00:00.000000001Z"],
    )
    with pytest.raises(CandleValidationError, match="UTC calendar"):
        validate_candles(shifted, expected_timeframe="1mo")
