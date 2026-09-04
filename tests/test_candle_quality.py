from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gold_forecasting.resampling import ResamplingError, resample_candles
from gold_forecasting.validation import CandleValidationError, detect_gaps, validate_candles


def _candles(
    periods: int = 15,
    *,
    start: str = "2024-01-02T00:00:00Z",
) -> pd.DataFrame:
    opens = pd.date_range(start, periods=periods, freq="1min", tz="UTC")
    base = pd.Series(range(periods), dtype=float) + 2_000.0
    return pd.DataFrame(
        {
            "timestamp_open_utc": opens,
            "timestamp_close_utc": opens + pd.Timedelta(minutes=1),
            "bid_open": base,
            "bid_high": base + 1.5,
            "bid_low": base - 1.0,
            "bid_close": base + 0.5,
            "instrument": "XAU_USD",
            "timeframe": "1min",
            "is_complete": True,
            "source": "fixture",
            "raw_file_hash": "a" * 64,
            "ingested_at_utc": pd.Timestamp("2025-01-01T00:00:00Z"),
            "dataset_version": "sha256:" + "b" * 64,
        }
    )


def test_valid_candles_are_copied_and_report_no_gaps() -> None:
    source = _candles(3)

    result = validate_candles(source, expected_timeframe="1min")

    assert result.candles is not source
    assert result.duplicates_removed == 0
    assert result.gaps.gap_count == 0
    assert result.gaps.missing_candles == 0
    assert str(result.candles["timestamp_open_utc"].dtype) == "datetime64[ns, UTC]"


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("bid_high", 1_999.0, r"bid_high must be >= max"),
        ("bid_low", 2_001.0, r"bid_low must be <= min"),
        ("bid_open", 0.0, "strictly positive"),
        ("bid_close", float("nan"), "finite prices"),
    ],
)
def test_invalid_prices_and_ohlc_are_rejected(
    column: str,
    value: float,
    message: str,
) -> None:
    candles = _candles(1)
    candles.loc[0, column] = value

    with pytest.raises(CandleValidationError, match=message):
        validate_candles(candles)


def test_missing_schema_column_is_rejected() -> None:
    candles = _candles(1).drop(columns="raw_file_hash")

    with pytest.raises(CandleValidationError, match=r"missing required.*raw_file_hash"):
        validate_candles(candles)


@pytest.mark.parametrize("column", ["ingested_at_utc", "dataset_version"])
def test_required_mvp_metadata_is_enforced(column: str) -> None:
    candles = _candles(1).drop(columns=column)

    with pytest.raises(CandleValidationError, match=column):
        validate_candles(candles)


def test_naive_and_wrong_close_timestamps_are_rejected() -> None:
    naive = _candles(1)
    naive["timestamp_open_utc"] = naive["timestamp_open_utc"].dt.tz_localize(None)
    with pytest.raises(CandleValidationError, match="timezone-aware"):
        validate_candles(naive)

    wrong_close = _candles(1)
    wrong_close.loc[0, "timestamp_close_utc"] += pd.Timedelta(seconds=1)
    with pytest.raises(CandleValidationError, match="plus the timeframe"):
        validate_candles(wrong_close)


def test_non_utc_aware_timestamps_are_normalized_to_utc() -> None:
    candles = _candles(1)
    candles["timestamp_open_utc"] = candles["timestamp_open_utc"].dt.tz_convert("Europe/Brussels")
    candles["timestamp_close_utc"] = candles["timestamp_close_utc"].dt.tz_convert(
        "Europe/Brussels"
    )

    result = validate_candles(candles)

    assert str(result.candles["timestamp_open_utc"].dtype) == "datetime64[ns, UTC]"
    assert result.candles.loc[0, "timestamp_open_utc"] == pd.Timestamp("2024-01-02T00:00:00Z")


def test_incomplete_and_out_of_order_candles_are_rejected() -> None:
    incomplete = _candles(2)
    incomplete.loc[1, "is_complete"] = False
    with pytest.raises(CandleValidationError, match="incomplete candles"):
        validate_candles(incomplete)

    reversed_rows = _candles(2).iloc[::-1].reset_index(drop=True)
    with pytest.raises(CandleValidationError, match="strictly increasing"):
        validate_candles(reversed_rows)


def test_duplicate_policy_only_drops_identical_repeats() -> None:
    source = _candles(2)
    repeated = pd.concat([source, source.iloc[[0]]], ignore_index=True)

    with pytest.raises(CandleValidationError, match="duplicate candle keys"):
        validate_candles(repeated)

    result = validate_candles(repeated, duplicate_policy="drop_identical")
    assert len(result.candles) == 2
    assert result.duplicates_removed == 1

    conflicting = repeated.copy()
    conflicting.loc[2, "bid_close"] += 0.25
    with pytest.raises(CandleValidationError, match="conflicting duplicate"):
        validate_candles(conflicting, duplicate_policy="drop_identical")


def test_gap_report_is_exact_and_does_not_interpolate() -> None:
    source = _candles(5).drop(index=[1, 2]).reset_index(drop=True)

    report = detect_gaps(source)

    assert len(source) == 3
    assert report.gap_count == 1
    assert report.missing_candles == 2
    assert report.gaps[0].expected_next_open_utc == pd.Timestamp("2024-01-02T00:01:00Z")
    assert report.gaps[0].actual_next_open_utc == pd.Timestamp("2024-01-02T00:03:00Z")
    assert report.to_frame().loc[0, "gap_duration_seconds"] == 120


def test_three_minute_resampling_uses_canonical_ohlc() -> None:
    source = _candles(3)

    result = resample_candles(source, "3min")

    assert len(result) == 1
    candle = result.iloc[0]
    assert candle["timestamp_open_utc"] == pd.Timestamp("2024-01-02T00:00:00Z")
    assert candle["timestamp_close_utc"] == pd.Timestamp("2024-01-02T00:03:00Z")
    assert candle["bid_open"] == source.loc[0, "bid_open"]
    assert candle["bid_high"] == source["bid_high"].max()
    assert candle["bid_low"] == source["bid_low"].min()
    assert candle["bid_close"] == source.loc[2, "bid_close"]
    assert candle["timeframe"] == "3min"
    assert bool(candle["is_complete"])
    assert candle["ingested_at_utc"] == source["ingested_at_utc"].max()
    assert candle["dataset_version"] == source.loc[0, "dataset_version"]


def test_five_three_minute_windows_cover_one_fifteen_minute_window() -> None:
    source = _candles(15)

    three_minute = resample_candles(source, "3min")
    fifteen_minute = resample_candles(source, "15min")

    assert len(three_minute) == 5
    assert len(fifteen_minute) == 1
    assert three_minute["timestamp_open_utc"].tolist() == list(
        pd.date_range("2024-01-02T00:00:00Z", periods=5, freq="3min")
    )
    assert fifteen_minute.loc[0, "bid_open"] == three_minute.loc[0, "bid_open"]
    assert fifteen_minute.loc[0, "bid_high"] == three_minute["bid_high"].max()
    assert fifteen_minute.loc[0, "bid_low"] == three_minute["bid_low"].min()
    assert fifteen_minute.loc[0, "bid_close"] == three_minute.loc[4, "bid_close"]


def test_partial_gapped_and_incomplete_windows_are_dropped_not_filled() -> None:
    partial = _candles(2)
    assert resample_candles(partial, "3min").empty

    gapped = _candles(3).drop(index=1)
    assert resample_candles(gapped, "3min").empty

    incomplete = _candles(6)
    incomplete.loc[1, "is_complete"] = False
    result = resample_candles(incomplete, "3min")
    assert result["timestamp_open_utc"].tolist() == [pd.Timestamp("2024-01-02T00:03:00Z")]


def test_resampling_drops_unaligned_boundary_fragments() -> None:
    source = _candles(5, start="2024-01-02T00:01:00Z")

    result = resample_candles(source, "3min")

    assert result["timestamp_open_utc"].tolist() == [pd.Timestamp("2024-01-02T00:03:00Z")]


def test_resampling_is_deterministic_for_shuffled_input() -> None:
    source = _candles(15)
    shuffled = source.sample(frac=1, random_state=17).reset_index(drop=True)

    expected = resample_candles(source, "15min")
    actual = resample_candles(shuffled, "15min")

    assert_frame_equal(actual, expected)


def test_resampling_rejects_non_1min_input_and_unknown_target() -> None:
    source = _candles(3)
    source["timeframe"] = "3min"
    source["timestamp_close_utc"] = source["timestamp_open_utc"] + pd.Timedelta(minutes=3)

    with pytest.raises(ResamplingError, match="must all use timeframe '1min'"):
        resample_candles(source, "15min")
    with pytest.raises(ResamplingError, match="target_timeframe"):
        resample_candles(_candles(3), "5min")
