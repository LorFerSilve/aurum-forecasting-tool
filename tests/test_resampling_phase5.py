from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gold_forecasting.resampling import (
    RESAMPLING_LOGIC_VERSION,
    SUPPORTED_TARGET_TIMEFRAMES,
    resample_candles,
    resample_mvp_timeframes,
    resample_timeframes,
)
from gold_forecasting.validation import validate_candles
from gold_forecasting.validation.quality import (
    ExplicitMarketCalendarPolicy,
    ObservedSourceCalendarPolicy,
    UtcMarketWindow,
    WeeklyOpenWindow,
)


def _candles(
    periods: int,
    *,
    start: str = "2024-01-01T00:00:00Z",
) -> pd.DataFrame:
    opens = pd.date_range(start, periods=periods, freq="1min", tz="UTC")
    base = pd.Series(range(periods), dtype=float) + 2_000.0
    return pd.DataFrame(
        {
            "timestamp_open_utc": opens,
            "timestamp_close_utc": opens + pd.Timedelta(minutes=1),
            "bid_open": base,
            "bid_high": base + 2.0,
            "bid_low": base - 1.0,
            "bid_close": base + 0.25,
            "instrument": "XAU_USD",
            "timeframe": "1min",
            "is_complete": True,
            "source": "phase5-fixture",
            "raw_file_hash": "a" * 64,
            "ingested_at_utc": pd.Timestamp("2025-01-01T00:00:00Z"),
            "dataset_version": "sha256:" + "b" * 64,
        }
    )


@pytest.mark.parametrize(
    ("timeframe", "minutes"),
    [
        ("3min", 3),
        ("5min", 5),
        ("15min", 15),
        ("30min", 30),
        ("1h", 60),
        ("3h", 180),
        ("1d", 1_440),
    ],
)
def test_fixed_timeframe_golden_aggregation(timeframe: str, minutes: int) -> None:
    source = _candles(minutes)

    result = resample_candles(source, timeframe)

    assert len(result) == 1
    candle = result.iloc[0]
    assert candle["timestamp_open_utc"] == pd.Timestamp("2024-01-01T00:00:00Z")
    assert candle["timestamp_close_utc"] == pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(
        minutes=minutes
    )
    assert candle["bid_open"] == source.iloc[0]["bid_open"]
    assert candle["bid_high"] == source["bid_high"].max()
    assert candle["bid_low"] == source["bid_low"].min()
    assert candle["bid_close"] == source.iloc[-1]["bid_close"]
    assert candle["timeframe"] == timeframe
    assert bool(candle["is_complete"])


@pytest.mark.parametrize(
    ("timeframe", "minutes", "first_complete_open"),
    [
        ("5min", 5, "2024-01-02T00:00:00Z"),
        ("30min", 30, "2024-01-02T00:00:00Z"),
        ("1h", 60, "2024-01-02T00:00:00Z"),
        ("3h", 180, "2024-01-02T00:00:00Z"),
    ],
)
def test_fixed_windows_follow_midnight_anchored_utc_grid(
    timeframe: str,
    minutes: int,
    first_complete_open: str,
) -> None:
    source = _candles(minutes + 2, start="2024-01-01T23:59:00Z")

    result = resample_candles(source, timeframe)

    assert result["timestamp_open_utc"].tolist() == [pd.Timestamp(first_complete_open)]


@pytest.mark.parametrize("timeframe", ["5min", "30min", "1h", "3h"])
def test_gap_or_incomplete_minute_drops_only_affected_window(timeframe: str) -> None:
    minutes = {"5min": 5, "30min": 30, "1h": 60, "3h": 180}[timeframe]
    source = _candles(minutes * 2)
    source = source.drop(index=minutes // 2).reset_index(drop=True)
    source.loc[minutes, "is_complete"] = False

    result = resample_candles(source, timeframe)

    assert result.empty


def test_calendar_month_uses_real_leap_february_boundaries() -> None:
    source = _candles(29 * 1_440, start="2024-02-01T00:00:00Z")

    result = resample_candles(source, "1mo")

    assert len(result) == 1
    candle = result.iloc[0]
    assert candle["timestamp_open_utc"] == pd.Timestamp("2024-02-01T00:00:00Z")
    assert candle["timestamp_close_utc"] == pd.Timestamp("2024-03-01T00:00:00Z")
    assert candle["bid_open"] == source.iloc[0]["bid_open"]
    assert candle["bid_close"] == source.iloc[-1]["bid_close"]
    assert candle["timeframe"] == "1mo"


def test_calendar_month_handles_year_transition() -> None:
    source = _candles(31 * 1_440, start="2024-12-01T00:00:00Z")

    result = resample_candles(source, "1mo")

    assert result["timestamp_open_utc"].tolist() == [pd.Timestamp("2024-12-01T00:00:00Z")]
    assert result["timestamp_close_utc"].tolist() == [pd.Timestamp("2025-01-01T00:00:00Z")]


def test_calendar_month_is_never_treated_as_thirty_days() -> None:
    incomplete_january = _candles(30 * 1_440, start="2024-01-01T00:00:00Z")
    assert resample_candles(incomplete_january, "1mo").empty
    full_january = _candles(31 * 1_440, start="2024-01-01T00:00:00Z")
    result = resample_candles(full_january, "1mo")
    assert result["timestamp_close_utc"].tolist() == [pd.Timestamp("2024-02-01T00:00:00Z")]


def test_only_closed_calendar_month_is_emitted() -> None:
    # February is closed at the knowledge cutoff; March is still in progress.
    source = _candles((29 + 14) * 1_440, start="2024-02-01T00:00:00Z")
    source["ingested_at_utc"] = pd.Timestamp("2024-03-15T00:00:00Z")

    result = resample_candles(source, "1mo")

    assert result["timestamp_open_utc"].tolist() == [pd.Timestamp("2024-02-01T00:00:00Z")]
    assert result["timestamp_close_utc"].tolist() == [pd.Timestamp("2024-03-01T00:00:00Z")]


@pytest.mark.parametrize(("timeframe", "periods"), [("1d", 1_440), ("1mo", 29 * 1_440)])
def test_sparse_windows_require_explicit_scheduled_closures(
    timeframe: str, periods: int
) -> None:
    source = _candles(periods, start="2024-02-01T00:00:00Z").drop(
        index=[10, 11, 12]
    ).reset_index(drop=True)
    calendar = ExplicitMarketCalendarPolicy(
        weekly_open_windows=tuple(WeeklyOpenWindow(day, 0, 1_440) for day in range(7)),
        weekend_weekdays=frozenset(),
        maintenance_windows=(
            UtcMarketWindow(
                pd.Timestamp("2024-02-01T00:10:00Z"),
                pd.Timestamp("2024-02-01T00:13:00Z"),
                "fixture scheduled maintenance",
            ),
        ),
    )

    assert resample_candles(source, timeframe).empty
    assert resample_candles(
        source, timeframe, calendar_policy=ObservedSourceCalendarPolicy()
    ).empty
    result = resample_candles(source, timeframe, calendar_policy=calendar)
    assert len(result) == 1
    assert result.iloc[0]["bid_close"] == source.iloc[-1]["bid_close"]
    assert resample_candles(
        source, timeframe, calendar_policy=calendar, require_dense=True
    ).empty

    # A real missing market minute remains disqualifying despite another gap
    # having an explicit closure reason.
    source = source.drop(index=20).reset_index(drop=True)
    assert resample_candles(source, timeframe, calendar_policy=calendar).empty


@pytest.mark.parametrize(("timeframe", "periods"), [("1d", 1_440), ("1mo", 29 * 1_440)])
def test_boundary_fragments_never_become_complete_historical_windows(
    timeframe: str, periods: int
) -> None:
    source = _candles(periods, start="2024-02-01T00:00:00Z")
    for truncated in (source.iloc[1:], source.iloc[:-1], source.iloc[:2]):
        assert resample_candles(truncated, timeframe).empty
        assert resample_candles(
            truncated, timeframe, calendar_policy=ObservedSourceCalendarPolicy()
        ).empty


def test_absent_entire_day_prevents_false_complete_month() -> None:
    source = _candles(29 * 1_440, start="2024-02-01T00:00:00Z")
    source = source.loc[source["timestamp_open_utc"].dt.day.ne(10)].reset_index(drop=True)
    assert resample_candles(source, "1mo").empty


def test_incomplete_minute_drops_its_sparse_daily_window() -> None:
    source = _candles(2 * 1_440)
    source.loc[10, "is_complete"] = False

    result = resample_candles(source, "1d")

    assert result["timestamp_open_utc"].tolist() == [pd.Timestamp("2024-01-02T00:00:00Z")]


def test_explicit_knowledge_cutoff_prevents_open_daily_window() -> None:
    source = _candles(1_440)

    result = resample_candles(
        source,
        "1d",
        closed_through_utc="2024-01-01T12:00:00Z",
    )

    assert result.empty


def test_resampling_rejects_naive_cutoff_and_non_boolean_density() -> None:
    source = _candles(5)

    with pytest.raises(ValueError, match="timezone-aware"):
        resample_candles(source, "5min", closed_through_utc="2024-01-02")
    with pytest.raises(ValueError, match="require_dense"):
        resample_candles(source, "5min", require_dense="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="explicit calendar_policy"):
        resample_candles(source, "5min", require_dense=False)


@pytest.mark.parametrize(
    ("timeframe", "periods"),
    [
        ("5min", 35),
        ("30min", 180),
        ("1h", 240),
        ("3h", 720),
        ("1d", 2_880),
        ("1mo", 29 * 1_440),
    ],
)
def test_phase5_resampling_is_deterministic_for_shuffled_input(
    timeframe: str,
    periods: int,
) -> None:
    start = "2024-02-01T00:00:00Z" if timeframe == "1mo" else "2024-01-01T00:00:00Z"
    source = _candles(periods, start=start)
    shuffled = source.sample(frac=1, random_state=20240904).reset_index(drop=True)

    expected = resample_candles(source, timeframe)
    actual = resample_candles(shuffled, timeframe)

    assert_frame_equal(actual, expected)


def test_public_contract_adds_phase5_targets_without_changing_mvp_helper() -> None:
    assert RESAMPLING_LOGIC_VERSION == "2.0.0"
    assert SUPPORTED_TARGET_TIMEFRAMES == (
        "3min",
        "5min",
        "15min",
        "30min",
        "1h",
        "3h",
        "1d",
        "1mo",
    )

    result = resample_mvp_timeframes(_candles(15))
    assert tuple(result) == ("3min", "15min")


def test_multi_timeframe_builder_matches_individual_outputs_and_validates_empty_schema() -> None:
    source = _candles(180)
    combined = resample_timeframes(source)
    assert tuple(combined) == SUPPORTED_TARGET_TIMEFRAMES
    for timeframe, actual in combined.items():
        assert_frame_equal(actual, resample_candles(source, timeframe))
        validate_candles(actual, expected_timeframe=timeframe)
    empty = resample_timeframes(source.iloc[:0])
    for timeframe, actual in empty.items():
        assert actual.empty
        validate_candles(actual, expected_timeframe=timeframe)


def test_multi_timeframe_builder_rejects_duplicate_codes() -> None:
    with pytest.raises(ValueError, match="unique timeframe codes"):
        resample_timeframes(_candles(15), ("3min", "3min"))
