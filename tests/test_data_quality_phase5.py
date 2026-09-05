from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gold_forecasting.validation import (
    DataQualityError,
    ExplicitMarketCalendarPolicy,
    KnownGap,
    ObservedSourceCalendarPolicy,
    UtcMarketWindow,
    WeeklyOpenWindow,
    build_daily_data_quality,
)


def _candles(opens: list[str], *, source: str = "histdata") -> pd.DataFrame:
    timestamps = pd.DatetimeIndex(pd.to_datetime(opens, utc=True))
    base = pd.Series(range(len(opens)), dtype=float) + 2_000.0
    return pd.DataFrame(
        {
            "timestamp_open_utc": timestamps,
            "timestamp_close_utc": timestamps + pd.Timedelta(minutes=1),
            "bid_open": base,
            "bid_high": base + 1.0,
            "bid_low": base - 1.0,
            "bid_close": base + 0.25,
            "instrument": "XAU_USD",
            "timeframe": "1min",
            "is_complete": True,
            "source": source,
            "raw_file_hash": "a" * 64,
            "ingested_at_utc": pd.Timestamp("2025-01-01T00:00:00Z"),
            "dataset_version": "sha256:" + "b" * 64,
        }
    )


def _weekday_calendar(
    *,
    holidays: tuple[UtcMarketWindow, ...] = (),
    maintenance: tuple[UtcMarketWindow, ...] = (),
) -> ExplicitMarketCalendarPolicy:
    return ExplicitMarketCalendarPolicy(
        weekly_open_windows=tuple(
            WeeklyOpenWindow(weekday=weekday, start_minute_utc=0, end_minute_utc=1_440)
            for weekday in range(5)
        ),
        holidays=holidays,
        maintenance_windows=maintenance,
        version="fixture-calendar-v3",
    )


def test_observed_source_policy_marks_unknown_gap_and_counts_row_defects() -> None:
    candles = _candles(
        [
            "2024-01-02T00:00:00Z",
            "2024-01-02T00:02:00Z",
            "2024-01-02T00:03:00Z",
            "2024-01-02T00:03:00Z",
        ]
    )
    candles.loc[1, "is_complete"] = False
    candles.loc[2, "bid_high"] = candles.loc[2, "bid_open"] - 1.0

    report = build_daily_data_quality(
        candles,
        calendar=ObservedSourceCalendarPolicy(),
        as_of_utc=pd.Timestamp("2024-01-03T00:00:00Z"),
        stale_after=pd.Timedelta(minutes=5),
    )

    assert len(report.records) == 1
    record = report.records[0]
    assert record.expectation_basis == "observed_source"
    assert record.market_status == "unknown"
    assert record.row_count == 4
    assert record.observed_count == 3
    assert record.expected_count == 4
    assert record.missing_count == 1
    assert record.stale_count == 1
    assert record.incomplete_count == 1
    assert record.duplicate_count == 1
    assert record.ohlc_invalid_count == 1
    assert record.unknown_gap_count == 1
    assert record.unexplained_gap_count == 0
    assert record.quality_status == "fail"
    assert report.gaps[0].classification == "unknown_market_status"
    assert report.gaps[0].market_status == "unknown"
    assert report.gaps[0].counts_as_missing
    assert report.gaps[0].is_stale
    assert not report.gaps[0].is_explained


def test_explicit_calendar_separates_weekend_closure_from_missing_market_data() -> None:
    candles = _candles(
        [
            "2024-01-05T23:59:00Z",  # Friday
            "2024-01-08T00:01:00Z",  # Monday
        ]
    )

    report = build_daily_data_quality(
        candles,
        calendar=_weekday_calendar(),
        as_of_utc=pd.Timestamp("2024-01-09T00:00:00Z"),
        stale_after=pd.Timedelta(minutes=5),
    )

    by_day = {record.day_utc: record for record in report.records}
    assert by_day[date(2024, 1, 6)].market_status == "weekend"
    assert by_day[date(2024, 1, 6)].expected_count == 0
    assert by_day[date(2024, 1, 7)].expected_count == 0
    assert by_day[date(2024, 1, 8)].missing_count == 1
    assert by_day[date(2024, 1, 8)].unexplained_gap_count == 1

    weekend_gaps = [gap for gap in report.gaps if gap.market_status == "weekend"]
    assert len(weekend_gaps) == 2
    assert all(gap.classification == "scheduled_closure" for gap in weekend_gaps)
    assert all(gap.is_explained and not gap.counts_as_missing for gap in weekend_gaps)
    assert all(not gap.is_stale for gap in weekend_gaps)
    monday_gap = next(gap for gap in report.gaps if gap.day_utc == date(2024, 1, 8))
    assert monday_gap.classification == "missing_data"
    assert not monday_gap.is_explained


def test_known_gap_annotation_explains_expected_missing_interval() -> None:
    candles = _candles(["2024-01-08T00:00:00Z", "2024-01-08T00:02:00Z"])
    annotation = KnownGap(
        start_utc=pd.Timestamp("2024-01-08T00:01:00Z"),
        end_utc=pd.Timestamp("2024-01-08T00:02:00Z"),
        code="provider-outage-42",
        reason="provider incident confirmed in immutable ingest manifest",
        source="histdata",
        instrument="XAU_USD",
        timeframe="1min",
    )

    report = build_daily_data_quality(
        candles,
        calendar=_weekday_calendar(),
        as_of_utc=pd.Timestamp("2024-01-09T00:00:00Z"),
        stale_after=pd.Timedelta(0),
        known_gaps=(annotation,),
    )

    record = report.records[0]
    assert record.missing_count == 1
    assert record.explained_gap_count == 1
    assert record.unexplained_gap_count == 0
    assert record.quality_status == "marked"
    assert report.gaps[0].classification == "known_gap"
    assert report.gaps[0].annotation_code == "provider-outage-42"
    assert report.gaps[0].counts_as_missing
    assert report.gaps[0].is_stale
    assert report.gaps[0].is_explained


def test_holiday_and_maintenance_are_only_claimed_when_explicitly_configured() -> None:
    holiday = UtcMarketWindow(
        pd.Timestamp("2024-01-09T00:00:00Z"),
        pd.Timestamp("2024-01-10T00:00:00Z"),
        "verified exchange holiday source",
    )
    maintenance = UtcMarketWindow(
        pd.Timestamp("2024-01-08T00:01:00Z"),
        pd.Timestamp("2024-01-08T00:02:00Z"),
        "published provider maintenance window",
    )
    candles = _candles(["2024-01-08T00:00:00Z", "2024-01-10T00:00:00Z"])

    report = build_daily_data_quality(
        candles,
        calendar=_weekday_calendar(holidays=(holiday,), maintenance=(maintenance,)),
        as_of_utc=pd.Timestamp("2024-01-11T00:00:00Z"),
        stale_after=pd.Timedelta(minutes=5),
    )

    holiday_record = next(record for record in report.records if record.day_utc == date(2024, 1, 9))
    assert holiday_record.market_status == "holiday"
    assert holiday_record.expected_count == 0
    statuses = {gap.market_status for gap in report.gaps}
    assert {"maintenance", "holiday"} <= statuses
    assert all(
        gap.classification == "scheduled_closure"
        for gap in report.gaps
        if gap.market_status in {"maintenance", "holiday"}
    )


def test_output_is_deterministic_for_input_order_and_mixed_sources() -> None:
    first = _candles(["2024-01-02T00:00:00Z", "2024-01-02T00:02:00Z"])
    second = _candles(
        ["2024-01-02T00:00:00Z", "2024-01-02T00:01:00Z"], source="replacement"
    )
    candles = pd.concat([first, second], ignore_index=True)
    kwargs = {
        "calendar": ObservedSourceCalendarPolicy(),
        "as_of_utc": pd.Timestamp("2024-01-03T00:00:00Z"),
        "stale_after": pd.Timedelta(minutes=5),
    }

    ordered = build_daily_data_quality(candles, **kwargs)
    shuffled = build_daily_data_quality(
        candles.sample(frac=1.0, random_state=7).reset_index(drop=True), **kwargs
    )

    assert_frame_equal(ordered.to_frame(), shuffled.to_frame())
    assert_frame_equal(ordered.gaps_to_frame(), shuffled.gaps_to_frame())


def test_explicit_scope_reports_fully_absent_days_without_inventing_expected_source_data() -> None:
    candles = _candles(["2024-01-02T12:00:00Z"])

    report = build_daily_data_quality(
        candles,
        calendar=ObservedSourceCalendarPolicy(),
        as_of_utc=pd.Timestamp("2024-01-04T00:00:00Z"),
        stale_after=pd.Timedelta(0),
        start_day_utc=date(2024, 1, 1),
        end_day_utc=date(2024, 1, 3),
    )

    assert len(report.records) == 3
    empty_days = [record for record in report.records if record.observed_count == 0]
    assert len(empty_days) == 2
    assert all(record.expected_count == 0 for record in empty_days)
    assert all(record.market_status == "unknown" for record in empty_days)
    assert all(record.missing_count == 0 for record in empty_days)
    empty_day_gaps = [gap for gap in report.gaps if gap.day_utc != date(2024, 1, 2)]
    assert all(not gap.counts_as_missing and not gap.is_stale for gap in empty_day_gaps)


def test_audit_rejects_naive_reference_time_and_ambiguous_timeframe() -> None:
    candles = _candles(["2024-01-02T00:00:00Z"])
    with pytest.raises(DataQualityError, match="as_of_utc must be timezone-aware"):
        build_daily_data_quality(
            candles,
            calendar=ObservedSourceCalendarPolicy(),
            as_of_utc=pd.Timestamp("2024-01-03"),
            stale_after=pd.Timedelta(0),
        )

    candles["timeframe"] = "1m"
    with pytest.raises(DataQualityError, match="unsupported or ambiguous timeframe"):
        build_daily_data_quality(
            candles,
            calendar=ObservedSourceCalendarPolicy(),
            as_of_utc=pd.Timestamp("2024-01-03T00:00:00Z"),
            stale_after=pd.Timedelta(0),
        )


def test_year_boundary_gap_is_visible_and_input_frame_is_unchanged() -> None:
    candles = _candles(["2023-12-31T23:58:00Z", "2024-01-01T00:01:00Z"])
    original = candles.copy(deep=True)

    report = build_daily_data_quality(
        candles,
        calendar=ObservedSourceCalendarPolicy(),
        as_of_utc=pd.Timestamp("2024-01-02T00:00:00Z"),
        stale_after=pd.Timedelta(0),
    )

    assert_frame_equal(candles, original)
    assert [record.day_utc for record in report.records] == [date(2023, 12, 31), date(2024, 1, 1)]
    assert sum(gap.missing_intervals for gap in report.gaps) == 2
    assert all(gap.classification == "unknown_market_status" for gap in report.gaps)
    assert report.gaps[0].start_utc == pd.Timestamp("2023-12-31T23:59:00Z")
    assert report.gaps[1].end_utc == pd.Timestamp("2024-01-01T00:01:00Z")


def test_cutoff_does_not_report_still_open_or_future_windows_as_missing() -> None:
    candles = _candles(["2024-01-08T00:00:00Z"])

    report = build_daily_data_quality(
        candles,
        calendar=_weekday_calendar(),
        as_of_utc=pd.Timestamp("2024-01-08T00:02:30Z"),
        stale_after=pd.Timedelta(0),
        end_day_utc=date(2024, 1, 8),
    )

    record = report.records[0]
    assert record.expected_count == 2
    assert record.missing_count == 1
    assert report.gaps[0].end_utc == pd.Timestamp("2024-01-08T00:02:00Z")


def test_incomplete_flag_includes_canonical_windows_not_closed_at_cutoff() -> None:
    candles = _candles(["2024-02-01T00:00:00Z"])
    candles["timeframe"] = "1mo"
    candles["timestamp_close_utc"] = pd.Timestamp("2024-03-01T00:00:00Z")

    report = build_daily_data_quality(
        candles,
        calendar=ObservedSourceCalendarPolicy(),
        as_of_utc=pd.Timestamp("2024-02-29T23:59:59Z"),
        stale_after=pd.Timedelta(0),
    )

    assert report.records[0].incomplete_count == 1
    assert report.records[0].quality_status == "fail"
    assert report.records[0].expected_count == 0
    assert report.records[0].missing_count == 0


def test_missing_month_uses_full_leap_february_window() -> None:
    candles = _candles(["2024-01-01T00:00:00Z", "2024-03-01T00:00:00Z"])
    candles["timeframe"] = "1mo"
    candles["timestamp_close_utc"] = candles["timestamp_open_utc"] + pd.offsets.MonthBegin(1)

    report = build_daily_data_quality(
        candles,
        calendar=ObservedSourceCalendarPolicy(),
        as_of_utc=pd.Timestamp("2024-04-01T00:00:00Z"),
        stale_after=pd.Timedelta(0),
    )

    assert len(report.gaps) == 1
    gap = report.gaps[0]
    assert gap.start_utc == pd.Timestamp("2024-02-01T00:00:00Z")
    assert gap.end_utc == pd.Timestamp("2024-03-01T00:00:00Z")
    assert gap.end_utc - gap.start_utc == pd.Timedelta(days=29)
    assert gap.missing_intervals == 1
    assert gap.classification == "unknown_market_status"


@pytest.mark.parametrize("timeframe", ["1min", "1h", "1d", "1mo"])
def test_misaligned_opens_are_rejected_before_counting_missing_windows(timeframe: str) -> None:
    candles = _candles(["2024-01-02T00:00:01Z"])
    candles["timeframe"] = timeframe

    with pytest.raises(DataQualityError, match="canonical UTC grid"):
        build_daily_data_quality(
            candles,
            calendar=ObservedSourceCalendarPolicy(),
            as_of_utc=pd.Timestamp("2024-02-01T00:00:00Z"),
            stale_after=pd.Timedelta(0),
        )


def test_month_starting_on_weekend_is_expected_if_it_contains_trading_sessions() -> None:
    candles = _candles(["2024-05-01T00:00:00Z", "2024-07-01T00:00:00Z"])
    candles["timeframe"] = "1mo"
    candles["timestamp_close_utc"] = candles["timestamp_open_utc"] + pd.offsets.MonthBegin(1)

    report = build_daily_data_quality(
        candles,
        calendar=_weekday_calendar(),
        as_of_utc=pd.Timestamp("2024-08-01T00:00:00Z"),
        stale_after=pd.Timedelta(0),
    )

    june = next(record for record in report.records if record.day_utc == date(2024, 6, 1))
    assert june.market_status == "open"
    assert june.expected_count == 1
    assert june.missing_count == 1
    assert report.gaps[0].classification == "missing_data"


def test_partial_session_day_is_expected_but_partial_intraday_window_is_not() -> None:
    calendar = ExplicitMarketCalendarPolicy(
        weekly_open_windows=(
            WeeklyOpenWindow(weekday=6, start_minute_utc=1_350, end_minute_utc=1_440),
        ),
        weekend_weekdays=frozenset({5}),
        version="fixture-sunday-evening-v1",
    )

    day = calendar.classify(
        pd.Timestamp("2024-01-07T00:00:00Z"),
        source="fixture",
        instrument="XAU_USD",
        timeframe="1d",
    )
    hour = calendar.classify(
        pd.Timestamp("2024-01-07T22:00:00Z"),
        source="fixture",
        instrument="XAU_USD",
        timeframe="1h",
    )
    assert day.expected is True
    assert hour.expected is False


def test_explicit_all_day_holiday_removes_daily_candle_expectation() -> None:
    calendar = _weekday_calendar(
        holidays=(
            UtcMarketWindow(
                pd.Timestamp("2024-01-08T00:00:00Z"),
                pd.Timestamp("2024-01-09T00:00:00Z"),
                "fixture closure",
            ),
        ),
    )
    state = calendar.classify(
        pd.Timestamp("2024-01-08T00:00:00Z"),
        source="fixture",
        instrument="XAU_USD",
        timeframe="1d",
    )
    assert state.status == "holiday"
    assert state.expected is False


def test_declared_empty_monthly_identity_reports_closed_month_without_a_fake_row() -> None:
    report = build_daily_data_quality(
        _candles([]),
        calendar=ObservedSourceCalendarPolicy(),
        as_of_utc=pd.Timestamp("2024-03-01T00:00:00Z"),
        stale_after=pd.Timedelta(0),
        start_day_utc=date(2024, 2, 1),
        end_day_utc=date(2024, 2, 29),
        identities=(("histdata", "XAU_USD", "1mo"),),
    )

    assert len(report.records) == 29
    assert all(record.row_count == 0 for record in report.records)
    assert all(record.expected_count == 0 for record in report.records)
    assert len(report.gaps) == 1
    assert report.gaps[0].start_utc == pd.Timestamp("2024-02-01T00:00:00Z")
    assert report.gaps[0].end_utc == pd.Timestamp("2024-03-01T00:00:00Z")
    assert report.gaps[0].classification == "unknown_market_status"
    assert not report.gaps[0].counts_as_missing


def test_empty_daily_identity_under_trusted_calendar_counts_expected_absent_days() -> None:
    report = build_daily_data_quality(
        _candles([]),
        calendar=_weekday_calendar(),
        as_of_utc=pd.Timestamp("2024-01-09T00:00:00Z"),
        stale_after=pd.Timedelta(0),
        start_day_utc=date(2024, 1, 6),
        end_day_utc=date(2024, 1, 8),
        identities=(("replacement", "XAU_USD", "1d"),),
    )

    assert [record.expected_count for record in report.records] == [0, 0, 1]
    assert [record.missing_count for record in report.records] == [0, 0, 1]
    assert report.records[-1].quality_status == "fail"


def test_declared_empty_identity_requires_explicit_scope_and_deduplicates_existing_groups() -> None:
    candles = _candles(["2024-01-08T00:00:00Z"])
    kwargs = {
        "calendar": ObservedSourceCalendarPolicy(),
        "as_of_utc": pd.Timestamp("2024-01-09T00:00:00Z"),
        "stale_after": pd.Timedelta(0),
    }
    with pytest.raises(DataQualityError, match="empty declared identities require"):
        build_daily_data_quality(
            candles,
            identities=(("histdata", "XAU_USD", "1mo"),),
            **kwargs,
        )

    declared = build_daily_data_quality(
        candles,
        identities=(("histdata", "XAU_USD", "1min"), ("histdata", "XAU_USD", "1min")),
        **kwargs,
    )
    original = build_daily_data_quality(candles, **kwargs)
    assert_frame_equal(declared.to_frame(), original.to_frame())
    assert_frame_equal(declared.gaps_to_frame(), original.gaps_to_frame())


@pytest.mark.parametrize("bad_price", [True, "2000.25", float("inf"), float("nan"), -1.0])
def test_ohlc_quality_does_not_accept_coercible_non_prices(bad_price: object) -> None:
    candles = _candles(["2024-01-08T00:00:00Z"])
    candles["bid_close"] = pd.Series([bad_price], dtype=object)

    report = build_daily_data_quality(
        candles,
        calendar=ObservedSourceCalendarPolicy(),
        as_of_utc=pd.Timestamp("2024-01-09T00:00:00Z"),
        stale_after=pd.Timedelta(0),
    )

    assert report.records[0].ohlc_invalid_count == 1
    assert report.records[0].quality_status == "fail"
