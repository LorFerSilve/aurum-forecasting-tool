from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gold_forecasting.phase10.events import join_events, local_event_time


def _predictions(*times: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"prediction_time_utc": pd.to_datetime(list(times), format="mixed", utc=True)}
    )


def _event(**updates: Any) -> dict[str, Any]:
    return {
        "event_id": "cpi_january",
        "event_type": "us_cpi",
        "scheduled_at_utc": "2024-01-11 13:30:00",
        "available_at_utc": "2024-01-01 00:00:00",
        "revision_id": "initial",
        "status": "scheduled",
        "source_uri": "https://example.test/calendar.csv",
        "raw_sha256": "a" * 64,
        **updates,
    }


def _events(*rows: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in ("scheduled_at_utc", "available_at_utc"):
        frame[column] = pd.to_datetime(frame[column], format="mixed", utc=True)
    return frame


def test_default_event_phase_windows_have_exact_nanosecond_boundaries() -> None:
    predictions = _predictions(
        "2024-01-11 12:29:59.999999999",
        "2024-01-11 12:30:00",
        "2024-01-11 13:29:59.999999999",
        "2024-01-11 13:30:00",
        "2024-01-11 13:34:59.999999999",
        "2024-01-11 13:35:00",
        "2024-01-11 14:30:00",
        "2024-01-11 14:30:00.000000001",
    )

    result = join_events(predictions, _events(_event()))

    assert result["event_phase"].tolist() == [
        "normal", "before", "before", "during", "during", "after", "after", "normal"
    ]
    assert result[[f"event_{phase}" for phase in ("normal", "before", "during", "after")]].sum(
        axis=1
    ).eq(1).all()
    assert result.loc[3, "event_minutes_until"] == 0
    assert result.loc[3, "event_minutes_since"] == 0
    assert result.loc[2, "event_minutes_until"] > 0
    assert result.loc[7, "event_minutes_since"] > 60


def test_calendar_release_is_available_at_exact_cutoff_and_never_backfilled() -> None:
    result = join_events(
        _predictions("2024-01-11 13:29:59.999999999", "2024-01-11 13:30:00"),
        _events(_event(available_at_utc="2024-01-11 13:30:00")),
    )

    assert result["event_calendar_available"].tolist() == [False, True]
    assert result["event_is_missing"].tolist() == [True, False]
    assert pd.isna(result.loc[0, "event_phase"])
    assert pd.isna(result.loc[0, "event_normal"])
    assert pd.isna(result.loc[0, "event_next_id"])
    assert result.loc[1, "event_phase"] == "during"


def test_later_rescheduling_and_cancellation_preserve_earlier_features() -> None:
    first = _event()
    rescheduled = _event(
        revision_id="reschedule", available_at_utc="2024-01-11 13:20:00",
        scheduled_at_utc="2024-01-11 14:30:00",
    )
    cancelled = _event(
        revision_id="cancel", status="cancelled", available_at_utc="2024-01-11 13:40:00",
        scheduled_at_utc="2024-01-11 14:30:00",
    )
    predictions = _predictions(
        "2024-01-11 13:19:59.999999999", "2024-01-11 13:20:00",
        "2024-01-11 13:39:59.999999999", "2024-01-11 13:40:00",
    )

    result = join_events(predictions, _events(cancelled, first, rescheduled))

    assert result["event_phase"].tolist() == ["before", "normal", "before", "normal"]
    assert result["event_next_revision_id"].iloc[:3].tolist() == [
        "initial", "reschedule", "reschedule"
    ]
    assert pd.isna(result.loc[3, "event_next_id"])
    assert result.loc[3, "event_calendar_available"]
    assert_frame_equal(
        result.iloc[:1], join_events(predictions.iloc[:1], _events(first))
    )
    assert_frame_equal(
        result.iloc[:3], join_events(predictions.iloc[:3], _events(first, rescheduled))
    )


def test_future_calendar_mutation_cannot_change_past_features() -> None:
    predictions = _predictions("2024-01-11 13:00:00", "2024-01-11 13:30:00")
    events = _events(
        _event(),
        _event(revision_id="late", available_at_utc="2024-01-11 14:00:00"),
        _event(event_id="future_jobs", event_type="us_jobs",
               scheduled_at_utc="2024-02-02 13:30:00", available_at_utc="2024-01-12 00:00:00"),
    )
    mutated = events.copy()
    mutated.loc[1:, "scheduled_at_utc"] += pd.Timedelta(days=30)
    mutated.loc[1:, "status"] = "cancelled"
    mutated.loc[1:, "raw_sha256"] = "b" * 64

    assert_frame_equal(join_events(predictions, events), join_events(predictions, mutated))


def test_equal_time_events_choose_lexical_id_independently_of_input_order() -> None:
    events = _events(
        _event(event_id="z_jobs", event_type="us_jobs"),
        _event(event_id="a_cpi"),
    )
    predictions = _predictions("2024-01-11 13:00:00", "2024-01-11 13:30:00", "2024-01-11 14:00:00")

    result = join_events(predictions, events)

    assert result["event_next_id"].iloc[:2].tolist() == ["a_cpi", "a_cpi"]
    assert result["event_previous_id"].iloc[1:].tolist() == ["a_cpi", "a_cpi"]
    assert_frame_equal(result, join_events(predictions, events.iloc[::-1]))


def test_overlapping_events_prioritize_during_then_before_then_after() -> None:
    events = _events(
        _event(),
        _event(event_id="fomc", event_type="fomc", scheduled_at_utc="2024-01-11 14:00:00"),
    )
    result = join_events(
        _predictions("2024-01-11 13:31:00", "2024-01-11 13:36:00"), events
    )

    assert result["event_phase"].tolist() == ["during", "before"]
    assert result["event_previous_type"].tolist() == ["us_cpi", "us_cpi"]
    assert result["event_next_type"].tolist() == ["fomc", "fomc"]


@pytest.mark.parametrize("empty", [False, True])
def test_absent_calendar_preserves_index_and_does_not_assure_normal_conditions(empty: bool) -> None:
    predictions = _predictions("2024-01-11 13:00:00", "2024-01-11 13:30:00")
    predictions.index = pd.Index([42, 11], name="input_row")
    events = _events(_event()).iloc[:0] if empty else None

    result = join_events(predictions, events)

    assert_frame_equal(result[predictions.columns], predictions)
    assert result["event_is_missing"].all()
    assert not result["event_calendar_available"].any()
    for column in ("event_phase", "event_normal", "event_before", "event_during", "event_after"):
        assert result[column].isna().all()
    assert result["event_minutes_until"].isna().all()
    assert result["event_minutes_since"].isna().all()


def test_nondefault_index_and_original_columns_survive_populated_calendar() -> None:
    predictions = _predictions("2024-01-11 13:00:00", "2024-01-11 13:30:00")
    predictions.index = pd.Index([4, 2])
    predictions["price_feature"] = [7.0, 9.0]
    result = join_events(predictions, _events(_event()))
    assert_frame_equal(result[predictions.columns], predictions)
    assert result["event_phase"].tolist() == ["before", "during"]
    assert result["event_next_source_uri"].eq("https://example.test/calendar.csv").all()
    assert result["event_next_raw_sha256"].eq("a" * 64).all()


@pytest.mark.parametrize("timestamp", ["2019-12-31 23:59:59", "2025-01-01 00:00:00"])
def test_prediction_cutoffs_cannot_open_non_development_periods(timestamp: str) -> None:
    with pytest.raises(ValueError, match="development 2020-2024"):
        join_events(_predictions(timestamp), None)


def test_preannounced_future_schedule_is_not_a_holdout_actual_observation() -> None:
    result = join_events(
        _predictions("2024-12-31 23:30:00"),
        _events(_event(scheduled_at_utc="2025-01-01 00:00:00")),
    )
    assert result.loc[0, "event_phase"] == "before"
    assert result.loc[0, "event_minutes_until"] == 30


@pytest.mark.parametrize(
    ("local", "utc"),
    [
        (datetime(2024, 1, 11, 8, 30), "2024-01-11 13:30:00"),
        (datetime(2024, 7, 11, 8, 30), "2024-07-11 12:30:00"),
    ],
)
def test_us_release_time_uses_seasonal_source_timezone(local: datetime, utc: str) -> None:
    assert local_event_time(local, "America/New_York") == pd.Timestamp(utc, tz="UTC")


@pytest.mark.parametrize("local", [datetime(2024, 3, 10, 2, 30), datetime(2024, 11, 3, 1, 30)])
def test_nonexistent_or_ambiguous_dst_local_schedule_is_rejected(local: datetime) -> None:
    with pytest.raises(ValueError, match="ambiguous, nonexistent"):
        local_event_time(local, "America/New_York")


def test_local_timezone_helper_rejects_an_already_aware_datetime() -> None:
    with pytest.raises(ValueError, match="naive datetime"):
        local_event_time(pd.Timestamp("2024-01-01", tz="UTC"), "America/New_York")


@pytest.mark.parametrize("zone", [None, "America/New_York"])
@pytest.mark.parametrize("column", ["scheduled_at_utc", "available_at_utc"])
def test_core_rejects_implicit_event_timezone_conversion(column: str, zone: str | None) -> None:
    events = _events(_event())
    events[column] = events[column].dt.tz_convert(zone)
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        join_events(_predictions("2024-01-11 13:30:00"), events)


@pytest.mark.parametrize("times", [
    ["2024-01-11 13:30:00", "2024-01-11 13:00:00"],
    ["2024-01-11 13:30:00", "2024-01-11 13:30:00"],
])
def test_prediction_times_must_be_sorted_and_unique(times: list[str]) -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        join_events(_predictions(*times), None)


@pytest.mark.parametrize("zone", [None, "America/New_York"])
def test_prediction_times_require_explicit_utc(zone: str | None) -> None:
    predictions = _predictions("2024-01-11 13:30:00")
    predictions["prediction_time_utc"] = predictions["prediction_time_utc"].dt.tz_convert(zone)
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        join_events(predictions, None)


@pytest.mark.parametrize(("update", "message"), [
    ({"status": "actual"}, "status"),
    ({"event_type": "unknown"}, "event_type"),
    ({"source_uri": ""}, "source_uri"),
    ({"raw_sha256": "not-a-hash"}, "raw_sha256"),
    ({"revision_id": 4}, "revision_id"),
])
def test_invalid_schedule_contract_is_rejected(update: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        join_events(_predictions("2024-01-11 13:30:00"), _events(_event(**update)))


def test_actual_or_surprise_fields_are_outside_the_schedule_contract() -> None:
    events = _events(_event())
    events["actual"] = 3.1
    with pytest.raises(ValueError, match="exactly the schedule"):
        join_events(_predictions("2024-01-11 13:30:00"), events)


@pytest.mark.parametrize(("revision", "message"), [
    ({"revision_id": "second"}, "ambiguous event revisions"),
    ({"available_at_utc": "2024-01-02 00:00:00"}, "duplicate event revision identity"),
    ({"available_at_utc": "2024-01-02 00:00:00", "revision_id": "second", "event_type": "fomc"},
     "event_type must remain stable"),
])
def test_ambiguous_or_inconsistent_revision_identity_fails_closed(
    revision: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        join_events(_predictions("2024-01-11 13:30:00"), _events(_event(), _event(**revision)))


@pytest.mark.parametrize(("name", "value"), [
    ("pre_event_minutes", -1), ("post_event_minutes", np.nan),
    ("during_event_minutes", 0), ("during_event_minutes", np.inf),
    ("pre_event_minutes", True),
])
def test_invalid_event_window_is_rejected(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=name):
        join_events(_predictions("2024-01-11 13:30:00"), None, **{name: value})


def test_output_columns_cannot_overwrite_input_features() -> None:
    predictions = _predictions("2024-01-11 13:30:00")
    predictions["event_phase"] = "existing"
    with pytest.raises(ValueError, match="overwrite"):
        join_events(predictions, None)
