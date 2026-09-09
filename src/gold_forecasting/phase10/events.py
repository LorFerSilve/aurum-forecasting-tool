"""Point-in-time scheduled-event context for optional research fixtures.

Calendar times are schedules, never actual release values or surprise measurements.
Future schedules may extend beyond 2024; prediction cutoffs may not. Availability
means schedule evidence exists as of the cutoff, not that a calendar is complete.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import dataclass
from datetime import datetime
from numbers import Real

import numpy as np
import pandas as pd

from gold_forecasting.phase10.contracts import DEVELOPMENT_END, DEVELOPMENT_START, utc_series

EVENT_COLUMNS = (
    "event_id", "event_type", "scheduled_at_utc", "available_at_utc", "revision_id",
    "status", "source_uri", "raw_sha256",
)
_DETAILS = (
    "id", "type", "scheduled_at_utc", "available_at_utc", "revision_id",
    "source_uri", "raw_sha256",
)
_PHASES = ("before", "during", "after", "normal")
_OUTPUT_COLUMNS = (
    "event_calendar_available", "event_is_missing", "event_minutes_until", "event_minutes_since",
    "event_phase", *(f"event_{phase}" for phase in _PHASES),
    *(f"event_{side}_{field}" for side in ("next", "previous") for field in _DETAILS),
)


@dataclass(frozen=True)
class _Event:
    event_id: str
    event_type: str
    scheduled_at_utc: pd.Timestamp
    available_at_utc: pd.Timestamp
    revision_id: str
    status: str
    source_uri: str
    raw_sha256: str

    @property
    def key(self) -> tuple[int, str]:
        return self.scheduled_at_utc.value, self.event_id


def local_event_time(value: datetime, timezone: str) -> pd.Timestamp:
    """Explicitly convert a naive provider-local schedule; reject DST ambiguity."""
    if not isinstance(value, datetime) or pd.isna(value) or value.tzinfo is not None:
        raise ValueError("local event time must be a non-null naive datetime")
    if not isinstance(timezone, str) or not timezone.strip():
        raise ValueError("event source timezone is required")
    try:
        return pd.Timestamp(value).tz_localize(
            timezone, ambiguous="raise", nonexistent="raise"
        ).tz_convert("UTC")
    except Exception as error:
        raise ValueError(
            "local event time is ambiguous, nonexistent or has an invalid timezone"
        ) from error


def _validate_events(events: pd.DataFrame) -> list[_Event]:
    if not events.columns.is_unique:
        raise ValueError("event columns must be unique")
    if set(events.columns) != set(EVENT_COLUMNS):
        raise ValueError("event table requires exactly the schedule and provenance columns")
    frame = events.loc[:, list(EVENT_COLUMNS)].copy()
    for name in ("scheduled_at_utc", "available_at_utc"):
        frame[name] = utc_series(frame[name], name)
    for name in set(EVENT_COLUMNS).difference(("scheduled_at_utc", "available_at_utc")):
        if not frame[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{name} must contain nonempty strings")
    if not frame["event_type"].isin(("us_cpi", "us_jobs", "fomc")).all():
        raise ValueError("event_type must be us_cpi, us_jobs or fomc")
    if not frame["status"].isin(("scheduled", "cancelled")).all():
        raise ValueError("event status must be scheduled or cancelled")
    if not frame["raw_sha256"].str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("raw_sha256 must identify the original event schedule artifact")
    if frame.duplicated(["event_id", "available_at_utc"]).any():
        raise ValueError("ambiguous event revisions at the same availability time")
    if frame.duplicated(["event_id", "revision_id"]).any():
        raise ValueError("duplicate event revision identity")
    if frame.groupby("event_id")["event_type"].nunique().gt(1).any():
        raise ValueError("event_type must remain stable across revisions of an event")
    frame = frame.sort_values(["available_at_utc", "event_id"])
    return [_Event(*row) for row in frame.itertuples(index=False, name=None)]


def _window_minutes(value: float, name: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def join_events(
    predictions: pd.DataFrame,
    events: pd.DataFrame | None,
    *,
    pre_event_minutes: float = 60,
    post_event_minutes: float = 60,
    during_event_minutes: float = 5,
) -> pd.DataFrame:
    """Join latest released schedules without revising earlier prediction rows.

    Before covers [event-pre, event); during covers [event, event+during);
    after covers [event+during, event+post]. Where events overlap the precedence
    is during, before, after, normal. Equal-time events use lexical event IDs for
    next/previous provenance. At the event instant both distances are zero.

    Without any released calendar evidence phase/flags remain unknown. A known
    cancellation removes that event from the schedule. These generic fixtures
    are not automatically admitted to a benchmark or integrated into a champion.
    """
    pre = _window_minutes(pre_event_minutes, "pre_event_minutes")
    post = _window_minutes(post_event_minutes, "post_event_minutes")
    during = _window_minutes(during_event_minutes, "during_event_minutes", positive=True)
    if not predictions.columns.is_unique:
        raise ValueError("prediction columns must be unique")
    if "prediction_time_utc" not in predictions:
        raise ValueError("prediction_time_utc is required")
    times = utc_series(predictions["prediction_time_utc"], "prediction_time_utc")
    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("prediction times must be sorted and unique")
    if ((times < DEVELOPMENT_START) | (times >= DEVELOPMENT_END)).any():
        raise ValueError("prediction times must remain in development 2020-2024")
    if set(_OUTPUT_COLUMNS).intersection(predictions.columns):
        raise ValueError("event columns would overwrite existing prediction fields")
    releases = _validate_events(events) if events is not None else []
    result = predictions.copy()
    detail: dict[str, list[object]] = {
        f"event_{side}_{field}": [] for side in ("next", "previous") for field in _DETAILS
    }
    available: list[bool] = []
    until_values: list[float] = []
    since_values: list[float] = []
    phases: list[str | None] = []
    latest: dict[str, _Event] = {}
    active: list[tuple[int, str]] = []
    released = 0
    for cutoff in times:
        while released < len(releases) and releases[released].available_at_utc <= cutoff:
            event = releases[released]
            previous = latest.get(event.event_id)
            if previous is not None and previous.status == "scheduled":
                active.pop(bisect_left(active, previous.key))
            latest[event.event_id] = event
            if event.status == "scheduled":
                insort(active, event.key)
            released += 1
        next_index = bisect_left(active, (cutoff.value, ""))
        previous_index = bisect_left(active, (cutoff.value + 1, "")) - 1
        upcoming = latest[active[next_index][1]] if next_index < len(active) else None
        recent = None
        if previous_index >= 0:
            first_tie = bisect_left(active, (active[previous_index][0], ""))
            recent = latest[active[first_tie][1]]
        # Timedelta.total_seconds truncates nanoseconds: use the integer delta.
        until = (upcoming.scheduled_at_utc.value - cutoff.value) / 60e9 if upcoming else np.nan
        since = (cutoff.value - recent.scheduled_at_utc.value) / 60e9 if recent else np.nan
        known = bool(released)
        if not known:
            phase = None
        elif 0 <= since < during:
            phase = "during"
        elif 0 < until <= pre:
            phase = "before"
        elif during <= since <= post:
            phase = "after"
        else:
            phase = "normal"
        available.append(known)
        until_values.append(until)
        since_values.append(since)
        phases.append(phase)
        for side, selected in (("next", upcoming), ("previous", recent)):
            for field in _DETAILS:
                attribute = {"id": "event_id", "type": "event_type"}.get(field, field)
                detail[f"event_{side}_{field}"].append(
                    getattr(selected, attribute) if selected else None
                )
    result["event_calendar_available"] = np.asarray(available, dtype=bool)
    result["event_is_missing"] = np.logical_not(available)
    result["event_minutes_until"] = np.asarray(until_values, dtype=float)
    result["event_minutes_since"] = np.asarray(since_values, dtype=float)
    result["event_phase"] = pd.array(phases, dtype="string")
    for phase in _PHASES:
        result[f"event_{phase}"] = pd.array(
            [value == phase if value is not None else None for value in phases], dtype="boolean"
        )
    for name, values in detail.items():
        dtype = "datetime64[ns, UTC]" if name.endswith("_at_utc") else "string"
        result[name] = pd.Series(values, index=result.index, dtype=dtype)
    return result
