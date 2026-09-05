"""Deterministic daily data-quality records with explicit market-calendar semantics.

The audit deliberately accepts defective candle rows so it can count them. It
never repairs prices, fills gaps, or guesses whether an OTC market was open.
Callers must choose a calendar policy. The conservative
``ObservedSourceCalendarPolicy`` only measures gaps inside observed coverage and
marks every closure whose cause is unknown.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Any, Literal, Protocol, cast, runtime_checkable

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from gold_forecasting.validation.candles import CANDLE_KEY_COLUMNS, PRICE_COLUMNS

MarketStatus = Literal["open", "weekend", "holiday", "maintenance", "unknown"]
ExpectationBasis = Literal["calendar", "observed_source"]
GapClassification = Literal[
    "missing_data",
    "known_gap",
    "scheduled_closure",
    "unknown_market_status",
]
QualityStatus = Literal["pass", "marked", "fail"]

_GROUP_COLUMNS = ("source", "instrument", "timeframe")
_AUDIT_COLUMNS = (
    "timestamp_open_utc",
    *PRICE_COLUMNS,
    "instrument",
    "timeframe",
    "is_complete",
    "source",
)
_FIXED_TIMEFRAME = re.compile(r"^(?P<count>[1-9][0-9]*)(?P<unit>min|h|d)$")


class DataQualityError(ValueError):
    """Raised when a quality audit cannot interpret its input contract."""


@dataclass(frozen=True, slots=True)
class MarketState:
    """Calendar classification for a single canonical candle open."""

    status: MarketStatus
    expected: bool | None
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("market-state reason must be non-empty")
        if self.status == "open" and self.expected is not True:
            raise ValueError("market status 'open' must be expected")
        if self.status in {"weekend", "holiday", "maintenance"} and self.expected is not False:
            raise ValueError(f"market status {self.status!r} must not be expected")


@runtime_checkable
class MarketCalendarPolicy(Protocol):
    """Explicit, versioned source of market-state decisions."""

    @property
    def version(self) -> str:
        """Stable identifier for the calendar rules used by the audit."""

    @property
    def expectation_basis(self) -> ExpectationBasis:
        """Whether expected counts come from a calendar or observed coverage."""

    def classify(
        self,
        timestamp_open_utc: pd.Timestamp,
        *,
        source: str,
        instrument: str,
        timeframe: str,
    ) -> MarketState:
        """Classify one UTC-aligned candle open."""


@dataclass(frozen=True, slots=True)
class ObservedSourceCalendarPolicy:
    """Fail-open-on-knowledge policy for sources without a trusted calendar.

    Expected counts are limited to the inclusive span from the first through
    the last observed candle in each UTC day. Gaps across days remain visible in
    the gap records, but are marked ``unknown_market_status`` instead of being
    mislabeled as holidays or weekends.
    """

    version: str = "observed-source-v1"
    expectation_basis: ExpectationBasis = field(default="observed_source", init=False)

    def classify(
        self,
        timestamp_open_utc: pd.Timestamp,
        *,
        source: str,
        instrument: str,
        timeframe: str,
    ) -> MarketState:
        del timestamp_open_utc, source, instrument, timeframe
        return MarketState(
            status="unknown",
            expected=None,
            reason="provider calendar unavailable; closure cause is not inferred",
        )


@dataclass(frozen=True, slots=True)
class WeeklyOpenWindow:
    """One same-day UTC market-open interval, expressed as minutes after midnight."""

    weekday: int
    start_minute_utc: int
    end_minute_utc: int

    def __post_init__(self) -> None:
        if not 0 <= self.weekday <= 6:
            raise ValueError("weekday must be in the inclusive range 0..6")
        if not 0 <= self.start_minute_utc < self.end_minute_utc <= 1_440:
            raise ValueError("weekly open window must satisfy 0 <= start < end <= 1440")

    def contains(self, timestamp: pd.Timestamp) -> bool:
        minute = timestamp.hour * 60 + timestamp.minute
        return (
            timestamp.weekday() == self.weekday
            and self.start_minute_utc <= minute < self.end_minute_utc
        )


@dataclass(frozen=True, slots=True)
class UtcMarketWindow:
    """One explicit half-open holiday or maintenance interval."""

    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    reason: str

    def __post_init__(self) -> None:
        start = _as_utc(self.start_utc, field_name="start_utc")
        end = _as_utc(self.end_utc, field_name="end_utc")
        if start >= end:
            raise ValueError("UTC market window start must be before end")
        if not self.reason.strip():
            raise ValueError("UTC market window reason must be non-empty")
        object.__setattr__(self, "start_utc", start)
        object.__setattr__(self, "end_utc", end)

    def contains(self, timestamp: pd.Timestamp) -> bool:
        return self.start_utc <= timestamp < self.end_utc


@dataclass(frozen=True, slots=True)
class ExplicitMarketCalendarPolicy:
    """Calendar whose market hours and exceptions are all supplied explicitly.

    Dates not covered by an open window, weekend rule, holiday, or maintenance
    window are treated as a known out-of-session closure with status ``unknown``.
    This distinguishes an explicitly closed schedule from the genuinely unknown
    calendar of :class:`ObservedSourceCalendarPolicy` via ``expected=False``.
    Intraday candles are expected only if their entire interval is scheduled
    open (the dense resampling policy). Daily and monthly candles are expected
    if any scheduled open interval remains after explicit closures (the sparse
    resampling policy). A month's opening weekday alone never determines the
    state of the entire month.
    """

    weekly_open_windows: tuple[WeeklyOpenWindow, ...]
    weekend_weekdays: frozenset[int] = frozenset({5, 6})
    holidays: tuple[UtcMarketWindow, ...] = ()
    maintenance_windows: tuple[UtcMarketWindow, ...] = ()
    version: str = "explicit-calendar-v1"
    expectation_basis: ExpectationBasis = field(default="calendar", init=False)

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("calendar version must be non-empty")
        if any(day < 0 or day > 6 for day in self.weekend_weekdays):
            raise ValueError("weekend weekdays must be in the inclusive range 0..6")
        ordered = tuple(
            sorted(
                self.weekly_open_windows,
                key=lambda item: (item.weekday, item.start_minute_utc, item.end_minute_utc),
            )
        )
        object.__setattr__(self, "weekly_open_windows", ordered)

    def classify(
        self,
        timestamp_open_utc: pd.Timestamp,
        *,
        source: str,
        instrument: str,
        timeframe: str,
    ) -> MarketState:
        del source, instrument
        timestamp = _as_utc(timestamp_open_utc, field_name="timestamp_open_utc")
        period = _parse_timeframe(timeframe)
        end = period.close_of(timestamp)
        # Weekly/session boundaries are minute aligned. Avoid constructing a
        # daily boundary grid for every minute in a multi-year audit unless an
        # explicitly supplied exception actually divides this minute.
        if timeframe == "1min" and not any(
            timestamp < boundary < end
            for exception in (*self.holidays, *self.maintenance_windows)
            for boundary in (exception.start_utc, exception.end_utc)
        ):
            return self._classify_at(timestamp)
        boundaries = {timestamp, end}
        for day in pd.date_range(timestamp.normalize(), end, freq="D", inclusive="left"):
            boundaries.add(max(timestamp, day))
            for window in self.weekly_open_windows:
                if day.weekday() != window.weekday:
                    continue
                for minute in (window.start_minute_utc, window.end_minute_utc):
                    boundary = day + pd.Timedelta(minutes=minute)
                    if timestamp < boundary < end:
                        boundaries.add(boundary)
        for exception in (*self.holidays, *self.maintenance_windows):
            for boundary in (exception.start_utc, exception.end_utc):
                if timestamp < boundary < end:
                    boundaries.add(boundary)
        states = [self._classify_at(boundary) for boundary in sorted(boundaries)[:-1]]
        expected_states = [state for state in states if state.expected is True]
        is_sparse = timeframe in {"1d", "1mo"}
        if expected_states and (is_sparse or len(expected_states) == len(states)):
            return MarketState("open", True, "configured market hours within candle interval")
        if all(state == states[0] for state in states):
            return states[0]
        return MarketState("unknown", False, "candle interval includes configured market closures")

    def _classify_at(self, timestamp: pd.Timestamp) -> MarketState:
        maintenance = _matching_window(timestamp, self.maintenance_windows)
        if maintenance is not None:
            return MarketState("maintenance", False, maintenance.reason)
        holiday = _matching_window(timestamp, self.holidays)
        if holiday is not None:
            return MarketState("holiday", False, holiday.reason)
        if timestamp.weekday() in self.weekend_weekdays:
            return MarketState("weekend", False, "configured weekend closure")
        if any(window.contains(timestamp) for window in self.weekly_open_windows):
            return MarketState("open", True, "configured weekly market hours")
        return MarketState("unknown", False, "outside configured weekly market hours")


@dataclass(frozen=True, slots=True)
class KnownGap:
    """Explicit explanation for a half-open interval with absent candles."""

    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    code: str
    reason: str
    source: str | None = None
    instrument: str | None = None
    timeframe: str | None = None

    def __post_init__(self) -> None:
        start = _as_utc(self.start_utc, field_name="start_utc")
        end = _as_utc(self.end_utc, field_name="end_utc")
        if start >= end:
            raise ValueError("known-gap start must be before end")
        if not self.code.strip() or not self.reason.strip():
            raise ValueError("known-gap code and reason must be non-empty")
        object.__setattr__(self, "start_utc", start)
        object.__setattr__(self, "end_utc", end)

    def matches(
        self,
        timestamp: pd.Timestamp,
        *,
        source: str,
        instrument: str,
        timeframe: str,
    ) -> bool:
        identities_match = (
            (self.source is None or self.source == source)
            and (self.instrument is None or self.instrument == instrument)
            and (self.timeframe is None or self.timeframe == timeframe)
        )
        return identities_match and self.start_utc <= timestamp < self.end_utc


@dataclass(frozen=True, slots=True)
class GapQualityRecord:
    """One contiguous absent-candle segment with an explicit explanation state."""

    day_utc: date
    source: str
    instrument: str
    timeframe: str
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    missing_intervals: int
    market_status: MarketStatus
    classification: GapClassification
    reason: str
    annotation_code: str | None
    counts_as_missing: bool
    is_stale: bool
    is_explained: bool

    def as_record(self) -> dict[str, object]:
        return {
            "day_utc": self.day_utc.isoformat(),
            "source": self.source,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "missing_intervals": self.missing_intervals,
            "market_status": self.market_status,
            "classification": self.classification,
            "reason": self.reason,
            "annotation_code": self.annotation_code,
            "counts_as_missing": self.counts_as_missing,
            "is_stale": self.is_stale,
            "is_explained": self.is_explained,
        }


@dataclass(frozen=True, slots=True)
class DailyDataQualityRecord:
    """Deterministic quality counters for one UTC day and identity group."""

    day_utc: date
    source: str
    instrument: str
    timeframe: str
    calendar_version: str
    expectation_basis: ExpectationBasis
    market_status: MarketStatus
    row_count: int
    observed_count: int
    expected_count: int
    missing_count: int
    stale_count: int
    incomplete_count: int
    duplicate_count: int
    ohlc_invalid_count: int
    gap_count: int
    explained_gap_count: int
    unknown_gap_count: int
    unexplained_gap_count: int
    quality_status: QualityStatus

    def as_record(self) -> dict[str, object]:
        record = {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}
        record["day_utc"] = self.day_utc.isoformat()
        return record


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    """Sorted daily records and gap evidence from one deterministic audit."""

    records: tuple[DailyDataQualityRecord, ...]
    gaps: tuple[GapQualityRecord, ...]
    as_of_utc: pd.Timestamp

    def to_frame(self) -> pd.DataFrame:
        columns = tuple(DailyDataQualityRecord.__dataclass_fields__)
        return pd.DataFrame.from_records(
            (record.as_record() for record in self.records), columns=columns
        )

    def gaps_to_frame(self) -> pd.DataFrame:
        columns = tuple(GapQualityRecord.__dataclass_fields__)
        return pd.DataFrame.from_records((gap.as_record() for gap in self.gaps), columns=columns)


@dataclass(frozen=True, slots=True)
class _Timeframe:
    code: str
    delta: pd.Timedelta | None

    def grid_for_day(self, day: date) -> pd.DatetimeIndex:
        start = pd.Timestamp(day, tz="UTC")
        if self.code == "1mo":
            values = [start] if day.day == 1 else []
            return pd.DatetimeIndex(values, dtype="datetime64[ns, UTC]")
        assert self.delta is not None
        periods = int(pd.Timedelta(days=1) // self.delta)
        if periods <= 0 or pd.Timedelta(days=1) % self.delta != pd.Timedelta(0):
            raise DataQualityError(
                f"timeframe {self.code!r} must divide a UTC day for daily quality records"
            )
        return pd.date_range(start, periods=periods, freq=self.delta, tz="UTC")

    def close_of(self, timestamp: pd.Timestamp) -> pd.Timestamp:
        if self.code == "1mo":
            return timestamp + pd.offsets.MonthBegin(1)
        assert self.delta is not None
        return timestamp + self.delta

    def closes_of(self, timestamps: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if self.code == "1mo":
            return timestamps + pd.offsets.MonthBegin(1)
        assert self.delta is not None
        return timestamps + self.delta

    def validate_alignment(self, timestamps: pd.DatetimeIndex) -> None:
        if self.code == "1mo":
            aligned = (timestamps.day == 1) & (timestamps == timestamps.normalize())
        else:
            assert self.delta is not None
            aligned = timestamps == timestamps.floor(self.code)
        if not aligned.all():
            raise DataQualityError(f"{self.code} candle opens must follow the canonical UTC grid")


@dataclass(frozen=True, slots=True)
class _GapSlot:
    timestamp: pd.Timestamp
    state: MarketState
    classification: GapClassification
    reason: str
    annotation_code: str | None
    counts_as_missing: bool
    is_stale: bool
    is_explained: bool

    @property
    def grouping_key(self) -> tuple[object, ...]:
        return (
            self.state.status,
            self.classification,
            self.reason,
            self.annotation_code,
            self.counts_as_missing,
            self.is_stale,
            self.is_explained,
        )


def build_daily_data_quality(
    candles: pd.DataFrame,
    *,
    calendar: MarketCalendarPolicy,
    as_of_utc: pd.Timestamp,
    stale_after: pd.Timedelta,
    known_gaps: Sequence[KnownGap] = (),
    start_day_utc: date | None = None,
    end_day_utc: date | None = None,
    identities: Sequence[tuple[str, str, str]] = (),
) -> DataQualityReport:
    """Audit candle quality without repairing, interpolating, or dropping rows.

    ``start_day_utc`` and ``end_day_utc`` are inclusive. When omitted, the
    audit scope is bounded by the first and last observed candle per identity.
    ``as_of_utc`` is mandatory so stale counts never depend on wall-clock time.
    Only windows closed by that cutoff can be missing. Present rows whose
    canonical close is later than the cutoff count as incomplete even when a
    provider incorrectly labels them complete. Input opens must use the
    canonical UTC grid; otherwise missing-interval counts would be ambiguous.
    ``identities`` declares additional (source, instrument, timeframe) groups,
    including entirely absent datasets. Empty groups require both explicit
    scope dates so the audit does not invent coverage or placeholder candles.
    """

    _validate_policy(calendar)
    as_of = _as_utc(as_of_utc, field_name="as_of_utc")
    stale_delta = pd.Timedelta(stale_after)
    if pd.isna(stale_delta) or stale_delta < pd.Timedelta(0):
        raise DataQualityError("stale_after must be finite and non-negative")
    if start_day_utc is not None and end_day_utc is not None and start_day_utc > end_day_utc:
        raise DataQualityError("start_day_utc must not be after end_day_utc")

    prepared = _prepare_for_audit(candles)
    declared: set[tuple[str, str, str]] = set()
    for identity in identities:
        if len(identity) != 3 or any(
            not isinstance(value, str) or not value.strip() for value in identity
        ):
            raise DataQualityError(
                "identities must contain (source, instrument, timeframe) strings"
            )
        _parse_timeframe(identity[2])
        declared.add(identity)
    if prepared.empty and not declared:
        return DataQualityReport(records=(), gaps=(), as_of_utc=as_of)

    records: list[DailyDataQualityRecord] = []
    all_gaps: list[GapQualityRecord] = []
    ordered = prepared.sort_values([*_GROUP_COLUMNS, "timestamp_open_utc"], kind="stable")
    groups: dict[tuple[str, str, str], pd.DataFrame] = {}
    for raw_key, group in ordered.groupby(list(_GROUP_COLUMNS), sort=True, dropna=False):
        groups[cast(tuple[str, str, str], raw_key)] = group
    for identity in declared - groups.keys():
        groups[identity] = ordered.iloc[:0]
    for identity, group in sorted(groups.items()):
        source, instrument, timeframe_code = identity
        timeframe = _parse_timeframe(timeframe_code)
        group_records, group_gaps = _audit_group(
            group,
            source=source,
            instrument=instrument,
            timeframe=timeframe,
            calendar=calendar,
            as_of=as_of,
            stale_after=stale_delta,
            known_gaps=known_gaps,
            requested_start=start_day_utc,
            requested_end=end_day_utc,
        )
        records.extend(group_records)
        all_gaps.extend(group_gaps)

    return DataQualityReport(
        records=tuple(records),
        gaps=tuple(all_gaps),
        as_of_utc=as_of,
    )


def _audit_group(
    group: pd.DataFrame,
    *,
    source: str,
    instrument: str,
    timeframe: _Timeframe,
    calendar: MarketCalendarPolicy,
    as_of: pd.Timestamp,
    stale_after: pd.Timedelta,
    known_gaps: Sequence[KnownGap],
    requested_start: date | None,
    requested_end: date | None,
) -> tuple[list[DailyDataQualityRecord], list[GapQualityRecord]]:
    if group.empty and (requested_start is None or requested_end is None):
        raise DataQualityError("empty declared identities require start_day_utc and end_day_utc")
    # The caller sorted each identity by open time. Binary-search slices avoid
    # rescanning multi-year minute histories once for every UTC day.
    timestamps = pd.DatetimeIndex(group["timestamp_open_utc"], dtype="datetime64[ns, UTC]")
    timeframe.validate_alignment(timestamps)
    incomplete = _incomplete_mask(group["is_complete"]).to_numpy(dtype=bool)
    incomplete |= timeframe.closes_of(timestamps) > as_of
    duplicates = group.duplicated(subset=list(CANDLE_KEY_COLUMNS), keep="first").to_numpy()
    invalid_ohlc = _invalid_ohlc_mask(group).to_numpy(dtype=bool)
    observed_start = cast(pd.Timestamp, group["timestamp_open_utc"].min())
    observed_end = cast(pd.Timestamp, group["timestamp_open_utc"].max())
    first_day = requested_start or observed_start.date()
    last_day = requested_end or observed_end.date()
    if first_day > last_day:
        return [], []

    explicit_start = requested_start is not None
    explicit_end = requested_end is not None
    scope_start = pd.Timestamp(first_day, tz="UTC") if explicit_start else observed_start
    scope_end = (
        pd.Timestamp(last_day, tz="UTC") + pd.Timedelta(days=1)
        if explicit_end
        else timeframe.close_of(observed_end)
    )

    daily_rows: list[DailyDataQualityRecord] = []
    daily_gaps: list[GapQualityRecord] = []
    for day_timestamp in pd.date_range(first_day, last_day, freq="D"):
        day = day_timestamp.date()
        day_start = pd.Timestamp(day, tz="UTC")
        day_end = day_start + pd.Timedelta(days=1)
        left = int(timestamps.searchsorted(day_start, side="left"))
        right = int(timestamps.searchsorted(day_end, side="left"))
        rows = group.iloc[left:right]
        grid = timeframe.grid_for_day(day)
        in_scope = grid[
            (grid >= scope_start) & (grid < scope_end) & (timeframe.closes_of(grid) <= as_of)
        ]
        states = {
            timestamp: calendar.classify(
                timestamp,
                source=source,
                instrument=instrument,
                timeframe=timeframe.code,
            )
            for timestamp in in_scope
        }
        _validate_states(states)

        observed = timestamps[left:right].unique()
        expected = _expected_grid(in_scope, observed, calendar=calendar, states=states)
        observed_set = set(observed)
        missing_expected = [timestamp for timestamp in expected if timestamp not in observed_set]
        missing_expected_set = set(missing_expected)
        stale_count = sum(
            timeframe.close_of(timestamp) + stale_after <= as_of
            for timestamp in missing_expected
        )
        absent = [timestamp for timestamp in in_scope if timestamp not in observed_set]
        gap_slots = [
            _classify_gap_slot(
                timestamp,
                states[timestamp],
                source=source,
                instrument=instrument,
                timeframe=timeframe.code,
                known_gaps=known_gaps,
                counts_as_missing=timestamp in missing_expected_set,
                is_stale=(
                    timestamp in missing_expected_set
                    and timeframe.close_of(timestamp) + stale_after <= as_of
                ),
            )
            for timestamp in absent
        ]
        gap_records = _collapse_gap_slots(
            gap_slots,
            day=day,
            source=source,
            instrument=instrument,
            timeframe=timeframe,
        )
        daily_gaps.extend(gap_records)

        duplicate_count = int(duplicates[left:right].sum())
        incomplete_count = int(incomplete[left:right].sum())
        invalid_ohlc_count = int(invalid_ohlc[left:right].sum())
        market_status = _aggregate_market_status(states.values())
        explained = sum(gap.is_explained for gap in gap_records)
        unknown = sum(gap.classification == "unknown_market_status" for gap in gap_records)
        unexplained = sum(gap.classification == "missing_data" for gap in gap_records)
        quality_status = _quality_status(
            duplicate_count=duplicate_count,
            incomplete_count=incomplete_count,
            invalid_ohlc_count=invalid_ohlc_count,
            unexplained_gap_count=unexplained,
            explained_gap_count=explained,
            unknown_gap_count=unknown,
        )
        daily_rows.append(
            DailyDataQualityRecord(
                day_utc=day,
                source=source,
                instrument=instrument,
                timeframe=timeframe.code,
                calendar_version=calendar.version,
                expectation_basis=calendar.expectation_basis,
                market_status=market_status,
                row_count=len(rows),
                observed_count=len(observed),
                expected_count=len(expected),
                missing_count=len(missing_expected),
                stale_count=stale_count,
                incomplete_count=incomplete_count,
                duplicate_count=duplicate_count,
                ohlc_invalid_count=invalid_ohlc_count,
                gap_count=len(gap_records),
                explained_gap_count=explained,
                unknown_gap_count=unknown,
                unexplained_gap_count=unexplained,
                quality_status=quality_status,
            )
        )
    return daily_rows, daily_gaps


def _prepare_for_audit(candles: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(candles, pd.DataFrame):
        raise DataQualityError("candles must be a pandas DataFrame")
    missing = [column for column in _AUDIT_COLUMNS if column not in candles.columns]
    if missing:
        raise DataQualityError(f"missing data-quality columns: {', '.join(missing)}")
    prepared = candles.loc[:, list(_AUDIT_COLUMNS)].copy()
    if prepared.empty:
        return prepared
    prepared["timestamp_open_utc"] = _normalize_timestamp_series(
        prepared["timestamp_open_utc"], "timestamp_open_utc"
    )
    for column in _GROUP_COLUMNS:
        identities = prepared[column].unique()
        if any(not isinstance(value, str) or not value.strip() for value in identities):
            raise DataQualityError(f"{column} must contain non-empty strings")
    for timeframe in sorted(set(prepared["timeframe"])):
        _parse_timeframe(str(timeframe))
    return prepared


def _normalize_timestamp_series(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any():
        raise DataQualityError(f"{name} must not contain missing timestamps")
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return cast(pd.Series, values.dt.tz_convert("UTC"))
    normalized: list[pd.Timestamp] = []
    for value in values:
        normalized.append(_as_utc(value, field_name=name))
    return pd.Series(
        pd.DatetimeIndex(normalized, dtype="datetime64[ns, UTC]"),
        index=values.index,
        name=values.name,
    )


@lru_cache(maxsize=32)
def _parse_timeframe(value: str) -> _Timeframe:
    if value == "1mo":
        return _Timeframe(code=value, delta=None)
    match = _FIXED_TIMEFRAME.fullmatch(value)
    if match is None:
        raise DataQualityError(f"unsupported or ambiguous timeframe: {value!r}")
    count = int(match.group("count"))
    unit = cast(Literal["min", "h", "d"], match.group("unit"))
    if unit == "min":
        delta = pd.Timedelta(minutes=count)
    elif unit == "h":
        delta = pd.Timedelta(hours=count)
    else:
        delta = pd.Timedelta(days=count)
    return _Timeframe(code=value, delta=delta)


def _as_utc(value: object, *, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise DataQualityError(f"{field_name} must be a valid timestamp") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise DataQualityError(f"{field_name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _matching_window(
    timestamp: pd.Timestamp, windows: Sequence[UtcMarketWindow]
) -> UtcMarketWindow | None:
    return next((window for window in windows if window.contains(timestamp)), None)


def _validate_policy(calendar: MarketCalendarPolicy) -> None:
    if not isinstance(calendar, MarketCalendarPolicy):
        raise DataQualityError("calendar must implement MarketCalendarPolicy")
    if not calendar.version.strip():
        raise DataQualityError("calendar version must be non-empty")
    if calendar.expectation_basis not in {"calendar", "observed_source"}:
        raise DataQualityError("calendar expectation_basis is unsupported")


def _validate_states(states: Mapping[pd.Timestamp, MarketState]) -> None:
    for timestamp, state in states.items():
        if not isinstance(state, MarketState):
            raise DataQualityError(f"calendar returned an invalid state for {timestamp}")


def _expected_grid(
    grid: pd.DatetimeIndex,
    observed: pd.DatetimeIndex,
    *,
    calendar: MarketCalendarPolicy,
    states: Mapping[pd.Timestamp, MarketState],
) -> list[pd.Timestamp]:
    if calendar.expectation_basis == "calendar":
        return [timestamp for timestamp in grid if states[timestamp].expected is True]
    if observed.empty:
        return []
    lower = observed.min()
    upper = observed.max()
    return [timestamp for timestamp in grid if lower <= timestamp <= upper]


def _classify_gap_slot(
    timestamp: pd.Timestamp,
    state: MarketState,
    *,
    source: str,
    instrument: str,
    timeframe: str,
    known_gaps: Sequence[KnownGap],
    counts_as_missing: bool,
    is_stale: bool,
) -> _GapSlot:
    annotation = next(
        (
            gap
            for gap in known_gaps
            if gap.matches(
                timestamp,
                source=source,
                instrument=instrument,
                timeframe=timeframe,
            )
        ),
        None,
    )
    if annotation is not None:
        return _GapSlot(
            timestamp=timestamp,
            state=state,
            classification="known_gap",
            reason=annotation.reason,
            annotation_code=annotation.code,
            counts_as_missing=counts_as_missing,
            is_stale=is_stale,
            is_explained=True,
        )
    if state.expected is False:
        return _GapSlot(
            timestamp=timestamp,
            state=state,
            classification="scheduled_closure",
            reason=state.reason,
            annotation_code=None,
            counts_as_missing=False,
            is_stale=False,
            is_explained=True,
        )
    if state.expected is True:
        return _GapSlot(
            timestamp=timestamp,
            state=state,
            classification="missing_data",
            reason="expected candle absent without a matching known-gap annotation",
            annotation_code=None,
            counts_as_missing=counts_as_missing,
            is_stale=is_stale,
            is_explained=False,
        )
    return _GapSlot(
        timestamp=timestamp,
        state=state,
        classification="unknown_market_status",
        reason=state.reason,
        annotation_code=None,
        counts_as_missing=counts_as_missing,
        is_stale=is_stale,
        is_explained=False,
    )


def _collapse_gap_slots(
    slots: Sequence[_GapSlot],
    *,
    day: date,
    source: str,
    instrument: str,
    timeframe: _Timeframe,
) -> list[GapQualityRecord]:
    if not slots:
        return []
    records: list[GapQualityRecord] = []
    current: list[_GapSlot] = [slots[0]]
    for slot in slots[1:]:
        previous = current[-1]
        is_adjacent = timeframe.close_of(previous.timestamp) == slot.timestamp
        if is_adjacent and previous.grouping_key == slot.grouping_key:
            current.append(slot)
            continue
        records.append(
            _gap_record(current, day=day, source=source, instrument=instrument, timeframe=timeframe)
        )
        current = [slot]
    records.append(
        _gap_record(current, day=day, source=source, instrument=instrument, timeframe=timeframe)
    )
    return records


def _gap_record(
    slots: Sequence[_GapSlot],
    *,
    day: date,
    source: str,
    instrument: str,
    timeframe: _Timeframe,
) -> GapQualityRecord:
    first = slots[0]
    last = slots[-1]
    return GapQualityRecord(
        day_utc=day,
        source=source,
        instrument=instrument,
        timeframe=timeframe.code,
        start_utc=first.timestamp,
        end_utc=timeframe.close_of(last.timestamp),
        missing_intervals=len(slots),
        market_status=first.state.status,
        classification=first.classification,
        reason=first.reason,
        annotation_code=first.annotation_code,
        counts_as_missing=first.counts_as_missing,
        is_stale=first.is_stale,
        is_explained=first.is_explained,
    )


def _incomplete_mask(values: pd.Series) -> pd.Series:
    if is_bool_dtype(values.dtype):
        return values.isna() | ~values.fillna(False)
    return values.map(lambda value: not isinstance(value, (bool, np.bool_)) or not bool(value))


def _invalid_ohlc_mask(rows: pd.DataFrame) -> pd.Series:
    invalid = pd.Series(False, index=rows.index)
    numeric: dict[str, pd.Series] = {}
    for column in PRICE_COLUMNS:
        values = rows[column]
        if not is_numeric_dtype(values.dtype) or is_bool_dtype(values.dtype):
            invalid |= values.map(
                lambda value: not isinstance(value, (int, float, np.integer, np.floating))
                or isinstance(value, (bool, np.bool_))
            ).astype(bool)
            coerced = pd.to_numeric(values, errors="coerce")
        else:
            coerced = values.astype(float)
        numeric[column] = coerced
        finite = pd.Series(
            np.isfinite(coerced.to_numpy(dtype=float, na_value=np.nan)),
            index=rows.index,
        )
        invalid |= ~finite | coerced.le(0)
    invalid |= numeric["bid_high"].lt(
        pd.concat([numeric["bid_open"], numeric["bid_close"]], axis=1).max(axis=1)
    )
    invalid |= numeric["bid_low"].gt(
        pd.concat([numeric["bid_open"], numeric["bid_close"]], axis=1).min(axis=1)
    )
    invalid |= numeric["bid_high"].lt(numeric["bid_low"])
    return invalid


def _aggregate_market_status(states: Iterable[MarketState]) -> MarketStatus:
    state_list = list(states)
    if not state_list:
        return "unknown"
    present = {state.status for state in state_list}
    priorities: tuple[MarketStatus, ...] = (
        "open",
        "holiday",
        "maintenance",
        "weekend",
        "unknown",
    )
    for status in priorities:
        if status in present:
            return status
    return "unknown"


def _quality_status(
    *,
    duplicate_count: int,
    incomplete_count: int,
    invalid_ohlc_count: int,
    unexplained_gap_count: int,
    explained_gap_count: int,
    unknown_gap_count: int,
) -> QualityStatus:
    if duplicate_count or incomplete_count or invalid_ohlc_count or unexplained_gap_count:
        return "fail"
    if explained_gap_count or unknown_gap_count:
        return "marked"
    return "pass"


__all__ = [
    "DailyDataQualityRecord",
    "DataQualityError",
    "DataQualityReport",
    "ExpectationBasis",
    "ExplicitMarketCalendarPolicy",
    "GapClassification",
    "GapQualityRecord",
    "KnownGap",
    "MarketCalendarPolicy",
    "MarketState",
    "MarketStatus",
    "ObservedSourceCalendarPolicy",
    "QualityStatus",
    "UtcMarketWindow",
    "WeeklyOpenWindow",
    "build_daily_data_quality",
]
