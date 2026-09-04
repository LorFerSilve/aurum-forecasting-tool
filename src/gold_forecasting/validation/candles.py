"""Fail-closed validation and gap reporting for canonical candle frames.

Validation deliberately does not fill missing rows or repair malformed prices.
The only optional cleanup is an explicit ``drop_identical`` duplicate policy,
which is useful when the exact same immutable import is supplied twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

CANDLE_KEY_COLUMNS = (
    "instrument",
    "timeframe",
    "timestamp_open_utc",
    "source",
)
PRICE_COLUMNS = ("bid_open", "bid_high", "bid_low", "bid_close")
REQUIRED_CANDLE_COLUMNS = (
    "timestamp_open_utc",
    "timestamp_close_utc",
    *PRICE_COLUMNS,
    "instrument",
    "timeframe",
    "is_complete",
    "source",
    "raw_file_hash",
    "ingested_at_utc",
    "dataset_version",
)

DuplicatePolicy = Literal["raise", "drop_identical"]

_TIMEFRAME_MINUTES = {"1min": 1, "3min": 3, "15min": 15}
_DUPLICATE_VALUE_COLUMNS = (
    "timestamp_close_utc",
    *PRICE_COLUMNS,
    "is_complete",
    "raw_file_hash",
    "ingested_at_utc",
    "dataset_version",
)


class CandleValidationError(ValueError):
    """Raised when candles violate the canonical research-data contract."""


@dataclass(frozen=True, slots=True)
class CandleGap:
    """One missing contiguous interval between two observed candles."""

    instrument: str
    timeframe: str
    source: str
    previous_open_utc: pd.Timestamp
    expected_next_open_utc: pd.Timestamp
    actual_next_open_utc: pd.Timestamp
    missing_candles: int

    @property
    def duration(self) -> pd.Timedelta:
        """Elapsed time with no observed candle opens."""

        return self.actual_next_open_utc - self.expected_next_open_utc

    def as_record(self) -> dict[str, object]:
        """Return a JSON/CSV-friendly representation."""

        return {
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "source": self.source,
            "previous_open_utc": self.previous_open_utc,
            "expected_next_open_utc": self.expected_next_open_utc,
            "actual_next_open_utc": self.actual_next_open_utc,
            "missing_candles": self.missing_candles,
            "gap_duration_seconds": int(self.duration.total_seconds()),
        }


@dataclass(frozen=True, slots=True)
class GapReport:
    """Deterministic collection of observed candle gaps."""

    gaps: tuple[CandleGap, ...] = ()

    @property
    def gap_count(self) -> int:
        return len(self.gaps)

    @property
    def missing_candles(self) -> int:
        return sum(gap.missing_candles for gap in self.gaps)

    @property
    def has_gaps(self) -> bool:
        return bool(self.gaps)

    def to_frame(self) -> pd.DataFrame:
        """Return one row per gap without altering or interpolating candles."""

        columns = (
            "instrument",
            "timeframe",
            "source",
            "previous_open_utc",
            "expected_next_open_utc",
            "actual_next_open_utc",
            "missing_candles",
            "gap_duration_seconds",
        )
        return pd.DataFrame.from_records(
            (gap.as_record() for gap in self.gaps),
            columns=columns,
        )


@dataclass(frozen=True, slots=True)
class CandleValidationResult:
    """Validated candles plus non-destructive quality metadata."""

    candles: pd.DataFrame
    gaps: GapReport
    duplicates_removed: int = 0

    @property
    def gap_report(self) -> GapReport:
        """Readable alias for callers that prefer the full report name."""

        return self.gaps


def _raise(message: str) -> None:
    raise CandleValidationError(message)


def _require_dataframe(candles: pd.DataFrame) -> None:
    if not isinstance(candles, pd.DataFrame):
        _raise("candles must be a pandas DataFrame")


def _require_columns(candles: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_CANDLE_COLUMNS if column not in candles.columns]
    if missing:
        _raise(f"missing required candle columns: {', '.join(missing)}")


def _normalize_utc_column(candles: pd.DataFrame, column: str) -> None:
    values = candles[column]
    if values.isna().any():
        _raise(f"{column} must not contain missing timestamps")

    if isinstance(values.dtype, pd.DatetimeTZDtype):
        candles[column] = values.dt.tz_convert("UTC")
        return
    if pd.api.types.is_datetime64_dtype(values.dtype):
        _raise(f"{column} must contain timezone-aware timestamps")

    normalized: list[pd.Timestamp] = []
    for row_index, value in values.items():
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise CandleValidationError(
                f"{column} contains an invalid timestamp at index {row_index!r}: {value!r}"
            ) from exc
        if pd.isna(timestamp):
            _raise(f"{column} contains an invalid timestamp at index {row_index!r}")
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            _raise(f"{column} must contain timezone-aware timestamps")
        normalized.append(timestamp.tz_convert("UTC"))
    candles[column] = pd.DatetimeIndex(normalized, dtype="datetime64[ns, UTC]")


def _validate_text_columns(candles: pd.DataFrame) -> None:
    for column in (
        "instrument",
        "timeframe",
        "source",
        "raw_file_hash",
        "dataset_version",
    ):
        invalid = candles[column].map(
            lambda value: not isinstance(value, str) or not value.strip()
        )
        if invalid.any():
            indices = candles.index[invalid].tolist()[:5]
            _raise(f"{column} must contain non-empty strings; invalid rows: {indices}")

    unsupported = sorted(set(candles["timeframe"]) - _TIMEFRAME_MINUTES.keys())
    if unsupported:
        _raise(f"unsupported timeframe values: {unsupported}")


def _validate_completeness(candles: pd.DataFrame, *, require_complete: bool) -> None:
    complete = candles["is_complete"]
    if not is_bool_dtype(complete.dtype) or complete.isna().any():
        _raise("is_complete must contain non-missing boolean values")
    if require_complete and not complete.all():
        indices = candles.index[~complete].tolist()[:5]
        _raise(f"incomplete candles are not valid model input; rows: {indices}")


def _validate_prices(candles: pd.DataFrame) -> None:
    for column in PRICE_COLUMNS:
        values = candles[column]
        if not is_numeric_dtype(values.dtype) or is_bool_dtype(values.dtype):
            _raise(f"{column} must be numeric")
        finite = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        if not finite.all():
            indices = candles.index[~finite].tolist()[:5]
            _raise(f"{column} must contain only finite prices; invalid rows: {indices}")
        positive = values > 0
        if not positive.all():
            indices = candles.index[~positive].tolist()[:5]
            _raise(f"{column} must contain strictly positive prices; invalid rows: {indices}")

    high_valid = candles["bid_high"] >= candles[["bid_open", "bid_close"]].max(axis=1)
    if not high_valid.all():
        indices = candles.index[~high_valid].tolist()[:5]
        _raise(f"bid_high must be >= max(bid_open, bid_close); invalid rows: {indices}")

    low_valid = candles["bid_low"] <= candles[["bid_open", "bid_close"]].min(axis=1)
    if not low_valid.all():
        indices = candles.index[~low_valid].tolist()[:5]
        _raise(f"bid_low must be <= min(bid_open, bid_close); invalid rows: {indices}")

    range_valid = candles["bid_high"] >= candles["bid_low"]
    if not range_valid.all():
        indices = candles.index[~range_valid].tolist()[:5]
        _raise(f"bid_high must be >= bid_low; invalid rows: {indices}")


def _validate_timestamps(candles: pd.DataFrame, *, check_order: bool) -> None:
    durations = candles["timeframe"].map(_TIMEFRAME_MINUTES).map(
        lambda minutes: pd.Timedelta(minutes=int(minutes))
    )
    expected_closes = candles["timestamp_open_utc"] + durations
    closes_match = candles["timestamp_close_utc"] == expected_closes
    if not closes_match.all():
        indices = candles.index[~closes_match].tolist()[:5]
        _raise(
            "timestamp_close_utc must equal timestamp_open_utc plus the timeframe; "
            f"invalid rows: {indices}"
        )

    for timeframe, minutes in _TIMEFRAME_MINUTES.items():
        timeframe_rows = candles["timeframe"] == timeframe
        if not timeframe_rows.any():
            continue
        opens = candles.loc[timeframe_rows, "timestamp_open_utc"]
        aligned = opens == opens.dt.floor(f"{minutes}min")
        if not aligned.all():
            indices = opens.index[~aligned].tolist()[:5]
            _raise(f"{timeframe} candle opens must align to the UTC grid; invalid rows: {indices}")

    if check_order:
        grouping = ["instrument", "timeframe", "source"]
        for group, group_rows in candles.groupby(grouping, sort=False, dropna=False):
            if not group_rows["timestamp_open_utc"].is_monotonic_increasing:
                _raise(
                    "timestamp_open_utc must be strictly increasing within each "
                    f"instrument/timeframe/source group; invalid group: {group}"
                )


def _deduplicate(candles: pd.DataFrame, policy: DuplicatePolicy) -> tuple[pd.DataFrame, int]:
    if policy not in ("raise", "drop_identical"):
        _raise(f"unsupported duplicate policy: {policy!r}")

    duplicate_rows = candles.duplicated(subset=list(CANDLE_KEY_COLUMNS), keep=False)
    if not duplicate_rows.any():
        return candles, 0

    duplicate_count = int(duplicate_rows.sum())
    if policy == "raise":
        _raise(
            f"found {duplicate_count} rows with duplicate candle keys; "
            "use duplicate_policy='drop_identical' only for identical repeats"
        )

    duplicated = candles.loc[duplicate_rows]
    for key, rows in duplicated.groupby(list(CANDLE_KEY_COLUMNS), sort=False, dropna=False):
        for column in _DUPLICATE_VALUE_COLUMNS:
            if rows[column].nunique(dropna=False) != 1:
                _raise(f"conflicting duplicate candles for key {key}; column {column} differs")

    deduplicated = candles.drop_duplicates(subset=list(CANDLE_KEY_COLUMNS), keep="first")
    return deduplicated, len(candles) - len(deduplicated)


def _prepare_candles(
    candles: pd.DataFrame,
    *,
    duplicate_policy: DuplicatePolicy,
    require_complete: bool,
    check_order: bool,
) -> tuple[pd.DataFrame, int]:
    _require_dataframe(candles)
    _require_columns(candles)
    prepared = candles.copy(deep=True)

    _normalize_utc_column(prepared, "timestamp_open_utc")
    _normalize_utc_column(prepared, "timestamp_close_utc")
    _normalize_utc_column(prepared, "ingested_at_utc")
    _validate_text_columns(prepared)
    _validate_completeness(prepared, require_complete=require_complete)
    _validate_prices(prepared)
    prepared, duplicates_removed = _deduplicate(prepared, duplicate_policy)
    _validate_timestamps(prepared, check_order=check_order)
    return prepared, duplicates_removed


def _detect_gaps_in_prepared(candles: pd.DataFrame) -> GapReport:
    grouping = ["instrument", "timeframe", "source"]
    ordered = candles.sort_values([*grouping, "timestamp_open_utc"], kind="stable").copy()
    previous = ordered.groupby(grouping, sort=False, dropna=False)[
        "timestamp_open_utc"
    ].shift(1)
    cadence = pd.to_timedelta(
        ordered["timeframe"].map(_TIMEFRAME_MINUTES).astype("int64"),
        unit="min",
    )
    expected = previous + cadence
    is_gap = previous.notna() & ordered["timestamp_open_utc"].gt(expected)
    if not is_gap.any():
        return GapReport()

    gap_rows = ordered.loc[is_gap, [*grouping, "timestamp_open_utc"]].copy()
    gap_rows["previous_open_utc"] = previous.loc[is_gap]
    gap_rows["expected_next_open_utc"] = expected.loc[is_gap]
    gap_rows["missing_candles"] = (
        (gap_rows["timestamp_open_utc"] - gap_rows["previous_open_utc"])
        // cadence.loc[is_gap]
        - 1
    ).astype("int64")

    found = tuple(
        CandleGap(
            instrument=str(row.instrument),
            timeframe=str(row.timeframe),
            source=str(row.source),
            previous_open_utc=cast(pd.Timestamp, row.previous_open_utc),
            expected_next_open_utc=cast(pd.Timestamp, row.expected_next_open_utc),
            actual_next_open_utc=cast(pd.Timestamp, row.timestamp_open_utc),
            missing_candles=cast(int, row.missing_candles),
        )
        for row in gap_rows.itertuples(index=False)
    )
    return GapReport(found)


def detect_gaps(candles: pd.DataFrame) -> GapReport:
    """Validate and report missing intervals without synthesizing any rows."""

    prepared, _ = _prepare_candles(
        candles,
        duplicate_policy="raise",
        require_complete=True,
        check_order=True,
    )
    return _detect_gaps_in_prepared(prepared)


def validate_candles(
    candles: pd.DataFrame,
    *,
    expected_timeframe: str | None = None,
    duplicate_policy: DuplicatePolicy = "raise",
) -> CandleValidationResult:
    """Validate canonical candles and return an unmodified-gap quality result.

    The returned frame is a defensive copy. Timezone-aware timestamp inputs are
    normalized to UTC, while naive timestamps are rejected because their source
    timezone cannot be inferred safely.
    """

    prepared, duplicates_removed = _prepare_candles(
        candles,
        duplicate_policy=duplicate_policy,
        require_complete=True,
        check_order=True,
    )
    if expected_timeframe is not None:
        if expected_timeframe not in _TIMEFRAME_MINUTES:
            _raise(f"unsupported expected timeframe: {expected_timeframe!r}")
        observed = set(prepared["timeframe"])
        if observed != {expected_timeframe}:
            _raise(
                f"expected only {expected_timeframe} candles, observed: {sorted(observed)}"
            )

    gaps = _detect_gaps_in_prepared(prepared)
    return CandleValidationResult(
        candles=prepared.reset_index(drop=True),
        gaps=gaps,
        duplicates_removed=duplicates_removed,
    )


__all__ = [
    "CANDLE_KEY_COLUMNS",
    "PRICE_COLUMNS",
    "REQUIRED_CANDLE_COLUMNS",
    "CandleGap",
    "CandleValidationError",
    "CandleValidationResult",
    "DuplicatePolicy",
    "GapReport",
    "detect_gaps",
    "validate_candles",
]
