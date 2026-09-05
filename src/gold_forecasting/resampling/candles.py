"""Build complete, UTC-aligned candles from canonical one-minute data."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime

import pandas as pd

from gold_forecasting.validation.candles import (
    REQUIRED_CANDLE_COLUMNS,
    CandleValidationError,
    _prepare_candles,
    validate_candles,
)
from gold_forecasting.validation.quality import MarketCalendarPolicy

# This version belongs to the resampling algorithm, not to a data snapshot or
# provider adapter.  Manifests can therefore pin it independently when phase 5
# persistence is wired into the pipeline.
RESAMPLING_LOGIC_VERSION = "2.0.0"

# ``None`` denotes the one calendar-aware target.  Keeping the accepted public
# codes explicit prevents pandas aliases (notably ``M``/``ME``) from leaking
# into the project's stable timeframe vocabulary.
_TARGET_MINUTES: dict[str, int | None] = {
    "3min": 3,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1h": 60,
    "3h": 180,
    "1d": 1_440,
    "1mo": None,
}
SUPPORTED_TARGET_TIMEFRAMES = tuple(_TARGET_MINUTES)


class ResamplingError(ValueError):
    """Raised when input cannot be resampled under the candle contract."""


def _lineage_hash(values: pd.Series) -> str:
    unique = sorted(set(values.astype(str)))
    if len(unique) == 1:
        return unique[0]
    digest = hashlib.sha256("\n".join(unique).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _window_bounds(
    opens: pd.Series,
    target_timeframe: str,
) -> tuple[pd.Series, pd.Series]:
    """Return UTC window opens and exclusive closes for every source row."""

    target_minutes = _TARGET_MINUTES[target_timeframe]
    if target_minutes is not None:
        window_open = opens.dt.floor(f"{target_minutes}min")
        return window_open, window_open + pd.Timedelta(minutes=target_minutes)

    # A month is a UTC calendar interval [month-start, next-month-start), not a
    # fixed timedelta. Construct both boundaries explicitly so February,
    # leap-years, and year transitions all have their true lengths.
    years = opens.dt.year
    months = opens.dt.month
    window_open = pd.to_datetime(
        {"year": years, "month": months, "day": 1},
        utc=True,
    )
    next_months = months.mod(12).add(1)
    next_years = years.add(months.eq(12).astype(int))
    window_close = pd.to_datetime(
        {"year": next_years, "month": next_months, "day": 1},
        utc=True,
    )
    return window_open, window_close


def _normalize_cutoff(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    """Return one timezone-aware UTC cutoff or fail with a domain error."""

    try:
        cutoff = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ResamplingError(f"closed_through_utc is not a valid timestamp: {value!r}") from exc
    if pd.isna(cutoff) or cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ResamplingError("closed_through_utc must be a timezone-aware timestamp")
    return cutoff.tz_convert("UTC")


def _prepare_source(candles: pd.DataFrame) -> pd.DataFrame:
    """Validate once and establish stable ordering for any number of targets."""
    if isinstance(candles, pd.DataFrame) and "timeframe" in candles.columns:
        source_timeframes = set(candles["timeframe"])
        if source_timeframes and source_timeframes != {"1min"}:
            raise ResamplingError(
                "source candles must all use timeframe '1min'; "
                f"observed {source_timeframes}"
            )

    try:
        prepared, _ = _prepare_candles(
            candles,
            duplicate_policy="raise",
            require_complete=False,
            check_order=False,
        )
    except CandleValidationError as exc:
        raise ResamplingError(f"invalid source candles: {exc}") from exc

    observed = set(prepared["timeframe"])
    if observed and observed != {"1min"}:
        raise ResamplingError(f"source candles must all use timeframe '1min'; observed {observed}")
    return prepared.sort_values(
        ["instrument", "source", "timestamp_open_utc"], kind="stable"
    ).reset_index(drop=True)


def _resample_prepared(
    prepared: pd.DataFrame,
    target_timeframe: str,
    *,
    require_dense: bool | None,
    closed_through_utc: str | datetime | pd.Timestamp | None,
    calendar_policy: MarketCalendarPolicy | None,
) -> pd.DataFrame:
    if target_timeframe not in _TARGET_MINUTES:
        raise ResamplingError(
            f"target_timeframe must be one of {sorted(_TARGET_MINUTES)}, "
            f"got {target_timeframe!r}"
        )
    if require_dense is not None and not isinstance(require_dense, bool):
        raise ResamplingError("require_dense must be a boolean or None")
    if calendar_policy is not None and not isinstance(calendar_policy, MarketCalendarPolicy):
        raise ResamplingError("calendar_policy must implement MarketCalendarPolicy")
    dense = calendar_policy is None if require_dense is None else require_dense
    if not dense and calendar_policy is None:
        raise ResamplingError("require_dense=False requires an explicit calendar_policy")
    explicit_cutoff = (
        None if closed_through_utc is None else _normalize_cutoff(closed_through_utc)
    )
    output_columns = list(REQUIRED_CANDLE_COLUMNS)
    if prepared.empty:
        return prepared.loc[:, output_columns].copy()

    cadence = pd.Timedelta(minutes=1)
    grouping = ["instrument", "source"]
    ordered = prepared.copy()
    window_open, window_close = _window_bounds(
        ordered["timestamp_open_utc"],
        target_timeframe,
    )
    ordered["_window_open_utc"] = window_open
    ordered["_window_close_utc"] = window_close
    if explicit_cutoff is None:
        ordered["_closed_through_utc"] = ordered.groupby(
            grouping,
            sort=False,
            dropna=False,
        )["ingested_at_utc"].transform("max")
    else:
        ordered["_closed_through_utc"] = explicit_cutoff

    group_keys = [*grouping, "_window_open_utc"]
    grouped = ordered.groupby(group_keys, sort=True, dropna=False)
    result = grouped.agg(
        timestamp_close_utc=("timestamp_close_utc", "last"),
        bid_open=("bid_open", "first"),
        bid_high=("bid_high", "max"),
        bid_low=("bid_low", "min"),
        bid_close=("bid_close", "last"),
        is_complete=("is_complete", "all"),
        raw_file_hash=("raw_file_hash", "first"),
        ingested_at_utc=("ingested_at_utc", "max"),
        dataset_version=("dataset_version", "first"),
        _row_count=("timestamp_open_utc", "size"),
        _first_open_utc=("timestamp_open_utc", "first"),
        _last_open_utc=("timestamp_open_utc", "last"),
        _window_close_utc=("_window_close_utc", "first"),
        _closed_through_utc=("_closed_through_utc", "first"),
        _raw_versions=("raw_file_hash", "nunique"),
        _dataset_versions=("dataset_version", "nunique"),
    ).reset_index()

    # A UTC-aligned bucket has exactly one source row per elapsed minute. With
    # duplicate keys rejected above, count + first/last proves that no internal
    # minute is missing without interpolation or a Python loop per bucket.
    expected_count = (
        (result["_window_close_utc"] - result["_window_open_utc"])
        .dt.total_seconds()
        .floordiv(cadence.total_seconds())
        .astype("int64")
    )
    expected_last = result["_window_close_utc"] - cadence
    complete_window = result["is_complete"] & result["_window_close_utc"].le(
        result["_closed_through_utc"]
    )
    dense_window = (
        result["_row_count"].eq(expected_count)
        & result["_first_open_utc"].eq(result["_window_open_utc"])
        & result["_last_open_utc"].eq(expected_last)
    )
    if not dense:
        assert calendar_policy is not None
        # An elapsed month is not necessarily a fully covered month. Every
        # absent minute, including leading/trailing fragments, needs an explicit
        # closed-market classification. Unknown calendar status cannot certify
        # a candle, even if every minute that happened to arrive is complete.
        for index in result.index[complete_window & ~dense_window]:
            key = tuple(result.loc[index, group_keys])
            window = result.loc[index]
            expected_opens = pd.date_range(
                window["_window_open_utc"],
                window["_window_close_utc"],
                freq="1min",
                inclusive="left",
            )
            observed_opens = pd.DatetimeIndex(grouped.get_group(key)["timestamp_open_utc"])
            missing_opens = expected_opens.difference(observed_opens)
            dense_window.loc[index] = all(
                calendar_policy.classify(
                    timestamp,
                    source=str(window["source"]),
                    instrument=str(window["instrument"]),
                    timeframe="1min",
                ).expected is False
                for timestamp in missing_opens
            )
    complete_window &= dense_window
    result = result.loc[complete_window].copy()

    # Mixed lineage is unusual (normally one immutable source archive per
    # partition), but preserve it exactly for provider-neutral concatenations.
    for index in result.index[result["_raw_versions"].gt(1)]:
        key = tuple(result.loc[index, group_keys])
        result.loc[index, "raw_file_hash"] = _lineage_hash(grouped.get_group(key)["raw_file_hash"])
    for index in result.index[result["_dataset_versions"].gt(1)]:
        key = tuple(result.loc[index, group_keys])
        result.loc[index, "dataset_version"] = _lineage_hash(
            grouped.get_group(key)["dataset_version"]
        )

    result = result.rename(columns={"_window_open_utc": "timestamp_open_utc"})
    result["timestamp_close_utc"] = result["_window_close_utc"]
    result["timeframe"] = target_timeframe
    result = result.loc[:, output_columns]
    if result.empty:
        return result

    result = result.sort_values(
        ["instrument", "source", "timestamp_open_utc"],
        kind="stable",
    ).reset_index(drop=True)
    try:
        return validate_candles(result, expected_timeframe=target_timeframe).candles
    except CandleValidationError as exc:  # pragma: no cover - aggregation guard
        raise ResamplingError(f"resampling produced invalid candles: {exc}") from exc


def resample_candles(
    candles: pd.DataFrame,
    target_timeframe: str,
    *,
    require_dense: bool | None = None,
    closed_through_utc: str | datetime | pd.Timestamp | None = None,
    calendar_policy: MarketCalendarPolicy | None = None,
) -> pd.DataFrame:
    """Aggregate complete one-minute inputs on fixed or calendar-aware UTC grids.

    Every minute in the target window is required by default, including for
    days and months. With an explicit calendar policy, missing minutes are
    allowed only when that policy certifies them as scheduled closures. A
    source-observed/unknown calendar cannot certify missing coverage. Setting
    ``require_dense=True`` requires all minutes even when a calendar is given.
    An incomplete source row always disqualifies its containing window.

    A window closes at or before ``closed_through_utc``. If omitted, the latest
    ingestion timestamp per instrument/source is the deterministic cutoff.
    Prices are never interpolated. An empty result retains the canonical schema.
    """

    return _resample_prepared(
        _prepare_source(candles),
        target_timeframe,
        require_dense=require_dense,
        closed_through_utc=closed_through_utc,
        calendar_policy=calendar_policy,
    )


def resample_timeframes(
    candles: pd.DataFrame,
    timeframes: Sequence[str] = SUPPORTED_TARGET_TIMEFRAMES,
    *,
    require_dense: bool | None = None,
    closed_through_utc: str | datetime | pd.Timestamp | None = None,
    calendar_policy: MarketCalendarPolicy | None = None,
) -> Mapping[str, pd.DataFrame]:
    """Build selected derived targets with one source validation and ordering pass."""

    if isinstance(timeframes, str) or len(set(timeframes)) != len(timeframes):
        raise ResamplingError("timeframes must be a sequence of unique timeframe codes")
    prepared = _prepare_source(candles)
    return {
        timeframe: _resample_prepared(
            prepared,
            timeframe,
            require_dense=require_dense,
            closed_through_utc=closed_through_utc,
            calendar_policy=calendar_policy,
        )
        for timeframe in timeframes
    }


def resample_mvp_timeframes(candles: pd.DataFrame) -> Mapping[str, pd.DataFrame]:
    """Build both MVP timeframes directly from the same canonical 1-minute frame."""

    return resample_timeframes(candles, ("3min", "15min"))


__all__ = [
    "RESAMPLING_LOGIC_VERSION",
    "SUPPORTED_TARGET_TIMEFRAMES",
    "ResamplingError",
    "resample_candles",
    "resample_mvp_timeframes",
    "resample_timeframes",
]
