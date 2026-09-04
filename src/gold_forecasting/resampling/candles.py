"""Build complete 3-minute and 15-minute candles from canonical 1-minute data."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import pandas as pd

from gold_forecasting.validation.candles import (
    REQUIRED_CANDLE_COLUMNS,
    CandleValidationError,
    _prepare_candles,
    validate_candles,
)

_TARGET_MINUTES = {"3min": 3, "15min": 15}


class ResamplingError(ValueError):
    """Raised when input cannot be resampled under the MVP contract."""


def _lineage_hash(values: pd.Series) -> str:
    unique = sorted(set(values.astype(str)))
    if len(unique) == 1:
        return unique[0]
    digest = hashlib.sha256("\n".join(unique).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def resample_candles(candles: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    """Aggregate exact, complete UTC windows of 1-minute bid candles.

    Missing, duplicated, misaligned, or incomplete subcandles never become a
    higher-timeframe candle. No interpolation or forward filling is performed.
    Output rows and columns have a stable order, independent of input order.
    """

    if target_timeframe not in _TARGET_MINUTES:
        raise ResamplingError(
            f"target_timeframe must be one of {sorted(_TARGET_MINUTES)}, "
            f"got {target_timeframe!r}"
        )

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

    output_columns = list(REQUIRED_CANDLE_COLUMNS)
    if prepared.empty:
        return pd.DataFrame(columns=output_columns)

    target_minutes = _TARGET_MINUTES[target_timeframe]
    cadence = pd.Timedelta(minutes=1)
    target_duration = pd.Timedelta(minutes=target_minutes)
    grouping = ["instrument", "source"]
    ordered = prepared.sort_values([*grouping, "timestamp_open_utc"], kind="stable").copy()
    ordered["_window_open_utc"] = ordered["timestamp_open_utc"].dt.floor(target_timeframe)

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
        _raw_versions=("raw_file_hash", "nunique"),
        _dataset_versions=("dataset_version", "nunique"),
    ).reset_index()

    # A UTC-aligned one-minute bucket has exactly ``target_minutes`` possible
    # opens. With duplicate keys rejected above, count + first/last therefore
    # proves that no internal minute is missing without a Python loop per
    # bucket. This keeps a five-year rebuild practical while remaining strict.
    expected_last = result["_window_open_utc"] + target_duration - cadence
    complete_window = (
        result["_row_count"].eq(target_minutes)
        & result["is_complete"]
        & result["_first_open_utc"].eq(result["_window_open_utc"])
        & result["_last_open_utc"].eq(expected_last)
    )
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
    result["timestamp_close_utc"] = result["timestamp_open_utc"] + target_duration
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
    except CandleValidationError as exc:  # pragma: no cover - guards aggregation regressions
        raise ResamplingError(f"resampling produced invalid candles: {exc}") from exc


def resample_mvp_timeframes(candles: pd.DataFrame) -> Mapping[str, pd.DataFrame]:
    """Build both MVP timeframes directly from the same canonical 1-minute frame."""

    return {
        timeframe: resample_candles(candles, timeframe)
        for timeframe in ("3min", "15min")
    }


__all__ = ["ResamplingError", "resample_candles", "resample_mvp_timeframes"]
