"""Provider-neutral causal price/context calculations shared by Phase 10 sources."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series[float]:
    column = frame[name]
    if (
        not is_numeric_dtype(column.dtype)
        or is_bool_dtype(column.dtype)
        or np.iscomplexobj(column.to_numpy())
    ):
        raise ValueError(f"{name} must be numeric")
    return pd.Series(
        column.to_numpy(dtype=np.float64, na_value=np.nan), index=frame.index, name=name
    )


def _validate_frame(frame: pd.DataFrame, source_id: str) -> None:
    if frame.columns.has_duplicates:
        raise ValueError("Feature input must not have duplicate columns")
    required = {
        "prediction_time_utc",
        "gold_close",
        f"{source_id}_value",
        f"{source_id}_is_missing",
        f"{source_id}_is_stale",
        f"{source_id}_age_seconds",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Feature input is missing columns: {sorted(missing)}")
    times = frame["prediction_time_utc"]
    if not isinstance(times.dtype, pd.DatetimeTZDtype) or str(times.dtype.tz) != "UTC":
        raise ValueError("prediction_time_utc must have a timezone-aware UTC datetime dtype")
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("prediction_time_utc must be nonmissing, sorted and unique")
    for name in (f"{source_id}_is_missing", f"{source_id}_is_stale"):
        if not is_bool_dtype(frame[name].dtype) or frame[name].isna().any():
            raise ValueError(f"{name} must contain nonmissing booleans")
    for name in ("gold_close", f"{source_id}_value"):
        prices = _numeric_column(frame, name)
        observed = prices.dropna()
        if not np.isfinite(observed).all() or observed.le(0).any():
            raise ValueError(f"{name} must contain positive finite prices or permitted NaN")
        if name == "gold_close" and prices.isna().any():
            raise ValueError("gold_close must not contain missing prices")
    ages = _numeric_column(frame, f"{source_id}_age_seconds")
    observed_ages = ages.dropna()
    if not np.isfinite(observed_ages).all() or observed_ages.lt(0).any():
        raise ValueError(f"{source_id}_age_seconds must be finite and nonnegative when present")
    if (ages.isna() & ~frame[f"{source_id}_is_missing"]).any():
        raise ValueError(
            f"{source_id}_age_seconds may be missing only for missing {source_id} observations"
        )


def paired_price_features(
    frame: pd.DataFrame,
    *,
    source_id: str,
    cadence_seconds: int = 180,
    momentum_steps: int = 5,
    correlation_steps: int = 20,
) -> dict[str, pd.Series[float]]:
    """Compute causal gold/raw-context features without mutating the input.

    The source-specific wrappers name and orient these common calculations.
    Missing/stale observations and gaps restart the original Phase-10 windows;
    a repeated as-of level is never treated as a new observation.
    """
    for name, value in (
        ("cadence_seconds", cadence_seconds),
        ("momentum_steps", momentum_steps),
        ("correlation_steps", correlation_steps),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if momentum_steps != 5 or correlation_steps != 20:
        raise ValueError(
            "The Phase 10 feature contract requires momentum_steps=5/correlation_steps=20"
        )
    _validate_frame(frame, source_id)
    output: dict[str, pd.Series[float]] = {}
    consecutive = frame["prediction_time_utc"].diff().eq(pd.Timedelta(seconds=cadence_seconds))
    gold_log = pd.Series(np.log(_numeric_column(frame, "gold_close").to_numpy()), index=frame.index)
    context = _numeric_column(frame, f"{source_id}_value").where(
        ~frame[f"{source_id}_is_missing"] & ~frame[f"{source_id}_is_stale"]
    )
    context_log = pd.Series(np.log(context.to_numpy()), index=frame.index)
    observed_time = frame["prediction_time_utc"] - pd.to_timedelta(
        frame[f"{source_id}_age_seconds"], unit="s"
    )
    context_consecutive = consecutive & observed_time.diff().eq(
        pd.Timedelta(seconds=cadence_seconds)
    )
    momentum_contiguous = (
        consecutive.rolling(momentum_steps, min_periods=momentum_steps).sum().eq(momentum_steps)
    )
    context_momentum_available = (
        context.notna()
        .rolling(momentum_steps + 1, min_periods=momentum_steps + 1)
        .sum()
        .eq(momentum_steps + 1)
    )
    context_momentum_contiguous = (
        context_consecutive.rolling(momentum_steps, min_periods=momentum_steps)
        .sum().eq(momentum_steps)
    )
    gold_return = gold_log.diff().mul(10_000.0).where(consecutive)
    # Reusing an as-of level is not a new candle and must not invent a zero return.
    context_return = context_log.diff().mul(10_000.0).where(context_consecutive)
    output["gold_return_1_bps"] = gold_return
    output["gold_momentum_5_bps"] = (
        gold_log.diff(momentum_steps).mul(10_000.0).where(momentum_contiguous)
    )
    output["context_return_1_bps"] = context_return
    output["context_momentum_5_bps"] = (
        context_log.diff(momentum_steps)
        .mul(10_000.0)
        .where(context_momentum_contiguous & context_momentum_available)
    )
    output["gold_context_log_ratio"] = gold_log - context_log
    correlation = gold_return.rolling(correlation_steps, min_periods=correlation_steps).corr(
        context_return
    )
    # Constant paired returns have undefined correlation; pandas can produce inf.
    output["gold_context_correlation_20"] = correlation.where(np.isfinite(correlation)).clip(-1, 1)
    return output
