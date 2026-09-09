"""Causal, gap-aware price and silver features for the Phase 10 ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

PRICE_FEATURE_NAMES = ("gold_return_1_bps", "gold_momentum_5_bps")
SILVER_FEATURE_NAMES = (
    "silver_return_1_bps",
    "silver_momentum_5_bps",
    "gold_silver_log_ratio",
    "gold_silver_correlation_20",
    "silver_is_missing",
    "silver_is_stale",
    "silver_age_seconds",
)


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


def _validate_frame(frame: pd.DataFrame) -> None:
    if frame.columns.has_duplicates:
        raise ValueError("Feature input must not have duplicate columns")
    required = {
        "prediction_time_utc",
        "gold_close",
        "silver_value",
        "silver_is_missing",
        "silver_is_stale",
        "silver_age_seconds",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Feature input is missing columns: {sorted(missing)}")
    times = frame["prediction_time_utc"]
    if not isinstance(times.dtype, pd.DatetimeTZDtype) or str(times.dtype.tz) != "UTC":
        raise ValueError("prediction_time_utc must have a timezone-aware UTC datetime dtype")
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("prediction_time_utc must be nonmissing, sorted and unique")
    for name in ("silver_is_missing", "silver_is_stale"):
        if not is_bool_dtype(frame[name].dtype) or frame[name].isna().any():
            raise ValueError(f"{name} must contain nonmissing booleans")
    for name in ("gold_close", "silver_value"):
        prices = _numeric_column(frame, name)
        observed = prices.dropna()
        if not np.isfinite(observed).all() or observed.le(0).any():
            raise ValueError(f"{name} must contain positive finite prices or permitted NaN")
        if name == "gold_close" and prices.isna().any():
            raise ValueError("gold_close must not contain missing prices")
    ages = _numeric_column(frame, "silver_age_seconds")
    observed_ages = ages.dropna()
    if not np.isfinite(observed_ages).all() or observed_ages.lt(0).any():
        raise ValueError("silver_age_seconds must be finite and nonnegative when present")
    if (ages.isna() & ~frame["silver_is_missing"]).any():
        raise ValueError("silver_age_seconds may be missing only for missing silver observations")


def build_silver_features(
    frame: pd.DataFrame,
    *,
    cadence_seconds: int = 180,
    momentum_steps: int = 5,
    correlation_steps: int = 20,
) -> pd.DataFrame:
    """Append features while retaining every input row, index and audit column.

    Input prices must already be point-in-time joined to each prediction cutoff.
    Price changes are log returns in basis points, consistent with Phase 7.
    The fixed feature names require five-step momentum and twenty-step Pearson
    correlation. A momentum needs six consecutive observations; correlation
    needs twenty paired one-step returns, hence twenty-one observations. Any
    cadence gap restarts history. Missing or stale silver never contributes to
    derived silver features, and no value is filled or interpolated.
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
    _validate_frame(frame)
    output = frame.copy(deep=True)
    consecutive = frame["prediction_time_utc"].diff().eq(pd.Timedelta(seconds=cadence_seconds))
    gold_log = pd.Series(np.log(_numeric_column(frame, "gold_close").to_numpy()), index=frame.index)
    silver = _numeric_column(frame, "silver_value").where(
        ~frame["silver_is_missing"] & ~frame["silver_is_stale"]
    )
    silver_log = pd.Series(np.log(silver.to_numpy()), index=frame.index)
    observed_time = frame["prediction_time_utc"] - pd.to_timedelta(
        frame["silver_age_seconds"], unit="s"
    )
    silver_consecutive = consecutive & observed_time.diff().eq(
        pd.Timedelta(seconds=cadence_seconds)
    )
    momentum_contiguous = (
        consecutive.rolling(momentum_steps, min_periods=momentum_steps).sum().eq(momentum_steps)
    )
    silver_momentum_available = (
        silver.notna()
        .rolling(momentum_steps + 1, min_periods=momentum_steps + 1)
        .sum()
        .eq(momentum_steps + 1)
    )
    silver_momentum_contiguous = (
        silver_consecutive.rolling(momentum_steps, min_periods=momentum_steps)
        .sum().eq(momentum_steps)
    )
    gold_return = gold_log.diff().mul(10_000.0).where(consecutive)
    # Reusing an as-of level is not a new candle and must not invent a zero return.
    silver_return = silver_log.diff().mul(10_000.0).where(silver_consecutive)
    output["gold_return_1_bps"] = gold_return
    output["gold_momentum_5_bps"] = (
        gold_log.diff(momentum_steps).mul(10_000.0).where(momentum_contiguous)
    )
    output["silver_return_1_bps"] = silver_return
    output["silver_momentum_5_bps"] = (
        silver_log.diff(momentum_steps)
        .mul(10_000.0)
        .where(silver_momentum_contiguous & silver_momentum_available)
    )
    output["gold_silver_log_ratio"] = gold_log - silver_log
    correlation = gold_return.rolling(correlation_steps, min_periods=correlation_steps).corr(
        silver_return
    )
    # Constant paired returns have undefined correlation; pandas can produce inf.
    output["gold_silver_correlation_20"] = correlation.where(np.isfinite(correlation)).clip(-1, 1)
    return output
