"""Causal daily real-yield features for the Phase-10 rate-source ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

RATE_FEATURE_NAMES = (
    "rate_level_pct",
    "rate_change_1obs_bps",
    "rate_change_5obs_bps",
    "rate_change_20obs_bps",
    "rate_is_missing",
    "rate_is_stale",
    "rate_age_seconds",
)
RATE_MODEL_FEATURES = tuple(
    name for name in RATE_FEATURE_NAMES if name not in {"rate_is_missing", "rate_is_stale"}
)


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
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


def _validate(frame: pd.DataFrame) -> None:
    if frame.columns.has_duplicates:
        raise ValueError("rate feature input must not contain duplicate columns")
    required = {
        "prediction_time_utc",
        "rate_observation_id",
        "rate_observed_at_utc",
        "rate_value",
        "rate_is_missing",
        "rate_is_stale",
        "rate_age_seconds",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"rate feature input is missing columns: {sorted(missing)}")
    times = frame["prediction_time_utc"]
    if not isinstance(times.dtype, pd.DatetimeTZDtype) or str(times.dtype.tz) != "UTC":
        raise ValueError("prediction_time_utc must be timezone-aware UTC")
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("prediction_time_utc must be nonmissing, sorted and unique")
    for name in ("rate_is_missing", "rate_is_stale"):
        if not is_bool_dtype(frame[name].dtype) or frame[name].isna().any():
            raise ValueError(f"{name} must contain nonmissing booleans")
    values = _numeric(frame, "rate_value")
    if np.isinf(values.to_numpy()).any():
        raise ValueError("rate_value must be finite when observed")
    ages = _numeric(frame, "rate_age_seconds")
    observed_age = ages.dropna()
    if not np.isfinite(observed_age).all() or observed_age.lt(0).any():
        raise ValueError("rate_age_seconds must be finite and nonnegative when present")
    if (ages.isna() & ~frame["rate_is_missing"]).any():
        raise ValueError("rate_age_seconds may be missing only when rate context is missing")


def _event_feature_map(frame: pd.DataFrame) -> pd.DataFrame:
    identifiers = frame["rate_observation_id"]
    changed = identifiers.notna() & identifiers.ne(identifiers.shift())
    events = frame.loc[
        changed,
        [
            "rate_observation_id",
            "rate_observed_at_utc",
            "rate_value",
            "rate_is_missing",
            "rate_is_stale",
        ],
    ].copy()
    if events.empty:
        return pd.DataFrame(
            columns=[
                "rate_level_pct",
                "rate_change_1obs_bps",
                "rate_change_5obs_bps",
                "rate_change_20obs_bps",
            ]
        )
    observed = events["rate_observed_at_utc"]
    if (
        observed.isna().any()
        or not isinstance(observed.dtype, pd.DatetimeTZDtype)
        or str(observed.dtype.tz) != "UTC"
        or observed.duplicated().any()
        or not observed.is_monotonic_increasing
    ):
        raise ValueError("distinct rate observations must have increasing unique UTC timestamps")
    if events["rate_observation_id"].duplicated().any():
        raise ValueError("rate observation IDs must not reappear after a newer observation")

    value = _numeric(events, "rate_value").where(
        ~events["rate_is_missing"] & ~events["rate_is_stale"]
    )
    result = pd.DataFrame(index=events.index)
    result["rate_level_pct"] = value
    for steps, name in (
        (1, "rate_change_1obs_bps"),
        (5, "rate_change_5obs_bps"),
        (20, "rate_change_20obs_bps"),
    ):
        complete = value.notna().rolling(steps + 1, min_periods=steps + 1).sum().eq(steps + 1)
        result[name] = value.sub(value.shift(steps)).mul(100.0).where(complete)
    result.index = events["rate_observation_id"].astype(str)
    return result


def build_rate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append the frozen DFII10 level/change feature contract without row loss.

    DFII10 is already a percentage yield, not a price. Yield moves are therefore
    additive percentage-point differences converted to basis points. The joined
    daily observation is repeated across many 3-minute gold rows; its most recent
    one-, five- and twenty-observation changes are deliberately carried unchanged
    until a newer H.15 observation becomes available. Repeated as-of rows therefore
    never manufacture zero rate changes.

    Missing or stale context is masked. A twenty-observation history is required
    before all model features are finite, so early/incomplete rows route to the
    exact frozen Phase-7 price-only fallback.
    """
    _validate(frame)
    output = frame.copy(deep=True)
    mapping = _event_feature_map(frame)
    identifiers = frame["rate_observation_id"].astype("string")
    fresh = ~frame["rate_is_missing"] & ~frame["rate_is_stale"]
    for name in (
        "rate_level_pct",
        "rate_change_1obs_bps",
        "rate_change_5obs_bps",
        "rate_change_20obs_bps",
    ):
        if mapping.empty:
            output[name] = np.nan
        else:
            output[name] = identifiers.map(mapping[name]).astype("float64").where(fresh)
    return output


__all__ = ["RATE_FEATURE_NAMES", "RATE_MODEL_FEATURES", "build_rate_features"]
