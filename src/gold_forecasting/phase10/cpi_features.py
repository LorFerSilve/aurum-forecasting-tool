"""Causal monthly CPI-U features for the Phase-10 CPI ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

CPI_FEATURE_NAMES = (
    "cpi_index",
    "cpi_mom_pct",
    "cpi_yoy_pct",
    "cpi_mom_accel_pp",
    "cpi_yoy_accel_pp",
    "cpi_release_age_seconds",
    "cpi_release_within_60m",
    "cpi_is_missing",
    "cpi_is_stale",
)
CPI_MODEL_FEATURES = tuple(
    name for name in CPI_FEATURE_NAMES if name not in {"cpi_is_missing", "cpi_is_stale"}
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
        raise ValueError("CPI feature input must not contain duplicate columns")
    required = {
        "prediction_time_utc",
        "cpi_observation_id",
        "cpi_observed_at_utc",
        "cpi_available_at_utc",
        "cpi_value",
        "cpi_is_missing",
        "cpi_is_stale",
        "cpi_age_seconds",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CPI feature input is missing columns: {sorted(missing)}")
    for name in ("prediction_time_utc", "cpi_observed_at_utc", "cpi_available_at_utc"):
        column = frame[name]
        if not isinstance(column.dtype, pd.DatetimeTZDtype) or str(column.dtype.tz) != "UTC":
            raise ValueError(f"{name} must be timezone-aware UTC")
    times = frame["prediction_time_utc"]
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("prediction_time_utc must be nonmissing, sorted and unique")
    for name in ("cpi_is_missing", "cpi_is_stale"):
        if not is_bool_dtype(frame[name].dtype) or frame[name].isna().any():
            raise ValueError(f"{name} must contain nonmissing booleans")
    values = _numeric(frame, "cpi_value")
    if np.isinf(values.to_numpy()).any():
        raise ValueError("cpi_value must be finite when observed")
    ages = _numeric(frame, "cpi_age_seconds")
    observed_age = ages.dropna()
    if not np.isfinite(observed_age).all() or observed_age.lt(0).any():
        raise ValueError("cpi_age_seconds must be finite and nonnegative when present")
    if (ages.isna() & ~frame["cpi_is_missing"]).any():
        raise ValueError("cpi_age_seconds may be missing only when CPI context is missing")


def _event_feature_map(frame: pd.DataFrame) -> pd.DataFrame:
    identifiers = frame["cpi_observation_id"]
    changed = identifiers.notna() & identifiers.ne(identifiers.shift())
    events = frame.loc[
        changed,
        ["cpi_observation_id", "cpi_observed_at_utc", "cpi_available_at_utc", "cpi_value"],
    ].copy()
    if events.empty:
        return pd.DataFrame(
            columns=[
                "cpi_index",
                "cpi_mom_pct",
                "cpi_yoy_pct",
                "cpi_mom_accel_pp",
                "cpi_yoy_accel_pp",
            ]
        )
    observed = events["cpi_observed_at_utc"]
    available = events["cpi_available_at_utc"]
    if (
        observed.isna().any()
        or available.isna().any()
        or observed.duplicated().any()
        or not observed.is_monotonic_increasing
        or not available.is_monotonic_increasing
    ):
        raise ValueError("distinct CPI observations/releases must be increasing and unique")
    if events["cpi_observation_id"].duplicated().any():
        raise ValueError("CPI observation IDs must not reappear after a newer observation")
    value = _numeric(events, "cpi_value")
    if value.isna().any() or not np.isfinite(value.to_numpy()).all() or value.le(0).any():
        raise ValueError("distinct CPI index observations must be finite and positive")

    result = pd.DataFrame(index=events.index)
    result["cpi_index"] = value
    result["cpi_mom_pct"] = value.div(value.shift(1)).sub(1.0).mul(100.0)
    result["cpi_yoy_pct"] = value.div(value.shift(12)).sub(1.0).mul(100.0)
    result["cpi_mom_accel_pp"] = result["cpi_mom_pct"].sub(result["cpi_mom_pct"].shift(1))
    result["cpi_yoy_accel_pp"] = result["cpi_yoy_pct"].sub(result["cpi_yoy_pct"].shift(1))
    result.index = events["cpi_observation_id"].astype(str)
    return result


def build_cpi_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append frozen CPI level/inflation/release-age features without row loss.

    Monthly changes are computed only over distinct CPI observations, then carried
    forward until the next released observation. Repeated intraday gold rows can
    therefore never manufacture zero monthly changes. The current CPI value and
    all derived changes are masked whenever the context row is missing or stale.

    The release-age feature is measured from the modeled BLS publication timestamp,
    not from the reference month. `cpi_release_within_60m` is one only during the
    first half-open hour [release, release+60m). Fourteen distinct observations are
    needed for year-over-year acceleration; earlier rows use exact Phase-7 fallback.
    """
    _validate(frame)
    output = frame.copy(deep=True)
    mapping = _event_feature_map(frame)
    identifiers = frame["cpi_observation_id"].astype("string")
    fresh = ~frame["cpi_is_missing"] & ~frame["cpi_is_stale"]
    for name in (
        "cpi_index",
        "cpi_mom_pct",
        "cpi_yoy_pct",
        "cpi_mom_accel_pp",
        "cpi_yoy_accel_pp",
    ):
        if mapping.empty:
            output[name] = np.nan
        else:
            output[name] = identifiers.map(mapping[name]).astype("float64").where(fresh)

    release_age = (frame["prediction_time_utc"] - frame["cpi_available_at_utc"]).dt.total_seconds()
    valid_release_age = release_age.where(fresh & release_age.ge(0.0))
    output["cpi_release_age_seconds"] = valid_release_age.astype("float64")
    output["cpi_release_within_60m"] = (
        valid_release_age.lt(3600.0).astype("float64").where(valid_release_age.notna())
    )
    return output


__all__ = ["CPI_FEATURE_NAMES", "CPI_MODEL_FEATURES", "build_cpi_features"]
