"""Backward as-of context joins, including revisions of older observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gold_forecasting.phase10.contracts import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    OBSERVATION_COLUMNS,
    ContextSource,
    utc_series,
    validate_observations,
)


def join_context(
    predictions: pd.DataFrame,
    observations: pd.DataFrame | None,
    source: ContextSource,
) -> pd.DataFrame:
    """Attach the latest observed value *as known* at each prediction cutoff.

    A late revision of an older period must not replace a newer period. Resolve
    that release stream first, then use a backward-only as-of join. No value is
    backfilled. Raw stale values remain auditable; model feature builders mask
    them. A disabled or absent optional source preserves every prediction row.
    """
    if not predictions.columns.is_unique:
        raise ValueError("prediction columns must be unique")
    if "prediction_time_utc" not in predictions:
        raise ValueError("prediction_time_utc is required")
    times = utc_series(predictions["prediction_time_utc"], "prediction_time_utc")
    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("prediction times must be sorted and unique")
    if ((times < DEVELOPMENT_START) | (times >= DEVELOPMENT_END)).any():
        raise ValueError("prediction times must remain in development 2020-2024")
    prefix = source.source_id + "_"
    fields = [name for name in OBSERVATION_COLUMNS if name != "source_id"]
    new_names = [prefix + name for name in (*fields, "age_seconds", "is_stale", "is_missing")]
    if set(new_names).intersection(predictions.columns):
        raise ValueError("context columns would overwrite existing prediction fields")
    result = predictions.copy()
    if source.enabled and observations is not None:
        validated = validate_observations(observations, source)
    else:
        validated = None
    if validated is None or validated.empty:
        for name in fields:
            if name.endswith("_at_utc"):
                result[prefix + name] = pd.Series(pd.NaT, index=result.index, dtype=times.dtype)
            else:
                result[prefix + name] = np.nan if name == "value" else None
    else:
        # A single ordered pass keeps only releases affecting the newest period.
        keep: list[int] = []
        latest_observed: pd.Timestamp | None = None
        for index, observed in enumerate(validated["observed_at_utc"]):
            if latest_observed is None or observed >= latest_observed:
                latest_observed = observed
                keep.append(index)
        timeline = validated.iloc[keep].loc[:, fields].rename(
            columns={name: prefix + name for name in fields}
        )
        aligned = pd.merge_asof(
            pd.DataFrame({"prediction_time_utc": times.reset_index(drop=True)}),
            timeline,
            left_on="prediction_time_utc",
            right_on=prefix + "available_at_utc",
            direction="backward",
            allow_exact_matches=True,
        )
        for name in fields:
            result[prefix + name] = aligned[prefix + name].array
    # Age tracks the observation, so a late revision cannot make old data fresh.
    age = (times - result[prefix + "observed_at_utc"]).dt.total_seconds()
    result[prefix + "age_seconds"] = age
    result[prefix + "is_stale"] = age.gt(source.stale_after_seconds)
    result[prefix + "is_missing"] = result[prefix + "value"].isna()
    return result


def context_coverage(frame: pd.DataFrame, source_id: str) -> dict[str, int | float | None]:
    """Report raw availability separately from fresh usable values."""
    missing = frame[source_id + "_is_missing"]
    stale = frame[source_id + "_is_stale"]
    age = frame[source_id + "_age_seconds"].dropna()
    return {
        "rows": len(frame),
        "missing_rows": int(missing.sum()),
        "stale_rows": int(stale.sum()),
        "usable_rows": int((~missing & ~stale).sum()),
        "usable_fraction": float((~missing & ~stale).mean()) if len(frame) else None,
        "maximum_age_seconds": float(age.max()) if len(age) else None,
    }
