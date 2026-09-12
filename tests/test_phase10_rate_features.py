"""Causal DFII10 feature construction on repeated intraday as-of rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold_forecasting.phase10.rate_features import RATE_MODEL_FEATURES, build_rate_features

_DERIVED = (
    "rate_level_pct",
    "rate_change_1obs_bps",
    "rate_change_5obs_bps",
    "rate_change_20obs_bps",
)


def _frame(observations: int = 25, repeats: int = 2) -> pd.DataFrame:
    ids: list[str] = []
    observed: list[pd.Timestamp] = []
    values: list[float] = []
    times: list[pd.Timestamp] = []
    for index in range(observations):
        reference = pd.Timestamp("2020-01-01T00:00:00Z") + pd.Timedelta(days=index)
        for repeat in range(repeats):
            ids.append(f"dfii10-{reference:%Y-%m-%d}")
            observed.append(reference)
            values.append(1.00 + index * 0.01)
            times.append(
                pd.Timestamp("2020-02-01T00:00:00Z")
                + pd.Timedelta(minutes=3 * (index * repeats + repeat))
            )
    n = len(ids)
    return pd.DataFrame(
        {
            "prediction_time_utc": pd.DatetimeIndex(times),
            "rate_observation_id": ids,
            "rate_observed_at_utc": pd.DatetimeIndex(observed),
            "rate_value": values,
            "rate_is_missing": False,
            "rate_is_stale": False,
            "rate_age_seconds": np.full(n, 3600.0),
        }
    )


def test_rate_changes_use_distinct_daily_observations_not_intraday_repetitions() -> None:
    result = build_rate_features(_frame())
    twentieth = result.loc[result.rate_observation_id == "dfii10-2020-01-21"]
    assert twentieth.rate_level_pct.tolist() == pytest.approx([1.20, 1.20])
    assert twentieth.rate_change_1obs_bps.tolist() == pytest.approx([1.0, 1.0])
    assert twentieth.rate_change_5obs_bps.tolist() == pytest.approx([5.0, 5.0])
    assert twentieth.rate_change_20obs_bps.tolist() == pytest.approx([20.0, 20.0])
    assert not set(("rate_is_missing", "rate_is_stale")) & set(RATE_MODEL_FEATURES)


def test_rate_history_warmup_and_staleness_route_features_to_missing() -> None:
    frame = _frame()
    result = build_rate_features(frame)
    assert (
        result.loc[
            result.rate_observation_id == "dfii10-2020-01-01",
            "rate_change_1obs_bps",
        ]
        .isna()
        .all()
    )
    assert (
        result.loc[
            result.rate_observation_id == "dfii10-2020-01-05",
            "rate_change_5obs_bps",
        ]
        .isna()
        .all()
    )
    assert (
        result.loc[
            result.rate_observation_id == "dfii10-2020-01-20",
            "rate_change_20obs_bps",
        ]
        .isna()
        .all()
    )

    stale = frame.copy()
    stale.loc[stale.rate_observation_id == "dfii10-2020-01-25", "rate_is_stale"] = True
    masked = build_rate_features(stale)
    rows = masked.rate_observation_id == "dfii10-2020-01-25"
    assert masked.loc[rows, list(_DERIVED)].isna().all().all()
    assert masked.loc[rows, "rate_age_seconds"].eq(3600.0).all()


def test_future_rate_value_cannot_rewrite_earlier_feature_rows() -> None:
    frame = _frame()
    original = build_rate_features(frame)
    changed = frame.copy()
    changed.loc[changed.rate_observation_id == "dfii10-2020-01-25", "rate_value"] += 5.0
    mutated = build_rate_features(changed)
    cutoff = changed.index[changed.rate_observation_id == "dfii10-2020-01-24"].max()
    pd.testing.assert_frame_equal(original.loc[:cutoff], mutated.loc[:cutoff])


def test_reappearing_old_rate_observation_fails_closed() -> None:
    frame = _frame(observations=3, repeats=1)
    frame.loc[2, "rate_observation_id"] = frame.loc[0, "rate_observation_id"]
    frame.loc[2, "rate_observed_at_utc"] = frame.loc[0, "rate_observed_at_utc"]
    with pytest.raises(ValueError, match=r"increasing unique|must not reappear"):
        build_rate_features(frame)
