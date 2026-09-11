"""Frozen CPI feature semantics and row-preserving warm-up behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold_forecasting.phase10.cpi_features import CPI_MODEL_FEATURES, build_cpi_features


def _frame() -> pd.DataFrame:
    months = pd.date_range("2020-01-01", periods=15, freq="MS", tz="UTC")
    rows: list[dict[str, object]] = []
    for index, observed in enumerate(months):
        available = observed + pd.Timedelta(days=40)
        for minutes in (30, 90):
            prediction = available + pd.Timedelta(minutes=minutes)
            rows.append(
                {
                    "prediction_time_utc": prediction,
                    "cpi_observation_id": f"cpi-{observed:%Y-%m}",
                    "cpi_observed_at_utc": observed,
                    "cpi_available_at_utc": available,
                    "cpi_value": 250.0 + index,
                    "cpi_is_missing": False,
                    "cpi_is_stale": False,
                    "cpi_age_seconds": (prediction - observed).total_seconds(),
                }
            )
    return pd.DataFrame(rows).sort_values("prediction_time_utc").reset_index(drop=True)


def test_cpi_changes_use_distinct_months_not_repeated_gold_rows() -> None:
    result = build_cpi_features(_frame())
    grouped = result.groupby("cpi_observation_id", sort=False)
    for name in (
        "cpi_index",
        "cpi_mom_pct",
        "cpi_yoy_pct",
        "cpi_mom_accel_pp",
        "cpi_yoy_accel_pp",
    ):
        assert grouped[name].nunique(dropna=False).eq(1).all()
    second = result.loc[result.cpi_observation_id.eq("cpi-2020-02")]
    expected_mom = (251.0 / 250.0 - 1.0) * 100.0
    assert np.allclose(second.cpi_mom_pct, expected_mom)
    assert second.cpi_yoy_pct.isna().all()


def test_cpi_yoy_acceleration_requires_fourteen_distinct_observations() -> None:
    result = build_cpi_features(_frame())
    month_13 = result.loc[result.cpi_observation_id.eq("cpi-2021-01")]
    month_14 = result.loc[result.cpi_observation_id.eq("cpi-2021-02")]
    assert month_13.cpi_yoy_pct.notna().all()
    assert month_13.cpi_yoy_accel_pp.isna().all()
    assert month_14.cpi_yoy_accel_pp.notna().all()
    assert month_14[list(CPI_MODEL_FEATURES)].notna().all().all()


def test_cpi_release_age_and_first_hour_indicator_are_causal() -> None:
    result = build_cpi_features(_frame())
    first_month = result.loc[result.cpi_observation_id.eq("cpi-2020-01")]
    assert first_month.cpi_release_age_seconds.tolist() == [1800.0, 5400.0]
    assert first_month.cpi_release_within_60m.tolist() == [1.0, 0.0]


def test_cpi_missing_or_stale_rows_mask_all_model_features() -> None:
    frame = _frame()
    frame.loc[28, "cpi_is_stale"] = True
    frame.loc[29, "cpi_is_missing"] = True
    result = build_cpi_features(frame)
    assert result.loc[28, list(CPI_MODEL_FEATURES)].isna().all()
    assert result.loc[29, list(CPI_MODEL_FEATURES)].isna().all()


def test_cpi_feature_builder_rejects_noncausal_or_invalid_input() -> None:
    frame = _frame()
    frame.loc[10, "cpi_age_seconds"] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        build_cpi_features(frame)

    duplicate = _frame()
    duplicate.loc[2, "prediction_time_utc"] = duplicate.loc[1, "prediction_time_utc"]
    duplicate = duplicate.sort_values("prediction_time_utc").reset_index(drop=True)
    with pytest.raises(ValueError, match="sorted and unique"):
        build_cpi_features(duplicate)
