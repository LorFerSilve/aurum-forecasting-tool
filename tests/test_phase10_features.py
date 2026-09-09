from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gold_forecasting.phase10.features import (
    PRICE_FEATURE_NAMES,
    SILVER_FEATURE_NAMES,
    build_silver_features,
)


def _frame(periods: int = 60) -> pd.DataFrame:
    steps = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "prediction_time_utc": pd.date_range(
                "2024-01-02", periods=periods, freq="3min", tz="UTC"
            ),
            "gold_close": 2_000.0 * np.exp(np.cumsum(0.0001 + np.sin(steps) * 0.001)),
            "silver_value": 25.0 * np.exp(np.cumsum(0.0002 + np.cos(steps) * 0.002)),
            "silver_is_missing": False,
            "silver_is_stale": False,
            "silver_age_seconds": 0.0,
            "audit_id": [f"row-{step}" for step in range(periods)],
        }
    )


def test_formulas_warmup_and_manual_paired_correlation() -> None:
    frame = _frame()
    result = build_silver_features(frame)
    assert result["gold_return_1_bps"].iloc[:1].isna().all()
    assert result["silver_return_1_bps"].iloc[:1].isna().all()
    assert result["gold_momentum_5_bps"].iloc[:5].isna().all()
    assert result["silver_momentum_5_bps"].iloc[:5].isna().all()
    assert result["gold_silver_correlation_20"].iloc[:20].isna().all()
    assert result.loc[5, "gold_momentum_5_bps"] == pytest.approx(
        10_000 * np.log(frame.loc[5, "gold_close"] / frame.loc[0, "gold_close"])
    )
    assert result.loc[5, "silver_momentum_5_bps"] == pytest.approx(
        10_000 * np.log(frame.loc[5, "silver_value"] / frame.loc[0, "silver_value"])
    )
    assert result.loc[20, "gold_silver_log_ratio"] == pytest.approx(
        np.log(frame.loc[20, "gold_close"] / frame.loc[20, "silver_value"])
    )
    assert result.loc[20, "silver_return_1_bps"] == pytest.approx(
        10_000 * np.log(frame.loc[20, "silver_value"] / frame.loc[19, "silver_value"])
    )
    gold_returns = np.diff(np.log(frame.loc[:20, "gold_close"].to_numpy()))
    silver_returns = np.diff(np.log(frame.loc[:20, "silver_value"].to_numpy()))
    assert result.loc[20, "gold_silver_correlation_20"] == pytest.approx(
        np.corrcoef(gold_returns, silver_returns)[0, 1]
    )


def test_preserves_input_rows_index_and_audit_columns_without_mutation() -> None:
    frame = _frame()
    frame.index = pd.Index([f"sample-{index}" for index in frame.index], name="sample")
    before = frame.copy(deep=True)
    result = build_silver_features(frame)
    assert_frame_equal(frame, before)
    assert_frame_equal(result.loc[:, frame.columns], frame)
    assert tuple(result.columns[len(frame.columns) :]) == (
        *PRICE_FEATURE_NAMES,
        *SILVER_FEATURE_NAMES[:4],
    )
    assert not result.columns.has_duplicates


def test_future_price_flags_and_age_mutation_does_not_change_history() -> None:
    frame = _frame()
    original = build_silver_features(frame)
    changed = frame.copy(deep=True)
    changed.loc[35:, "gold_close"] *= 2.0
    changed.loc[35:, "silver_value"] *= 0.5
    changed.loc[38:, "silver_is_stale"] = True
    changed.loc[38:, "silver_age_seconds"] = 3_600.0
    changed.loc[41:, "silver_is_missing"] = True
    changed.loc[41:, "silver_value"] = np.nan
    changed.loc[41:, "silver_age_seconds"] = np.nan
    mutated = build_silver_features(changed)
    truncated = build_silver_features(frame.iloc[:35])
    assert_frame_equal(mutated.iloc[:35], original.iloc[:35])
    assert_frame_equal(truncated, original.iloc[:35])


@pytest.mark.parametrize("flag", ["silver_is_missing", "silver_is_stale"])
def test_silver_outage_restarts_every_window_and_keeps_gold_features(flag: str) -> None:
    frame = _frame()
    baseline = build_silver_features(frame)
    frame.loc[25, flag] = True
    frame.loc[25, "silver_age_seconds"] = 600.0
    frame.loc[25, "silver_value"] = 100_000.0
    result = build_silver_features(frame)
    assert_frame_equal(
        result.loc[:, list(PRICE_FEATURE_NAMES)], baseline.loc[:, list(PRICE_FEATURE_NAMES)]
    )
    assert result.loc[25:26, "silver_return_1_bps"].isna().all()
    assert pd.notna(result.loc[27, "silver_return_1_bps"])
    assert result.loc[25:30, "silver_momentum_5_bps"].isna().all()
    assert pd.notna(result.loc[31, "silver_momentum_5_bps"])
    assert result.loc[25:45, "gold_silver_correlation_20"].isna().all()
    assert pd.notna(result.loc[46, "gold_silver_correlation_20"])
    assert pd.isna(result.loc[25, "gold_silver_log_ratio"])
    assert pd.notna(result.loc[26, "gold_silver_log_ratio"])
    assert result.loc[25, "silver_value"] == 100_000.0


def test_timestamp_gap_restarts_gold_and_silver_windows() -> None:
    frame = _frame().drop(index=25)
    result = build_silver_features(frame)
    for column in ("gold_return_1_bps", "silver_return_1_bps"):
        assert pd.isna(result.loc[26, column])
        assert pd.notna(result.loc[27, column])
    for column in ("gold_momentum_5_bps", "silver_momentum_5_bps"):
        assert result.loc[26:30, column].isna().all()
        assert pd.notna(result.loc[31, column])
    assert result.loc[26:45, "gold_silver_correlation_20"].isna().all()
    assert pd.notna(result.loc[46, "gold_silver_correlation_20"])


def test_missing_source_and_constant_prices_produce_no_fabricated_correlation() -> None:
    frame = _frame()
    frame["silver_value"] = np.nan
    frame["silver_is_missing"] = True
    frame["silver_age_seconds"] = np.nan
    result = build_silver_features(frame)
    assert result.loc[:, list(SILVER_FEATURE_NAMES[:4])].isna().all().all()
    frame = _frame()
    frame["silver_value"] = 25.0
    result = build_silver_features(frame)
    assert result["gold_silver_correlation_20"].isna().all()
    assert result["silver_return_1_bps"].iloc[1:].eq(0.0).all()


def test_asof_repeated_quote_is_not_a_new_candle_even_before_stale_threshold() -> None:
    frame = _frame()
    # Rows 25 and 26 still hold row 24's quote; freshness threshold is not breached.
    frame.loc[25:26, "silver_value"] = frame.loc[24, "silver_value"]
    frame.loc[25:26, "silver_age_seconds"] = [180.0, 360.0]
    result = build_silver_features(frame)
    assert result.loc[25:27, "silver_return_1_bps"].isna().all()
    assert pd.notna(result.loc[28, "silver_return_1_bps"])
    assert result.loc[25:31, "silver_momentum_5_bps"].isna().all()
    assert pd.notna(result.loc[32, "silver_momentum_5_bps"])
    assert result.loc[25:46, "gold_silver_correlation_20"].isna().all()
    assert pd.notna(result.loc[47, "gold_silver_correlation_20"])
    # A carried but fresh level is still an auditable ratio, with increasing age.
    assert result.loc[25:26, "gold_silver_log_ratio"].notna().all()


def test_empty_input_retains_schema() -> None:
    frame = _frame().iloc[:0]
    result = build_silver_features(frame)
    assert result.empty
    assert set(PRICE_FEATURE_NAMES + SILVER_FEATURE_NAMES).issubset(result.columns)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("gold_close", np.nan),
        ("gold_close", np.inf),
        ("gold_close", 0.0),
        ("silver_value", -1.0),
        ("silver_value", np.inf),
        ("silver_age_seconds", -1.0),
        ("silver_age_seconds", np.inf),
        ("silver_age_seconds", np.nan),
    ],
)
def test_rejects_invalid_price_and_age_values(column: str, value: float) -> None:
    frame = _frame()
    frame.loc[5, column] = value
    with pytest.raises(ValueError, match=column):
        build_silver_features(frame)


@pytest.mark.parametrize("column", ["gold_close", "silver_value", "silver_age_seconds"])
def test_rejects_nonnumeric_and_boolean_prices(column: str) -> None:
    frame = _frame()
    frame[column] = "25.0"
    with pytest.raises(ValueError, match="numeric"):
        build_silver_features(frame)
    frame[column] = True
    with pytest.raises(ValueError, match="numeric"):
        build_silver_features(frame)


@pytest.mark.parametrize("column", ["silver_is_missing", "silver_is_stale"])
def test_rejects_nonboolean_and_missing_flags(column: str) -> None:
    frame = _frame()
    frame[column] = 0
    with pytest.raises(ValueError, match="booleans"):
        build_silver_features(frame)
    frame[column] = pd.Series(False, index=frame.index, dtype="boolean")
    frame.loc[5, column] = pd.NA
    with pytest.raises(ValueError, match="booleans"):
        build_silver_features(frame)


def test_rejects_missing_or_duplicate_columns() -> None:
    frame = _frame()
    with pytest.raises(ValueError, match="missing columns"):
        build_silver_features(frame.drop(columns="gold_close"))
    with pytest.raises(ValueError, match="duplicate columns"):
        build_silver_features(pd.concat([frame, frame[["gold_close"]]], axis=1))


@pytest.mark.parametrize("problem", ["naive", "non_utc", "unsorted", "duplicate", "missing"])
def test_rejects_invalid_prediction_timestamps(problem: str) -> None:
    frame = _frame()
    if problem == "naive":
        frame["prediction_time_utc"] = frame["prediction_time_utc"].dt.tz_localize(None)
    elif problem == "non_utc":
        frame["prediction_time_utc"] = frame["prediction_time_utc"].dt.tz_convert("Europe/Brussels")
    elif problem == "unsorted":
        frame = frame.iloc[::-1]
    elif problem == "duplicate":
        frame.loc[5, "prediction_time_utc"] = frame.loc[4, "prediction_time_utc"]
    else:
        frame.loc[5, "prediction_time_utc"] = pd.NaT
    with pytest.raises(ValueError, match="prediction_time_utc"):
        build_silver_features(frame)


@pytest.mark.parametrize(
    "parameters",
    [
        {"cadence_seconds": 0},
        {"cadence_seconds": True},
        {"cadence_seconds": 180.0},
        {"momentum_steps": 1},
        {"momentum_steps": -1},
        {"correlation_steps": 5},
    ],
)
def test_rejects_invalid_or_misnamed_window_parameters(parameters: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        build_silver_features(_frame(), **parameters)
