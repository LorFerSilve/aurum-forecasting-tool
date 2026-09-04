from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from gold_forecasting.config import load_project_config
from gold_forecasting.features import build_mvp_features
from gold_forecasting.labels import build_mvp_labels, classify_future_returns

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _candles(timeframe: str, periods: int, *, gap_after: int | None = None) -> pd.DataFrame:
    minutes = {"1min": 1, "3min": 3}[timeframe]
    opens = list(
        pd.date_range("2024-01-02T00:00:00Z", periods=periods, freq=f"{minutes}min")
    )
    if gap_after is not None:
        for index in range(gap_after, periods):
            opens[index] += pd.Timedelta(minutes=minutes)
    bid_open = pd.Series(2_000.0 + np.arange(periods) * 0.2, dtype="float64")
    bid_close = bid_open + np.sin(np.arange(periods)) * 0.1
    return pd.DataFrame(
        {
            "timestamp_open_utc": pd.DatetimeIndex(opens),
            "timestamp_close_utc": pd.DatetimeIndex(opens) + pd.Timedelta(minutes=minutes),
            "bid_open": bid_open,
            "bid_high": np.maximum(bid_open, bid_close) + 0.15,
            "bid_low": np.minimum(bid_open, bid_close) - 0.15,
            "bid_close": bid_close,
            "instrument": "XAU_USD",
            "timeframe": timeframe,
            "is_complete": True,
            "source": "fixture",
            "raw_file_hash": "a" * 64,
            "ingested_at_utc": pd.Timestamp("2025-01-01T00:00:00Z"),
            "dataset_version": "sha256:" + "b" * 64,
        }
    )


def _project_config():  # type annotation would add noise to every test call
    return load_project_config(PROJECT_ROOT / "configs" / "mvp.yaml")


def test_feature_formulas_catalog_order_and_availability() -> None:
    candles = _candles("3min", 30)
    result = build_mvp_features(candles, _project_config().features)

    assert len(result.features) == 10
    assert result.dropped_for_history == 20
    assert tuple(result.features.columns[8:]) == result.catalog.feature_names
    row = result.features.iloc[0]
    source_index = 20
    expected_return = 10_000.0 * np.log(
        candles.loc[source_index, "bid_close"] / candles.loc[source_index - 1, "bid_close"]
    )
    expected_momentum = 10_000.0 * np.log(
        candles.loc[source_index, "bid_close"] / candles.loc[source_index - 5, "bid_close"]
    )
    assert np.isclose(row["close_log_return_1_bps"], expected_return)
    assert np.isclose(row["momentum_5_bps"], expected_momentum)
    assert row["feature_available_at_utc"] == row["prediction_time_utc"]
    assert row["feature_window_start_utc"] == candles.loc[0, "timestamp_open_utc"]


def test_zero_range_close_position_uses_configured_constant() -> None:
    candles = _candles("3min", 25)
    for column in ("bid_open", "bid_high", "bid_low", "bid_close"):
        candles[column] = 2_000.0

    result = build_mvp_features(candles, _project_config().features)

    assert result.features["close_position"].eq(0.5).all()


def test_rolling_features_restart_after_a_gap() -> None:
    candles = _candles("3min", 50, gap_after=25)

    result = build_mvp_features(candles, _project_config().features)

    assert len(result.features) == 10
    assert result.features.iloc[5]["feature_window_start_utc"] == candles.loc[
        25, "timestamp_open_utc"
    ]


def test_future_mutation_does_not_change_older_features() -> None:
    candles = _candles("3min", 45)
    original = build_mvp_features(candles, _project_config().features).features
    cutoff = candles.loc[32, "timestamp_close_utc"]
    changed = candles.copy()
    price_columns = ["bid_open", "bid_high", "bid_low", "bid_close"]
    changed.loc[33:, price_columns] = changed.loc[33:, price_columns] * 1.2

    mutated = build_mvp_features(changed, _project_config().features).features

    before = original.loc[original["prediction_time_utc"].le(cutoff)].reset_index(drop=True)
    after = mutated.loc[mutated["prediction_time_utc"].le(cutoff)].reset_index(drop=True)
    assert_frame_equal(after, before)


def test_labels_use_exact_entry_and_exit_and_preserve_audit_prices() -> None:
    candles = _candles("1min", 40)
    candidates = pd.DataFrame(
        {
            "instrument": ["XAU_USD"],
            "source": ["fixture"],
            "prediction_time_utc": [pd.Timestamp("2024-01-02T00:03:00Z")],
        }
    )

    result = build_mvp_labels(candles, candidates, _project_config().labels)

    assert result.output_row_count == 1
    row = result.labels.iloc[0]
    assert row["entry_time_utc"] == pd.Timestamp("2024-01-02T00:04:00Z")
    assert row["label_end_time_utc"] == pd.Timestamp("2024-01-02T00:19:00Z")
    assert row["entry_bid_open"] == candles.loc[4, "bid_open"]
    assert row["exit_bid_open"] == candles.loc[19, "bid_open"]
    assert row["entry_time_utc"] > row["prediction_time_utc"]


def test_missing_minute_inside_label_path_drops_the_sample() -> None:
    candles = _candles("1min", 40).drop(index=10).reset_index(drop=True)
    candidates = pd.DataFrame(
        {
            "instrument": ["XAU_USD"],
            "source": ["fixture"],
            "prediction_time_utc": [pd.Timestamp("2024-01-02T00:03:00Z")],
        }
    )

    result = build_mvp_labels(candles, candidates, _project_config().labels)

    assert result.labels.empty
    assert result.dropped_missing_path == 1


def test_class_thresholds_are_strict_with_inclusive_neutral_boundaries() -> None:
    values = pd.Series([-6.0001, -6.0, 0.0, 6.0, 6.0001])

    labels = classify_future_returns(values, threshold_bps=6.0)

    assert labels.tolist() == ["down", "neutral", "neutral", "neutral", "up"]
