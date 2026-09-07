from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gold_forecasting.config import load_project_config
from gold_forecasting.labels import LABEL_COLUMNS, LabelBuildError, build_mvp_labels
from gold_forecasting.labels.multihorizon import (
    MULTIHORIZON_LABEL_COLUMNS,
    build_horizon_labels,
)


def _candles(periods: int = 220, *, start: str = "2024-01-02T00:00:00Z") -> pd.DataFrame:
    opens = pd.date_range(start, periods=periods, freq="1min")
    prices = 2_000.0 + np.arange(periods) * 0.2
    return pd.DataFrame(
        {
            "timestamp_open_utc": opens,
            "timestamp_close_utc": opens + pd.Timedelta(minutes=1),
            "bid_open": prices,
            "bid_high": prices + 0.4,
            "bid_low": prices - 0.3,
            "bid_close": prices + 0.1,
            "instrument": "XAU_USD",
            "timeframe": "1min",
            "is_complete": True,
            "source": "fixture",
            "raw_file_hash": "a" * 64,
            "ingested_at_utc": pd.Timestamp("2026-01-01T00:00:00Z"),
            "dataset_version": "sha256:" + "b" * 64,
        }
    )


def _candidates(*times: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument": "XAU_USD",
            "source": "fixture",
            "prediction_time_utc": pd.to_datetime(list(times), utc=True),
        }
    )


@pytest.mark.parametrize("horizon", [3, 6, 9, 12, 15, 30, 60, 180])
def test_exact_multi_horizon_path_and_continuous_targets(horizon: int) -> None:
    candles = _candles()
    result = build_horizon_labels(
        candles, _candidates("2024-01-02T00:03:00Z"), horizon_minutes=horizon
    )

    assert result.candidate_count == result.output_row_count == 1
    assert result.dropped_missing_path == 0
    assert tuple(result.labels.columns) == MULTIHORIZON_LABEL_COLUMNS
    row = result.labels.iloc[0]
    entry_index, exit_index = 4, 4 + horizon
    entry_price = candles.loc[entry_index, "bid_open"]
    exit_price = candles.loc[exit_index, "bid_open"]
    assert row["entry_time_utc"] == candles.loc[entry_index, "timestamp_open_utc"]
    assert row["label_end_time_utc"] == candles.loc[exit_index, "timestamp_open_utc"]
    assert row["horizon_minutes"] == horizon
    assert row["entry_bid_open"] == entry_price
    assert row["exit_bid_open"] == exit_price
    assert row["future_return_bps"] == pytest.approx(10_000 * np.log(exit_price / entry_price))
    assert row["arithmetic_return_bps"] == pytest.approx(10_000 * (exit_price / entry_price - 1))
    expected_range = (
        (
            candles.loc[entry_index : exit_index - 1, "bid_high"].max()
            - candles.loc[entry_index : exit_index - 1, "bid_low"].min()
        )
        / entry_price
        * 10_000
    )
    expected_vol = (
        np.sqrt(np.square(np.diff(np.log(candles.loc[entry_index:exit_index, "bid_open"]))).sum())
        * 10_000
    )
    assert row["future_range_bps"] == pytest.approx(expected_range)
    assert row["future_realized_vol_bps"] == pytest.approx(expected_vol)
    assert row["label_spec_version"] == "v0.2"
    assert row["entry_raw_file_hash"] == "a" * 64
    assert row["exit_dataset_version"] == "sha256:" + "b" * 64


def test_fifteen_minute_legacy_columns_match_mvp_exactly() -> None:
    candles = _candles()
    candidates = _candidates("2024-01-02T00:03:00Z", "2024-01-02T00:06:00Z")
    config = load_project_config(Path(__file__).resolve().parents[1] / "configs" / "mvp.yaml")
    old_labels = build_mvp_labels(candles, candidates, config.labels).labels
    new_labels = build_horizon_labels(candles, candidates, horizon_minutes=15).labels

    assert_frame_equal(new_labels.loc[:, list(LABEL_COLUMNS)], old_labels)


@pytest.mark.parametrize("missing_index", [4, 5, 10, 19])
def test_any_missing_open_including_entry_and_exit_drops_sample(missing_index: int) -> None:
    candles = _candles().drop(index=missing_index).reset_index(drop=True)
    result = build_horizon_labels(candles, _candidates("2024-01-02T00:03:00Z"), horizon_minutes=15)

    assert result.labels.empty
    assert result.dropped_missing_path == 1
    assert tuple(result.labels.columns) == MULTIHORIZON_LABEL_COLUMNS


def test_future_mutation_after_exit_and_exit_candle_range_cannot_change_targets() -> None:
    candles = _candles()
    candidates = _candidates("2024-01-02T00:03:00Z")
    original = build_horizon_labels(candles, candidates, horizon_minutes=15).labels
    changed = candles.copy(deep=True)
    price_columns = ["bid_open", "bid_high", "bid_low", "bid_close"]
    changed.loc[20:, price_columns] *= 1.2
    changed.loc[19, "bid_high"] *= 1.3
    changed.loc[19, "bid_low"] *= 0.7
    changed.loc[19, "bid_close"] *= 1.1

    mutated = build_horizon_labels(changed, candidates, horizon_minutes=15).labels

    assert_frame_equal(mutated, original)


def test_changing_path_changes_targets_but_not_other_instrument_or_prior_path() -> None:
    candles = _candles()
    other = candles.assign(
        instrument="OTHER", bid_open=100.0, bid_high=110.0, bid_low=90.0, bid_close=100.0
    )
    candidates = _candidates("2024-01-02T00:03:00Z")
    original = build_horizon_labels(candles, candidates, horizon_minutes=15).labels
    combined = pd.concat([other, candles], ignore_index=True)
    result = build_horizon_labels(combined, candidates, horizon_minutes=15).labels
    assert_frame_equal(result, original)

    changed = candles.copy(deep=True)
    changed.loc[8, "bid_high"] = 2_100.0
    changed_labels = build_horizon_labels(changed, candidates, horizon_minutes=15).labels
    assert changed_labels.iloc[0]["future_range_bps"] > original.iloc[0]["future_range_bps"]
    assert changed_labels.iloc[0]["future_return_bps"] == original.iloc[0]["future_return_bps"]


def test_latency_and_cost_aware_neutral_threshold_are_explicit_and_versioned() -> None:
    result = build_horizon_labels(
        _candles(),
        _candidates("2024-01-02T00:03:00Z"),
        horizon_minutes=3,
        execution_latency_minutes=2,
        neutral_threshold_bps=100.0,
        label_spec_version="test-v2",
    )

    row = result.labels.iloc[0]
    assert row["entry_time_utc"] == pd.Timestamp("2024-01-02T00:05:00Z")
    assert row["label_end_time_utc"] == pd.Timestamp("2024-01-02T00:08:00Z")
    assert row["target_class"] == "neutral"
    assert row["execution_latency_minutes"] == 2
    assert row["neutral_threshold_bps"] == 100.0
    assert row["label_spec_version"] == "test-v2"


def test_observed_ask_prices_are_preserved_and_not_invented() -> None:
    candles = _candles()
    candles["ask_open"] = candles["bid_open"] + 0.5
    result = build_horizon_labels(candles, _candidates("2024-01-02T00:03:00Z"), horizon_minutes=15)

    assert result.labels.iloc[0]["entry_ask_open"] == candles.loc[4, "ask_open"]
    assert result.labels.iloc[0]["exit_ask_open"] == candles.loc[19, "ask_open"]


@pytest.mark.parametrize("ask", [0.0, -1.0, float("nan"), float("inf"), 1.0, "invalid"])
def test_invalid_observed_asks_fail_closed(ask: float | str) -> None:
    candles = _candles()
    candles["ask_open"] = ask
    with pytest.raises(LabelBuildError, match="ask_open"):
        build_horizon_labels(candles, _candidates("2024-01-02T00:03:00Z"), horizon_minutes=15)


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_bid_prices_fail_closed(price: float) -> None:
    candles = _candles()
    candles.loc[5, "bid_open"] = price
    with pytest.raises(LabelBuildError, match="invalid 1min label input"):
        build_horizon_labels(candles, _candidates("2024-01-02T00:03:00Z"), horizon_minutes=15)


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5])
def test_horizon_must_be_positive_integer(horizon: int) -> None:
    with pytest.raises(LabelBuildError, match="horizon_minutes"):
        build_horizon_labels(
            _candles(), _candidates("2024-01-02T00:03:00Z"), horizon_minutes=horizon
        )


@pytest.mark.parametrize("latency", [0, -1, True, 1.5])
def test_latency_must_be_strictly_positive_integer(latency: int) -> None:
    with pytest.raises(LabelBuildError, match="execution_latency_minutes"):
        build_horizon_labels(
            _candles(),
            _candidates("2024-01-02T00:03:00Z"),
            horizon_minutes=15,
            execution_latency_minutes=latency,
        )


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf"), True])
def test_neutral_threshold_must_be_finite_and_positive(threshold: float) -> None:
    with pytest.raises(LabelBuildError, match="neutral_threshold_bps"):
        build_horizon_labels(
            _candles(),
            _candidates("2024-01-02T00:03:00Z"),
            horizon_minutes=15,
            neutral_threshold_bps=threshold,
        )


def test_holdout_candles_and_prediction_candidates_are_rejected() -> None:
    with pytest.raises(LabelBuildError, match="reserved holdout"):
        build_horizon_labels(
            _candles(start="2025-01-01T00:00:00Z"),
            _candidates("2024-01-02T00:03:00Z"),
            horizon_minutes=15,
        )
    with pytest.raises(LabelBuildError, match="reserved holdout"):
        build_horizon_labels(_candles(), _candidates("2025-01-01T00:00:00Z"), horizon_minutes=15)


def test_candidate_exit_reaching_holdout_drops_without_accessing_holdout_data() -> None:
    candles = _candles(30, start="2024-12-31T23:30:00Z")
    result = build_horizon_labels(
        candles,
        _candidates("2024-12-31T23:30:00Z", "2024-12-31T23:45:00Z"),
        horizon_minutes=15,
    )

    assert result.candidate_count == 2
    assert result.output_row_count == result.dropped_missing_path == 1
    assert result.labels.iloc[0]["label_end_time_utc"] == pd.Timestamp("2024-12-31T23:46:00Z")


def test_naive_and_misaligned_prediction_timestamps_are_rejected() -> None:
    candidates = _candidates("2024-01-02T00:03:00Z")
    candidates["prediction_time_utc"] = candidates["prediction_time_utc"].dt.tz_localize(None)
    with pytest.raises(LabelBuildError, match="timezone-aware"):
        build_horizon_labels(_candles(), candidates, horizon_minutes=15)
    with pytest.raises(LabelBuildError, match="three-minute UTC grid"):
        build_horizon_labels(_candles(), _candidates("2024-01-02T00:04:00Z"), horizon_minutes=15)


def test_inputs_unchanged_and_empty_candle_input_reports_unavailable_path() -> None:
    candles = _candles()
    candidates = _candidates("2024-01-02T00:03:00Z")
    original_candles, original_candidates = candles.copy(deep=True), candidates.copy(deep=True)
    build_horizon_labels(candles, candidates, horizon_minutes=15)
    assert_frame_equal(candles, original_candles)
    assert_frame_equal(candidates, original_candidates)

    result = build_horizon_labels(candles.iloc[:0], candidates, horizon_minutes=15)
    assert result.output_row_count == 0
    assert result.dropped_missing_path == 1
