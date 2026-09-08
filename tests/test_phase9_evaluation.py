from __future__ import annotations

import numpy as np
import pytest

from gold_forecasting.phase9.evaluation import evaluate_future_path
from gold_forecasting.phase9.normalization import (
    aggregate_target_array,
    path_target_array,
)
from gold_forecasting.phase9.targets import build_future_path_targets
from tests.test_phase9_targets import _candles, _predictions


def test_exact_path_quantiles_produce_zero_median_error_and_full_coverage() -> None:
    labels = build_future_path_targets(
        _candles(),
        _predictions(_candles(), [10, 20, 30]),
    ).labels
    path = path_target_array(labels)
    aggregate = aggregate_target_array(labels)
    path_quantiles = np.stack((path - 1.0, path, path + 1.0), axis=-1)
    aggregate_quantiles = np.stack(
        (aggregate - 1.0, aggregate, aggregate + 1.0),
        axis=-1,
    )

    metrics = evaluate_future_path(
        labels,
        path_quantiles,
        aggregate_quantiles,
        clip_log_bps=5_000.0,
    )

    assert metrics["per_step"]["1"]["gap_log_bps"]["median_mae"] == 0.0
    assert metrics["per_step"]["1"]["gap_log_bps"]["q10_q90_coverage"] == 1.0
    assert metrics["aggregate"]["body_log_bps"]["median_mae"] == 0.0
    assert metrics["cumulative_15m_return_mae_bps"] == pytest.approx(0.0, abs=1e-10)
    assert all(
        value == pytest.approx(0.0, abs=1e-9)
        for value in metrics["reconstructed_ohlc_mae_bps"].values()
    )
