"""Pure structural checks for the Phase-9 real-data preflight."""

from __future__ import annotations

import numpy as np
import pytest

from gold_forecasting.phase9.preflight import (
    Phase9PreflightError,
    _validate_prediction_invariants,
)
from gold_forecasting.phase9.training import Phase9Prediction


def _prediction() -> Phase9Prediction:
    rows = 2
    path = np.zeros((rows, 5, 4, 3), dtype=np.float64)
    aggregate = np.zeros((rows, 4, 3), dtype=np.float64)
    path[..., 0] = -1.0
    path[..., 1] = 0.0
    path[..., 2] = 1.0
    aggregate[..., 0] = -1.0
    aggregate[..., 1] = 0.0
    aggregate[..., 2] = 1.0
    for component in (2, 3):
        path[:, :, component, :] = np.array(
            [0.0, 0.5, 1.0],
            dtype=np.float64,
        )
        aggregate[:, component, :] = np.array(
            [0.0, 0.5, 1.0],
            dtype=np.float64,
        )
    return Phase9Prediction(
        probabilities=np.array(
            [
                [0.2, 0.5, 0.3],
                [0.4, 0.1, 0.5],
            ],
            dtype=np.float64,
        ),
        expected_return_bps=np.zeros(rows),
        predicted_range_bps=np.ones(rows),
        predicted_volatility_bps=np.ones(rows),
        path_quantiles_log_bps=path,
        aggregate_quantiles_log_bps=aggregate,
        mean_fusion_weights={
            "1min": 1 / 3,
            "3min": 1 / 3,
            "15min": 1 / 3,
        },
        inference_seconds=0.1,
    )


def test_preflight_prediction_invariants_accept_valid_outputs() -> None:
    result = _validate_prediction_invariants(
        _prediction(),
        expected_rows=2,
    )

    assert result["probability_sum_max_abs_error"] == pytest.approx(0.0)
    assert result["minimum_wick_q10_log_bps"] == pytest.approx(0.0)


def test_preflight_prediction_invariants_reject_quantile_crossing() -> None:
    prediction = _prediction()
    changed = prediction.path_quantiles_log_bps.copy()
    changed[0, 0, 0, :] = np.array(
        [1.0, 0.0, 2.0],
        dtype=np.float64,
    )
    invalid = Phase9Prediction(
        probabilities=prediction.probabilities,
        expected_return_bps=prediction.expected_return_bps,
        predicted_range_bps=prediction.predicted_range_bps,
        predicted_volatility_bps=prediction.predicted_volatility_bps,
        path_quantiles_log_bps=changed,
        aggregate_quantiles_log_bps=prediction.aggregate_quantiles_log_bps,
        mean_fusion_weights=prediction.mean_fusion_weights,
        inference_seconds=prediction.inference_seconds,
    )

    with pytest.raises(
        Phase9PreflightError,
        match="crossed",
    ):
        _validate_prediction_invariants(
            invalid,
            expected_rows=2,
        )


def test_preflight_prediction_invariants_reject_negative_wick_q10() -> None:
    prediction = _prediction()
    changed = prediction.path_quantiles_log_bps.copy()
    changed[0, 0, 2, :] = np.array(
        [-0.1, 0.5, 1.0],
        dtype=np.float64,
    )
    invalid = Phase9Prediction(
        probabilities=prediction.probabilities,
        expected_return_bps=prediction.expected_return_bps,
        predicted_range_bps=prediction.predicted_range_bps,
        predicted_volatility_bps=prediction.predicted_volatility_bps,
        path_quantiles_log_bps=changed,
        aggregate_quantiles_log_bps=prediction.aggregate_quantiles_log_bps,
        mean_fusion_weights=prediction.mean_fusion_weights,
        inference_seconds=prediction.inference_seconds,
    )

    with pytest.raises(
        Phase9PreflightError,
        match="negative wick",
    ):
        _validate_prediction_invariants(
            invalid,
            expected_rows=2,
        )
