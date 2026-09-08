from __future__ import annotations

import numpy as np

from gold_forecasting.phase9.normalization import (
    aggregate_target_array,
    fit_path_target_normalizer,
    path_target_array,
)
from gold_forecasting.phase9.targets import build_future_path_targets
from tests.test_phase9_targets import _candles, _predictions


def test_path_normalizer_uses_only_explicit_training_rows() -> None:
    labels = build_future_path_targets(
        _candles(),
        _predictions(_candles(), [10, 20, 30, 40]),
    ).labels
    train_rows = np.array([0, 1], dtype=np.int64)
    original = fit_path_target_normalizer(labels, train_rows)

    mutated = labels.copy()
    target_columns = [
        column
        for column in mutated.columns
        if "path_step_" in column and column.endswith("_log_bps")
    ]
    target_columns += [
        column
        for column in mutated.columns
        if column.startswith("path_15m_") and column.endswith("_log_bps")
    ]
    mutated.loc[2:, target_columns] *= 1_000.0
    rebuilt = fit_path_target_normalizer(mutated, train_rows)

    np.testing.assert_array_equal(rebuilt.path_scales, original.path_scales)
    np.testing.assert_array_equal(
        rebuilt.aggregate_scales,
        original.aggregate_scales,
    )
    assert rebuilt.cumulative_return_scale == original.cumulative_return_scale


def test_path_normalizer_round_trips_targets() -> None:
    labels = build_future_path_targets(
        _candles(),
        _predictions(_candles(), [10, 20, 30, 40]),
    ).labels
    normalizer = fit_path_target_normalizer(
        labels,
        np.arange(len(labels), dtype=np.int64),
    )
    path = path_target_array(labels)
    aggregate = aggregate_target_array(labels)

    np.testing.assert_allclose(
        normalizer.inverse_path(normalizer.transform_path(path)),
        path,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        normalizer.inverse_aggregate(normalizer.transform_aggregate(aggregate)),
        aggregate,
        rtol=1e-6,
        atol=1e-6,
    )
