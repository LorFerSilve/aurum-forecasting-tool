"""Distributional evaluation for phase-9 future-candle paths."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gold_forecasting.phase9.normalization import (
    aggregate_target_array,
    path_target_array,
)
from gold_forecasting.phase9.targets import (
    OHLC_COMPONENTS,
    PATH_COMPONENTS,
    PATH_STEPS,
    aggregate_path_ohlc,
    reconstruct_path,
)


class Phase9EvaluationError(ValueError):
    """Raised when path predictions cannot be evaluated safely."""


def _quantile_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    if prediction.shape[:-1] != target.shape or prediction.shape[-1] != 3:
        raise Phase9EvaluationError("prediction/target quantile shapes do not align")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise Phase9EvaluationError("path evaluation inputs must be finite")
    q10 = prediction[..., 0]
    q50 = prediction[..., 1]
    q90 = prediction[..., 2]
    if (q50 < q10).any() or (q90 < q50).any():
        raise Phase9EvaluationError("predicted quantiles cross")
    return {
        "median_mae": float(np.mean(np.abs(q50 - target))),
        "q10_q90_coverage": float(np.mean((target >= q10) & (target <= q90))),
        "mean_interval_width": float(np.mean(q90 - q10)),
    }


def evaluate_future_path(
    labels: pd.DataFrame,
    path_quantiles: np.ndarray,
    aggregate_quantiles: np.ndarray,
    *,
    clip_log_bps: float,
    neutral_threshold_bps: float = 6.0,
) -> dict[str, Any]:
    """Evaluate physical-unit q10/q50/q90 path and aggregate predictions."""

    path_target = path_target_array(labels)
    aggregate_target = aggregate_target_array(labels)
    path_prediction = np.asarray(path_quantiles, dtype=np.float64)
    aggregate_prediction = np.asarray(aggregate_quantiles, dtype=np.float64)
    if path_prediction.shape != (*path_target.shape, 3):
        raise Phase9EvaluationError("path quantiles have an invalid shape")
    if aggregate_prediction.shape != (*aggregate_target.shape, 3):
        raise Phase9EvaluationError("aggregate quantiles have an invalid shape")
    if (
        not np.isfinite(neutral_threshold_bps)
        or neutral_threshold_bps <= 0.0
    ):
        raise Phase9EvaluationError(
            "neutral threshold must be finite and positive"
        )

    per_step: dict[str, Any] = {}
    for step in range(PATH_STEPS):
        per_component: dict[str, Any] = {}
        for component_index, component in enumerate(PATH_COMPONENTS):
            per_component[component] = _quantile_metrics(
                path_prediction[:, step, component_index, :],
                path_target[:, step, component_index],
            )
        per_step[str(step + 1)] = per_component

    aggregate: dict[str, Any] = {}
    for component_index, component in enumerate(PATH_COMPONENTS):
        aggregate[component] = _quantile_metrics(
            aggregate_prediction[:, component_index, :],
            aggregate_target[:, component_index],
        )

    median_representation = path_prediction[..., 1]
    anchor = labels["path_anchor_close"].to_numpy(dtype=np.float64)
    reconstructed = reconstruct_path(
        anchor,
        median_representation,
        clip_log_bps=clip_log_bps,
    )
    true_ohlc = np.empty((len(labels), PATH_STEPS, 4), dtype=np.float64)
    for step in range(PATH_STEPS):
        for component_index, component in enumerate(OHLC_COMPONENTS):
            true_ohlc[:, step, component_index] = labels[
                f"path_step_{step + 1}_bid_{component}"
            ].to_numpy(dtype=np.float64)
    anchor_scale = anchor[:, None, None]
    ohlc_error_bps = (
        10_000.0
        * np.abs(reconstructed - true_ohlc)
        / anchor_scale
    )
    ohlc_mae = {
        component: float(ohlc_error_bps[:, :, index].mean())
        for index, component in enumerate(OHLC_COMPONENTS)
    }

    reconstructed_aggregate = aggregate_path_ohlc(reconstructed)
    true_aggregate = np.column_stack(
        [
            labels[f"path_15m_bid_{component}"].to_numpy(dtype=np.float64)
            for component in OHLC_COMPONENTS
        ]
    )
    aggregate_ohlc_error_bps = (
        10_000.0
        * np.abs(reconstructed_aggregate - true_aggregate)
        / anchor[:, None]
    )
    aggregate_ohlc_mae = {
        component: float(aggregate_ohlc_error_bps[:, index].mean())
        for index, component in enumerate(OHLC_COMPONENTS)
    }
    reconstructed_range = (
        reconstructed_aggregate[:, 1] - reconstructed_aggregate[:, 2]
    )
    true_range = true_aggregate[:, 1] - true_aggregate[:, 2]
    aggregate_range_mae = float(
        np.mean(
            10_000.0
            * np.abs(reconstructed_range - true_range)
            / anchor
        )
    )

    direct_q50 = aggregate_prediction[..., 1]
    gap = np.clip(direct_q50[:, 0], -clip_log_bps, clip_log_bps)
    body = np.clip(direct_q50[:, 1], -clip_log_bps, clip_log_bps)
    upper = np.clip(direct_q50[:, 2], 0.0, clip_log_bps)
    lower = np.clip(direct_q50[:, 3], 0.0, clip_log_bps)
    direct_open = anchor * np.exp(gap / 10_000.0)
    direct_close = direct_open * np.exp(body / 10_000.0)
    direct_high = np.maximum(direct_open, direct_close) * np.exp(
        upper / 10_000.0
    )
    direct_low = np.minimum(direct_open, direct_close) * np.exp(
        -lower / 10_000.0
    )
    direct_aggregate_ohlc = np.column_stack(
        (direct_open, direct_high, direct_low, direct_close)
    )
    consistency_error_bps = (
        10_000.0
        * np.abs(reconstructed_aggregate - direct_aggregate_ohlc)
        / anchor[:, None]
    )
    consistency_ohlc_mae = {
        component: float(consistency_error_bps[:, index].mean())
        for index, component in enumerate(OHLC_COMPONENTS)
    }
    direct_range = direct_high - direct_low
    consistency_range_mae = float(
        np.mean(
            10_000.0
            * np.abs(reconstructed_range - direct_range)
            / anchor
        )
    )

    true_cumulative = labels[
        "path_15m_close_return_log_bps"
    ].to_numpy(dtype=np.float64)
    path_implied_return = (
        10_000.0
        * np.log(
            reconstructed_aggregate[:, 3]
            / anchor
        )
    )
    cumulative_mae = float(
        np.mean(
            np.abs(
                path_implied_return
                - true_cumulative
            )
        )
    )
    predicted_class = np.where(
        path_implied_return < -neutral_threshold_bps,
        0,
        np.where(
            path_implied_return > neutral_threshold_bps,
            2,
            1,
        ),
    )
    true_class = np.where(
        true_cumulative < -neutral_threshold_bps,
        0,
        np.where(
            true_cumulative > neutral_threshold_bps,
            2,
            1,
        ),
    )
    path_direction_accuracy = float(
        np.mean(predicted_class == true_class)
    )
    class_f1: list[float] = []
    for class_id in (0, 1, 2):
        true_positive = int(
            np.sum(
                (predicted_class == class_id)
                & (true_class == class_id)
            )
        )
        false_positive = int(
            np.sum(
                (predicted_class == class_id)
                & (true_class != class_id)
            )
        )
        false_negative = int(
            np.sum(
                (predicted_class != class_id)
                & (true_class == class_id)
            )
        )
        denominator = (
            2 * true_positive
            + false_positive
            + false_negative
        )
        class_f1.append(
            0.0
            if denominator == 0
            else float(
                2 * true_positive
                / denominator
            )
        )
    path_direction_macro_f1 = float(
        np.mean(class_f1)
    )

    return {
        "per_step": per_step,
        "aggregate": aggregate,
        "reconstructed_ohlc_mae_bps": ohlc_mae,
        "aggregate_reconstructed_ohlc_mae_bps": aggregate_ohlc_mae,
        "aggregate_reconstructed_range_mae_bps": aggregate_range_mae,
        "path_vs_direct_aggregate_ohlc_mae_bps": consistency_ohlc_mae,
        "path_vs_direct_aggregate_range_mae_bps": consistency_range_mae,
        "cumulative_15m_return_mae_bps": cumulative_mae,
        "path_implied_direction": {
            "accuracy": path_direction_accuracy,
            "macro_f1": path_direction_macro_f1,
            "neutral_threshold_bps": neutral_threshold_bps,
        },
    }


__all__ = [
    "Phase9EvaluationError",
    "evaluate_future_path",
]
