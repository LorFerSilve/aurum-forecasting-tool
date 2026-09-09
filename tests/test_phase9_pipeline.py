"""Run-level aggregation and safety tests for the formal Phase-9 pipeline."""

from __future__ import annotations

from typing import Any

import pytest

import gold_forecasting.phase9.pipeline as pipeline
from gold_forecasting.phase9.pipeline import (
    Phase9PipelineError,
    _aggregate_variant,
    _comparison_to_reference,
)


def _evaluation(
    *,
    macro_f1: float,
    brier: float,
    path_mae: float,
    net_bps: float,
) -> dict[str, Any]:
    component = {
        "median_mae": path_mae,
        "q10_q90_coverage": 0.8,
        "mean_interval_width": 2.0,
    }
    path = {
        "per_step": {
            str(step): {
                name: dict(component)
                for name in (
                    "gap_log_bps",
                    "body_log_bps",
                    "upper_wick_log_bps",
                    "lower_wick_log_bps",
                )
            }
            for step in range(1, 6)
        },
        "aggregate": {
            name: dict(component)
            for name in (
                "gap_log_bps",
                "body_log_bps",
                "upper_wick_log_bps",
                "lower_wick_log_bps",
            )
        },
        "reconstructed_ohlc_mae_bps": {
            "open": path_mae,
            "high": path_mae,
            "low": path_mae,
            "close": path_mae,
        },
        "aggregate_reconstructed_ohlc_mae_bps": {
            "open": path_mae,
            "high": path_mae,
            "low": path_mae,
            "close": path_mae,
        },
        "aggregate_reconstructed_range_mae_bps": path_mae,
        "path_vs_direct_aggregate_ohlc_mae_bps": {
            "open": path_mae,
            "high": path_mae,
            "low": path_mae,
            "close": path_mae,
        },
        "path_vs_direct_aggregate_range_mae_bps": path_mae,
        "cumulative_15m_return_mae_bps": path_mae,
        "path_implied_direction": {
            "accuracy": 0.7,
            "macro_f1": 0.6,
            "neutral_threshold_bps": 6.0,
        },
    }
    classification = {
        "accuracy": 0.6,
        "balanced_accuracy": 0.5,
        "macro_f1": macro_f1,
        "multiclass_brier": brier,
        "log_loss": 0.9,
        "return_mae_bps": 5.0,
        "expected_calibration_error": 0.1,
    }
    backtest = {
        "cumulative_net_return_bps": net_bps,
        "executed_trade_count": 20,
        "max_drawdown_bps": 4.0,
        "exposure_fraction": 0.1,
    }
    return {
        "classification": classification,
        "base_backtest": backtest,
        "stress_backtest": {
            **backtest,
            "cumulative_net_return_bps": net_bps - 1.0,
        },
        "path": path,
    }


def test_aggregate_variant_preserves_mean_and_worst_fold() -> None:
    aggregate = _aggregate_variant(
        [
            _evaluation(
                macro_f1=0.40,
                brier=0.50,
                path_mae=2.0,
                net_bps=3.0,
            ),
            _evaluation(
                macro_f1=0.50,
                brier=0.60,
                path_mae=4.0,
                net_bps=-1.0,
            ),
        ]
    )

    assert aggregate["macro_f1"]["mean"] == pytest.approx(0.45)
    assert aggregate["macro_f1"]["worst"] == pytest.approx(0.40)
    assert aggregate["brier"]["worst"] == pytest.approx(0.60)
    assert aggregate["cumulative_15m_return_mae_bps"]["mean"] == pytest.approx(
        3.0
    )
    assert aggregate["base_net_bps"]["worst"] == pytest.approx(-1.0)
    assert aggregate["path_implied_direction_macro_f1"]["mean"] == pytest.approx(
        0.6
    )


def test_reference_comparison_uses_candidate_minus_reference() -> None:
    candidate = _aggregate_variant(
        [
            _evaluation(
                macro_f1=0.50,
                brier=0.40,
                path_mae=2.0,
                net_bps=2.0,
            )
        ]
    )
    reference = {
        name: dict(value)
        for name, value in candidate.items()
        if name
        in {
            "macro_f1",
            "brier",
            "log_loss",
            "return_mae_bps",
            "base_net_bps",
            "stress_net_bps",
        }
    }
    reference["macro_f1"]["mean"] = 0.45
    reference["macro_f1"]["worst"] = 0.45
    reference["brier"]["mean"] = 0.45

    comparison = _comparison_to_reference(
        candidate,
        reference,
    )

    assert comparison["macro_f1_mean_delta"] == pytest.approx(0.05)
    assert comparison["brier_mean_delta"] == pytest.approx(-0.05)


def test_formal_run_never_opens_registry_when_preparation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = False

    def fail_preparation(_: object) -> object:
        raise Phase9PipelineError("preflight failed")

    def unexpected_start(*args: object, **kwargs: object) -> object:
        nonlocal opened
        opened = True
        raise AssertionError("formal registry must not open")

    monkeypatch.setattr(
        pipeline,
        "_prepare_formal_phase9",
        fail_preparation,
    )
    monkeypatch.setattr(
        pipeline.RunRegistry,
        "start_run",
        unexpected_start,
    )

    with pytest.raises(
        Phase9PipelineError,
        match="preflight failed",
    ):
        pipeline.run_phase9("configs/phase9.yaml")

    assert opened is False
