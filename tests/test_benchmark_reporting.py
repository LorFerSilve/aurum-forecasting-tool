"""Reporting must preserve scope and the economic meaning of development metrics."""

from __future__ import annotations

from typing import Any

import pytest

from gold_forecasting.benchmark.config import BenchmarkConfig
from gold_forecasting.benchmark.reporting import render_benchmark_report


def _fold_metric(values: list[float]) -> dict[str, Any]:
    return {"folds": values, "mean": sum(values) / len(values), "worst": min(values)}


def _summary(*, full_protocol_scope: bool = True) -> dict[str, Any]:
    models = {
        name: {
            "macro_f1": _fold_metric([0.3, 0.4, 0.5]),
            "brier": _fold_metric([0.5, 0.6, 0.7]),
            "net_bps": _fold_metric([10.0, -25.0, 60.0]),
            "stress_net_bps": _fold_metric([5.0, -35.0, 45.0]),
            "trades": _fold_metric([2, 5, 3]),
        }
        for name in ("reference", "logistic", "ridge", "xgboost")
    }
    for key in ("net_bps", "stress_net_bps", "trades"):
        models["ridge"][key] = _fold_metric([0, 0, 0])
    return {
        "protocol": "phase6-v1",
        "data_version": "sha256:synthetic",
        "full_protocol_scope": full_protocol_scope,
        "test_years": [2022, 2023, 2024],
        "horizons": {
            "15": {
                "label_coverage": {
                    "eligible": 900,
                    "candidates": 1000,
                    "dropped_missing_path": 100,
                },
                "decision": "retain_research_reference_no_promotion",
                "research_champion": "reference",
                "promotion_eligible_for_review": [],
                "models": models,
            }
        },
    }


def test_default_config_has_full_frozen_scope() -> None:
    assert BenchmarkConfig().full_protocol_scope
    assert BenchmarkConfig(output_directory="reports/separate_benchmark").full_protocol_scope


@pytest.mark.parametrize(
    "override",
    [
        {"horizons": (15,)},
        {"test_years": (2024,)},
        {"seed": 1},
        {"xgb_max_rounds": 60},
        {"xgb_early_stopping_rounds": 5},
        {"minimum_policy_trades": 1},
    ],
)
def test_changed_experimental_scope_cannot_claim_full_protocol(override: dict[str, Any]) -> None:
    config = BenchmarkConfig.model_validate(override)
    assert not config.full_protocol_scope


@pytest.mark.parametrize("full_protocol_scope", [True, False])
def test_report_labels_full_and_reduced_experiments(full_protocol_scope: bool) -> None:
    report = render_benchmark_report(_summary(full_protocol_scope=full_protocol_scope))

    if full_protocol_scope:
        assert "Scope: full frozen comparison." in report
        assert "does not complete phase 6" not in report
    else:
        assert "Scope: reduced exploratory run; does not complete phase 6." in report
        assert "Scope: full frozen comparison." not in report
    assert "development comparisons, not an independent profitability claim" in report
    assert "2024 was already inspected" in report
    assert "final 2025 holdout remains closed" in report


def test_report_uses_summed_fold_pnl_and_trade_counts() -> None:
    report = render_benchmark_report(_summary())
    row = next(line for line in report.splitlines() if line.startswith("| logistic |"))
    cells = [cell.strip() for cell in row.strip("|").split("|")]

    assert cells == ["logistic", "0.4000", "0.3000", "0.6000", "45.00", "15.00", "10", "-25.00"]
    assert "Total net bps" in report
    assert "SUM arithmetic P&L bps on a fixed entry-bid notional across folds" in report
    assert "NOT account-return percentages or compounded returns" in report


def test_report_does_not_present_cash_as_proven_trading_profitability() -> None:
    report = render_benchmark_report(_summary())
    row = next(line for line in report.splitlines() if line.startswith("| ridge |"))
    cells = [cell.strip() for cell in row.strip("|").split("|")]

    assert cells[4:] == ["0.00", "0.00", "0", "0.00"]
    assert "A cash policy may legitimately produce zero trades" in report
    assert "That is not demonstrated trading profitability" in report
    assert "No model is activated for trading" in report
    assert "promotion eligible for review: none" in report


def test_report_preserves_fold_order_and_discloses_execution_limits() -> None:
    report = render_benchmark_report(_summary())

    assert "Fold order: 2022, 2023, 2024." in report
    assert (
        "- logistic: 2022: 10.00 bps / 2 trades; 2023: -25.00 bps / 5 trades; "
        "2024: 60.00 bps / 3 trades."
    ) in report
    assert "Historical asks are 3-bps spread proxies" in report
    assert "Realized-exit drawdown excludes intratrade mark-to-market losses" in report
    assert "Probabilities are uncalibrated" in report


def test_report_rejects_mismatched_fold_inventory() -> None:
    summary = _summary()
    summary["test_years"] = [2022, 2023]

    with pytest.raises(ValueError, match="zip"):
        render_benchmark_report(summary)
