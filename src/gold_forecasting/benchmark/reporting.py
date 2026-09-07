"""Human-readable development results with explicit execution limitations."""

from typing import Any


def render_benchmark_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-6 development benchmark",
        "",
        f"Protocol: `{summary['protocol']}`. Data: `{summary['data_version']}`.",
        "",
        "Scope: "
        + (
            "full frozen comparison"
            if summary["full_protocol_scope"]
            else "reduced exploratory run; does not complete phase 6"
        )
        + ".",
        "",
        "These are development comparisons, not an independent profitability claim. "
        "2024 was already inspected during MVP research. The final 2025 holdout remains closed.",
        "",
        "Every candidate uses identical records within a horizon/fold. Training and policy "
        "selection precede outer evaluation; calibration quarters remain unused. Probabilities "
        "are uncalibrated. Naive one-hot references express artificial certainty.",
        "",
        "Net values below SUM arithmetic P&L bps on a fixed entry-bid notional across folds. "
        "They are NOT account-return percentages or compounded returns. Historical asks are "
        "3-bps spread proxies; stress uses 4.5 bps with identical signals and 0.5 bps slippage "
        "per side. Realized-exit drawdown excludes intratrade mark-to-market losses.",
        "",
        "A cash policy may legitimately produce zero trades. That is not demonstrated "
        "trading profitability. No model is activated for trading by this benchmark.",
        "",
    ]
    for horizon, result in summary["horizons"].items():
        coverage = result["label_coverage"]
        lines.extend(
            [
                f"## Horizon {horizon} minutes",
                "",
                f"Eligible labels: {coverage['eligible']:,}/{coverage['candidates']:,}; "
                f"missing/incomplete paths dropped: {coverage['dropped_missing_path']:,}.",
                "",
                f"Decision: `{result['decision']}`; research reference: "
                f"`{result['research_champion']}`; promotion eligible for review: "
                f"{', '.join(result['promotion_eligible_for_review']) or 'none'}.",
                "",
                "| Model | Mean macro-F1 | Worst macro-F1 | Mean Brier | Total net bps | "
                "Stress net bps | Trades | Worst-fold net bps |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for family, metrics in result["models"].items():
            lines.append(
                f"| {family} | {metrics['macro_f1']['mean']:.4f} | "
                f"{metrics['macro_f1']['worst']:.4f} | {metrics['brier']['mean']:.4f} | "
                f"{sum(metrics['net_bps']['folds']):.2f} | "
                f"{sum(metrics['stress_net_bps']['folds']):.2f} | "
                f"{sum(metrics['trades']['folds']):.0f} | {metrics['net_bps']['worst']:.2f} |"
            )
        lines.extend(["", "Fold order: " + ", ".join(map(str, summary["test_years"])) + ".", ""])
        for family in ("reference", "logistic", "ridge", "xgboost"):
            metrics = result["models"][family]
            values = "; ".join(
                f"{year}: {net:.2f} bps / {trades} trades"
                for year, net, trades in zip(
                    summary["test_years"],
                    metrics["net_bps"]["folds"],
                    metrics["trades"]["folds"],
                    strict=True,
                )
            )
            lines.append(f"- {family}: {values}.")
        lines.append("")
    lines.extend(
        [
            "## Artifact map",
            "",
            "- `schedule.json`: frozen time blocks and common embargo.",
            "- `horizon_H/model_table.parquet`: causal features and complete-path targets.",
            "- `horizon_H/test_YEAR/split_audit.json`: identical train/test sample hashes.",
            "- `.../MODEL/selection.json`: all inner candidate metrics and frozen decision policy.",
            "- `.../MODEL/outer_predictions.parquet`: ordered predictions and source lineage.",
            "- `.../MODEL/evaluation.json`: probability, error, session and execution metrics.",
            "- `completion.json`: inventory and hashes; only a succeeded run is valid.",
            "",
        ]
    )
    return "\n".join(lines)
