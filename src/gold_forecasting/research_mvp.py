"""End-to-end phase-4 research evaluation and artifact generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gold_forecasting.artifacts import (
    write_json_atomic,
    write_parquet_atomic,
    write_text_atomic,
)
from gold_forecasting.backtesting import BacktestResult, run_mvp_backtest
from gold_forecasting.config import load_project_config
from gold_forecasting.datasets import load_feature_catalog, load_mvp_model_table
from gold_forecasting.evaluation import (
    EvaluationReport,
    ModelEvaluationInput,
    build_evaluation_report,
    write_evaluation_report,
)
from gold_forecasting.inference import PredictionRecord
from gold_forecasting.models.baselines import build_mvp_baseline_predictions
from gold_forecasting.training import TrainingResult, train_mvp_model


class MVPAnalysisError(RuntimeError):
    """Raised when comparable test evaluation artifacts cannot be produced."""


@dataclass(frozen=True, slots=True)
class MVPAnalysisResult:
    training: TrainingResult
    evaluation: EvaluationReport
    backtest: BacktestResult
    stress_backtest: BacktestResult
    example_prediction: PredictionRecord
    output_directory: Path
    test_rows: int


def _write_selection(result: TrainingResult, path: Path) -> None:
    selection = result.selection
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "selection_data": "validation_only",
            "final_fit": "train_only",
            "primary_metric": "macro_f1",
            "tiebreak_metric": "log_loss",
            "selected": {
                "c": selection.selected_candidate.c,
                "class_weight": selection.selected_candidate.class_weight,
            },
            "train_rows": selection.train_row_count,
            "validation_rows": selection.validation_row_count,
            "feature_count": selection.feature_count,
            "candidates": [
                {
                    "c": score.candidate.c,
                    "class_weight": score.candidate.class_weight,
                    "validation_macro_f1": score.validation_macro_f1,
                    "validation_log_loss": score.validation_log_loss,
                }
                for score in selection.candidate_scores
            ],
        },
    )


def _write_evaluation_tables(report: EvaluationReport, directory: Path) -> None:
    comparison_rows: list[dict[str, object]] = []
    hourly_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    for model in report.models:
        comparison_rows.append({"model": model.model_name, **model.metrics.to_dict()})
        hourly_rows.extend(
            {"model": model.model_name, **row.to_dict()} for row in model.per_utc_hour
        )
        reliability_rows.extend(
            {"model": model.model_name, **row.to_dict()} for row in model.reliability
        )
    write_text_atomic(
        directory / "model_comparison.csv",
        pd.DataFrame(comparison_rows).to_csv(index=False),
    )
    write_text_atomic(
        directory / "hourly_metrics.csv",
        pd.DataFrame(hourly_rows).to_csv(index=False),
    )
    write_text_atomic(
        directory / "reliability_buckets.csv",
        pd.DataFrame(reliability_rows).to_csv(index=False),
    )


def _backtest_markdown(
    base: BacktestResult,
    stress: BacktestResult,
) -> str:
    lines = [
        "# MVP backtest",
        "",
        "> Research simulation on bid-only historical data; not a performance promise.",
        "",
        (
            "| Scenario | Cost (bps) | Trades | Hit rate | Gross (bps) | "
            "Net (bps) | Drawdown (bps) | Turnover |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in (("base", base), ("stress", stress)):
        metrics = result.metrics
        hit_rate = "n/a" if metrics.hit_rate is None else f"{metrics.hit_rate:.6f}"
        lines.append(
            f"| {name} | {result.round_trip_cost_bps:.3f} | "
            f"{metrics.executed_trade_count} | {hit_rate} | "
            f"{metrics.cumulative_gross_return_bps:.6f} | "
            f"{metrics.cumulative_net_return_bps:.6f} | "
            f"{metrics.max_drawdown_bps:.6f} | {metrics.turnover:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Confidence threshold: `{base.confidence_threshold:.2f}`.",
            "Position policy: `single_non_overlapping`.",
            "",
        ]
    )
    return "\n".join(lines)


def _prediction_frame(
    test: pd.DataFrame,
    probabilities: NDArray[np.float64],
    classes: Sequence[str],
) -> pd.DataFrame:
    records = test.loc[
        :,
        [
            "instrument",
            "prediction_time_utc",
            "entry_time_utc",
            "label_end_time_utc",
            "future_return_bps",
        ],
    ].copy()
    records["predicted_class"] = list(classes)
    records["p_down"] = probabilities[:, 0]
    records["p_neutral"] = probabilities[:, 1]
    records["p_up"] = probabilities[:, 2]
    return records


def run_mvp_analysis(
    config_path: str | Path,
    output_directory: str | Path,
) -> MVPAnalysisResult:
    """Train, evaluate, backtest, persist, and emit one unseen prediction."""

    config = load_project_config(config_path)
    output = Path(output_directory)
    training = train_mvp_model(config.config_path)
    table, _ = load_mvp_model_table(config.config_path)
    catalog = load_feature_catalog(config.config_path)
    train = table.loc[table["split"].eq("train")].reset_index(drop=True)
    test = table.loc[table["split"].eq("test")].reset_index(drop=True)
    if train.empty or test.empty:
        raise MVPAnalysisError("model table must contain non-empty train and test splits")

    logistic = training.bundle.predict(test)
    baseline_batches = build_mvp_baseline_predictions(
        train["target_class"],
        test.loc[:, list(catalog.feature_names)],
        momentum_threshold_bps=config.labels.classes.total_threshold_bps,
    )
    sample_ids = tuple(test["sample_id"].astype(str))
    model_inputs = [
        ModelEvaluationInput(
            model_name=name,
            sample_ids=sample_ids,
            predictions=batch,
        )
        for name, batch in baseline_batches.items()
    ]
    model_inputs.append(
        ModelEvaluationInput(
            model_name="logistic_regression",
            sample_ids=sample_ids,
            predictions=logistic,
        )
    )
    evaluation = build_evaluation_report(
        test["target_class"],
        test["prediction_time_utc"],
        model_inputs,
        reliability_bucket_count=config.model.evaluation.reliability_buckets,
    )

    training_directory = output / "training"
    evaluation_directory = output / "evaluation"
    backtest_directory = output / "backtest"
    prediction_directory = output / "prediction"
    _write_selection(training, training_directory / "selection.json")
    write_evaluation_report(
        evaluation,
        json_path=evaluation_directory / "model_comparison.json",
        markdown_path=evaluation_directory / "model_comparison.md",
    )
    _write_evaluation_tables(evaluation, evaluation_directory)

    records = _prediction_frame(test, logistic.probabilities, logistic.predicted_class)
    threshold = config.backtest.backtest.confidence_threshold
    notional = config.backtest.backtest.normalized_notional
    base_backtest = run_mvp_backtest(
        records,
        confidence_threshold=threshold,
        round_trip_cost_bps=config.costs.cost_model.total_round_trip_bps,
        normalized_notional=notional,
    )
    stress_backtest = run_mvp_backtest(
        records,
        confidence_threshold=threshold,
        round_trip_cost_bps=config.costs.stress_model.total_round_trip_bps,
        normalized_notional=notional,
    )
    write_parquet_atomic(backtest_directory / "trades.parquet", base_backtest.trades)
    write_parquet_atomic(backtest_directory / "decisions.parquet", base_backtest.decisions)
    write_json_atomic(
        backtest_directory / "summary.json",
        {
            "schema_version": 1,
            "position_policy": base_backtest.position_policy,
            "confidence_threshold": threshold,
            "base": {
                "round_trip_cost_bps": base_backtest.round_trip_cost_bps,
                "metrics": asdict(base_backtest.metrics),
            },
            "stress": {
                "round_trip_cost_bps": stress_backtest.round_trip_cost_bps,
                "metrics": asdict(stress_backtest.metrics),
            },
        },
    )
    write_text_atomic(
        backtest_directory / "summary.md",
        _backtest_markdown(base_backtest, stress_backtest),
    )

    example_row = test.iloc[-1]
    example_probabilities = logistic.probabilities[-1]
    example = PredictionRecord(
        prediction_time_utc=example_row["prediction_time_utc"],
        instrument=str(example_row["instrument"]),
        horizon_minutes=config.labels.prediction.primary_horizon_minutes,
        predicted_class=logistic.predicted_class[-1],
        p_down=float(example_probabilities[0]),
        p_neutral=float(example_probabilities[1]),
        p_up=float(example_probabilities[2]),
        model_version=training.bundle.model_version,
        data_version=training.bundle.data_version,
        calibration_status="preliminary",
    )
    write_json_atomic(
        prediction_directory / "example_prediction.json",
        example.model_dump(mode="json"),
    )
    return MVPAnalysisResult(
        training=training,
        evaluation=evaluation,
        backtest=base_backtest,
        stress_backtest=stress_backtest,
        example_prediction=example,
        output_directory=output,
        test_rows=len(test),
    )


__all__ = ["MVPAnalysisError", "MVPAnalysisResult", "run_mvp_analysis"]
