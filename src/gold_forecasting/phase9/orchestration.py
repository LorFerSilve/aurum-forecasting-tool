"""Frozen nested walk-forward orchestration for the phase-9 challengers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Literal

import numpy as np
import pandas as pd

from gold_forecasting.backtesting.v1 import run_backtest_v1
from gold_forecasting.benchmark.models import prediction_records
from gold_forecasting.benchmark.pipeline import (
    BASE_COSTS,
    STRESS_COSTS,
    _audit_records,
    _metrics,
    select_policy,
)
from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.evaluation.walk_forward import (
    WalkForwardFold,
    make_walk_forward_folds,
    select_block,
    validate_development_frame,
)
from gold_forecasting.phase8.sequences import Phase8SequenceBuildResult
from gold_forecasting.phase9.config import Phase9Config
from gold_forecasting.phase9.evaluation import evaluate_future_path
from gold_forecasting.phase9.training import (
    Phase9Prediction,
    Phase9TrainingResult,
    predict_phase9_model,
    train_phase9_model,
)

GAP_MINUTES = 181
Phase9Variant = Literal["direct", "recursive"]


class Phase9OrchestrationError(ValueError):
    """Raised when folds, rows, variants, or sample universes are inconsistent."""


@dataclass(frozen=True, slots=True)
class Phase9VariantFoldResult:
    variant: Phase9Variant
    summary: dict[str, Any]
    outer_records: pd.DataFrame
    inner_records: dict[str, pd.DataFrame]
    outer_path_quantiles: np.ndarray
    outer_aggregate_quantiles: np.ndarray
    checkpoints: dict[str, Phase9TrainingResult]


@dataclass(frozen=True, slots=True)
class Phase9FoldComparison:
    fold: str
    split_audit: dict[str, Any]
    variants: dict[str, Phase9VariantFoldResult]
    comparison: dict[str, Any]


def _rows(frame: pd.DataFrame) -> np.ndarray:
    return frame.index.to_numpy(dtype=np.int64)


def build_phase9_schedule(
    config: Phase9Config,
) -> tuple[WalkForwardFold, ...]:
    folds = make_walk_forward_folds(
        config.test_years,
        start_year=2020,
    )
    if tuple(int(fold.test.start.year) for fold in folds) != config.test_years:
        raise Phase9OrchestrationError(
            "phase-9 walk-forward schedule differs from the frozen outer years"
        )
    return folds


def validate_phase9_alignment(
    table: pd.DataFrame,
    sequences: Phase8SequenceBuildResult,
) -> None:
    """Require one immutable, row-aligned sample universe before any fitting."""

    validate_development_frame(table)
    if len(table) != len(sequences.prediction_times):
        raise Phase9OrchestrationError(
            "phase-9 table and sequence store have different row counts"
        )
    if table["sample_id"].duplicated().any():
        raise Phase9OrchestrationError(
            "phase-9 sample ids must be unique"
        )
    expected = table["prediction_time_utc"].reset_index(drop=True)
    actual = sequences.prediction_times.reset_index(drop=True)
    if not expected.equals(actual):
        raise Phase9OrchestrationError(
            "phase-9 sequence rows are not aligned to model-table prediction times"
        )
    availability = np.column_stack(
        [
            sequences.by_timeframe[name].available
            for name in sequences.by_timeframe
        ]
    ).astype(bool)
    if len(availability) != len(table) or (~availability).all(axis=1).any():
        raise Phase9OrchestrationError(
            "every phase-9 sample requires at least one available sequence"
        )


def _fold_rows(
    table: pd.DataFrame,
    fold: WalkForwardFold,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = select_block(
        table,
        fold.train,
        gap_minutes=GAP_MINUTES,
    )
    calibration = select_block(
        table,
        fold.calibration,
        purge=False,
    )
    test = select_block(
        table,
        fold.test,
        purge=False,
    ).assign(fold=fold.name)
    if train.empty or calibration.empty or test.empty:
        raise Phase9OrchestrationError(
            f"phase-9 outer fold {fold.name} has empty required coverage"
        )
    return train, calibration, test


def _records(
    frame: pd.DataFrame,
    prediction: Phase9Prediction,
) -> pd.DataFrame:
    records = prediction_records(
        _audit_records(frame),
        prediction.probabilities,
        prediction.expected_return_bps,
    )
    records["predicted_range_bps"] = prediction.predicted_range_bps
    records["predicted_volatility_bps"] = prediction.predicted_volatility_bps
    return records


def _mean_predictions(
    predictions: list[Phase9Prediction],
) -> Phase9Prediction:
    if not predictions:
        raise Phase9OrchestrationError(
            "cannot ensemble an empty phase-9 prediction list"
        )
    rows = len(predictions[0].expected_return_bps)
    if any(len(item.expected_return_bps) != rows for item in predictions):
        raise Phase9OrchestrationError(
            "phase-9 seed predictions are not row-aligned"
        )
    probabilities = np.mean(
        [item.probabilities for item in predictions],
        axis=0,
    )
    probabilities /= probabilities.sum(
        axis=1,
        keepdims=True,
    )
    path = np.mean(
        [
            item.path_quantiles_log_bps
            for item in predictions
        ],
        axis=0,
    )
    aggregate = np.mean(
        [
            item.aggregate_quantiles_log_bps
            for item in predictions
        ],
        axis=0,
    )
    if (
        (path[..., 1] < path[..., 0]).any()
        or (path[..., 2] < path[..., 1]).any()
        or (aggregate[..., 1] < aggregate[..., 0]).any()
        or (aggregate[..., 2] < aggregate[..., 1]).any()
    ):
        raise Phase9OrchestrationError(
            "seed-ensembled quantiles crossed"
        )
    names = tuple(
        predictions[0].mean_fusion_weights
    )
    return Phase9Prediction(
        probabilities=probabilities.astype(
            np.float64
        ),
        expected_return_bps=np.mean(
            [
                item.expected_return_bps
                for item in predictions
            ],
            axis=0,
        ),
        predicted_range_bps=np.mean(
            [
                item.predicted_range_bps
                for item in predictions
            ],
            axis=0,
        ),
        predicted_volatility_bps=np.mean(
            [
                item.predicted_volatility_bps
                for item in predictions
            ],
            axis=0,
        ),
        path_quantiles_log_bps=path.astype(
            np.float64
        ),
        aggregate_quantiles_log_bps=(
            aggregate.astype(np.float64)
        ),
        mean_fusion_weights={
            name: float(
                np.mean(
                    [
                        item.mean_fusion_weights[
                            name
                        ]
                        for item in predictions
                    ]
                )
            )
            for name in names
        },
        inference_seconds=float(
            sum(
                item.inference_seconds
                for item in predictions
            )
        ),
    )


def _evaluate_outer(
    test: pd.DataFrame,
    prediction: Phase9Prediction,
    policy: Any,
    config: Phase9Config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = _records(
        test,
        prediction,
    )
    base = run_backtest_v1(
        records,
        costs=BASE_COSTS,
        policy=policy,
    )
    stress = run_backtest_v1(
        records,
        costs=STRESS_COSTS,
        policy=policy,
        decision_costs=BASE_COSTS,
    )
    path = evaluate_future_path(
        test,
        prediction.path_quantiles_log_bps,
        prediction.aggregate_quantiles_log_bps,
        clip_log_bps=(
            config.reconstruction_clip_log_bps
        ),
        neutral_threshold_bps=(
            config.neutral_threshold_bps
        ),
    )
    return records, {
        "classification": _metrics(records),
        "base_backtest": base.metrics,
        "stress_backtest": stress.metrics,
        "policy": asdict(policy),
        "path": path,
        "sample_digest": sample_id_digest(
            records["sample_id"]
        ),
    }


def _evaluate_variant(
    table: pd.DataFrame,
    sequences: Phase8SequenceBuildResult,
    fold: WalkForwardFold,
    variant: Phase9Variant,
    config: Phase9Config,
) -> Phase9VariantFoldResult:
    train, calibration, test = _fold_rows(
        table,
        fold,
    )
    del calibration

    inner_by_name: dict[
        str,
        list[Phase9Prediction],
    ] = {
        inner.name: []
        for inner in fold.inner_folds
    }
    selected_epochs: dict[int, int] = {}
    checkpoints: dict[
        str,
        Phase9TrainingResult,
    ] = {}
    seed_audit: dict[str, Any] = {}

    for seed in config.seeds:
        seed_epochs: list[int] = []
        seed_inner: dict[str, Any] = {}
        for inner in fold.inner_folds:
            inner_train = select_block(
                table,
                inner.train,
                gap_minutes=GAP_MINUTES,
            )
            validation = select_block(
                table,
                inner.validation,
                purge=False,
            )
            if inner_train.empty or validation.empty:
                raise Phase9OrchestrationError(
                    "phase-9 inner fold is empty: "
                    f"{fold.name}/{inner.name}"
                )
            result = train_phase9_model(
                sequences,
                table,
                _rows(inner_train),
                _rows(validation),
                config,
                seed=seed,
                model_variant=variant,
            )
            prediction = predict_phase9_model(
                result,
                sequences,
                _rows(validation),
                config,
            )
            inner_by_name[
                inner.name
            ].append(prediction)
            seed_epochs.append(
                result.best_epoch
            )
            checkpoints[
                f"inner/{seed}/{inner.name}"
            ] = result
            seed_inner[inner.name] = {
                "best_epoch": result.best_epoch,
                "fit_seconds": result.fit_seconds,
                "parameter_count": (
                    result.parameter_count
                ),
                "optimizer_steps": (
                    result.optimizer_steps
                ),
                "amp_skipped_steps": (
                    result.amp_skipped_steps
                ),
                "device": result.device,
                "mixed_precision_used": (
                    result.mixed_precision_used
                ),
            }
        selected_epochs[seed] = max(
            1,
            round(median(seed_epochs)),
        )
        seed_audit[str(seed)] = {
            "inner": seed_inner,
            "selected_final_epoch": (
                selected_epochs[seed]
            ),
        }

    inner_records: dict[
        str,
        pd.DataFrame,
    ] = {}
    policy_inputs: list[pd.DataFrame] = []
    for inner in fold.inner_folds:
        validation = select_block(
            table,
            inner.validation,
            purge=False,
        )
        ensemble = _mean_predictions(
            inner_by_name[inner.name]
        )
        records = _records(
            validation,
            ensemble,
        )
        inner_records[inner.name] = records
        policy_inputs.append(records)

    policy, policy_audit = select_policy(
        policy_inputs,
        config.minimum_policy_trades,
    )

    final_predictions: list[
        Phase9Prediction
    ] = []
    outer_seed_audit: dict[str, Any] = {}
    for seed in config.seeds:
        result = train_phase9_model(
            sequences,
            table,
            _rows(train),
            None,
            config,
            seed=seed,
            forced_epochs=selected_epochs[seed],
            model_variant=variant,
        )
        prediction = predict_phase9_model(
            result,
            sequences,
            _rows(test),
            config,
        )
        checkpoints[
            f"final/{seed}"
        ] = result
        final_predictions.append(
            prediction
        )
        seed_records = _records(
            test,
            prediction,
        )
        outer_seed_audit[str(seed)] = {
            "classification": (
                _metrics(seed_records)
            ),
            "fit_seconds": result.fit_seconds,
            "parameter_count": (
                result.parameter_count
            ),
            "optimizer_steps": (
                result.optimizer_steps
            ),
            "amp_skipped_steps": (
                result.amp_skipped_steps
            ),
            "device": result.device,
            "mixed_precision_used": (
                result.mixed_precision_used
            ),
            "inference_seconds": (
                prediction.inference_seconds
            ),
            "fusion_weights": (
                prediction.mean_fusion_weights
            ),
        }

    ensemble = _mean_predictions(
        final_predictions
    )
    outer_records, evaluation = (
        _evaluate_outer(
            test,
            ensemble,
            policy,
            config,
        )
    )
    summary = {
        "variant": variant,
        "fold": fold.name,
        "evaluation": evaluation,
        "training": {
            "seeds": seed_audit,
            "outer_seed_metrics": (
                outer_seed_audit
            ),
            "selected_epochs": {
                str(seed): epoch
                for seed, epoch
                in selected_epochs.items()
            },
        },
        "policy_candidates": policy_audit,
        "ensemble_fusion_weights": (
            ensemble.mean_fusion_weights
        ),
    }
    return Phase9VariantFoldResult(
        variant=variant,
        summary=summary,
        outer_records=outer_records,
        inner_records=inner_records,
        outer_path_quantiles=(
            ensemble.path_quantiles_log_bps
        ),
        outer_aggregate_quantiles=(
            ensemble.aggregate_quantiles_log_bps
        ),
        checkpoints=checkpoints,
    )


def _mean_path_mae(
    result: Phase9VariantFoldResult,
) -> float:
    path = result.summary[
        "evaluation"
    ]["path"]["per_step"]
    values: list[float] = []
    for step in path.values():
        for component in step.values():
            values.append(
                float(component["median_mae"])
            )
    return float(np.mean(values))


def evaluate_phase9_fold(
    table: pd.DataFrame,
    sequences: Phase8SequenceBuildResult,
    fold: WalkForwardFold,
    config: Phase9Config,
) -> Phase9FoldComparison:
    """Evaluate direct and recursive variants on one immutable outer fold."""

    validate_phase9_alignment(
        table,
        sequences,
    )
    train, calibration, test = _fold_rows(
        table,
        fold,
    )
    split_audit: dict[str, Any] = {
        "fold": fold.name,
        "gap_minutes": GAP_MINUTES,
        "train_rows": len(train),
        "calibration_rows_reserved": (
            len(calibration)
        ),
        "test_rows": len(test),
        "train_digest": sample_id_digest(
            train["sample_id"]
        ),
        "calibration_digest": (
            sample_id_digest(
                calibration["sample_id"]
            )
        ),
        "test_digest": sample_id_digest(
            test["sample_id"]
        ),
    }

    variants: dict[
        str,
        Phase9VariantFoldResult,
    ] = {}
    for variant in config.benchmark_variants:
        variants[variant] = _evaluate_variant(
            table,
            sequences,
            fold,
            variant,
            config,
        )
    digests = {
        result.summary[
            "evaluation"
        ]["sample_digest"]
        for result in variants.values()
    }
    if digests != {
        split_audit["test_digest"]
    }:
        raise Phase9OrchestrationError(
            "phase-9 variants do not share the frozen test sample universe"
        )

    direct = variants["direct"]
    recursive = variants["recursive"]
    direct_eval = direct.summary[
        "evaluation"
    ]
    recursive_eval = recursive.summary[
        "evaluation"
    ]
    comparison = {
        "sample_digest": (
            split_audit["test_digest"]
        ),
        "direct_minus_recursive": {
            "macro_f1": float(
                direct_eval[
                    "classification"
                ]["macro_f1"]
                - recursive_eval[
                    "classification"
                ]["macro_f1"]
            ),
            "brier": float(
                direct_eval[
                    "classification"
                ]["multiclass_brier"]
                - recursive_eval[
                    "classification"
                ]["multiclass_brier"]
            ),
            "mean_path_median_mae": (
                _mean_path_mae(direct)
                - _mean_path_mae(recursive)
            ),
            "cumulative_15m_return_mae_bps": float(
                direct_eval["path"][
                    "cumulative_15m_return_mae_bps"
                ]
                - recursive_eval["path"][
                    "cumulative_15m_return_mae_bps"
                ]
            ),
        },
    }
    return Phase9FoldComparison(
        fold=fold.name,
        split_audit=split_audit,
        variants=variants,
        comparison=comparison,
    )


__all__ = [
    "GAP_MINUTES",
    "Phase9FoldComparison",
    "Phase9OrchestrationError",
    "Phase9VariantFoldResult",
    "build_phase9_schedule",
    "evaluate_phase9_fold",
    "validate_phase9_alignment",
]
