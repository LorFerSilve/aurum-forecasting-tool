"""Guarded phase-8 neural benchmark against the frozen v0.2 price-only reference."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
import torch

from gold_forecasting.artifacts import (
    content_version,
    file_digest,
    write_json_atomic,
    write_parquet_atomic,
    write_text_atomic,
)
from gold_forecasting.benchmark.data_guard import preflight_development_inputs
from gold_forecasting.benchmark.models import prediction_records
from gold_forecasting.benchmark.pipeline import (
    _audit_records,
    _evaluate_predictions,
    _metrics,
    select_policy,
    verify_benchmark,
)
from gold_forecasting.config import load_project_config
from gold_forecasting.data_pipeline import validate_existing_mvp_data
from gold_forecasting.datasets.mvp import _load_curated
from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.evaluation.walk_forward import (
    WalkForwardFold,
    make_walk_forward_folds,
    select_block,
    validate_development_frame,
)
from gold_forecasting.features import build_mvp_features
from gold_forecasting.labels.multihorizon import build_horizon_labels
from gold_forecasting.phase8.config import Phase8Config, load_phase8_config
from gold_forecasting.phase8.reference import Phase7Reference, load_phase7_reference
from gold_forecasting.phase8.sequences import (
    Phase8SequenceBuildResult,
    build_phase8_sequences,
    subset_phase8_sequences,
)
from gold_forecasting.phase8.training import (
    NeuralPrediction,
    predict_neural_model,
    resolve_device,
    save_checkpoint,
    train_neural_model,
)
from gold_forecasting.registry import RunRegistry, get_git_code_version

GAP_MINUTES = 181


def _require_clean_code_version(code_version: str) -> None:
    if code_version in {"unavailable", "uncommitted"} or code_version.endswith("+dirty"):
        raise RuntimeError(
            "formal phase-8 runs require a clean committed Git working tree; "
            f"found {code_version!r}"
        )


def _rows(frame: pd.DataFrame) -> np.ndarray:
    return frame.index.to_numpy(dtype=np.int64)


def _prediction_records(
    frame: pd.DataFrame,
    prediction: NeuralPrediction,
) -> pd.DataFrame:
    records = prediction_records(
        _audit_records(frame),
        prediction.probabilities,
        prediction.expected_return_bps,
    )
    records["predicted_range_bps"] = prediction.predicted_range_bps
    records["predicted_volatility_bps"] = prediction.predicted_volatility_bps
    return records


def _mean_predictions(predictions: list[NeuralPrediction]) -> NeuralPrediction:
    if not predictions:
        raise ValueError("cannot ensemble an empty neural prediction list")
    rows = len(predictions[0].expected_return_bps)
    if any(len(item.expected_return_bps) != rows for item in predictions):
        raise ValueError("seed predictions are not row-aligned")
    probabilities = np.mean(
        [item.probabilities for item in predictions],
        axis=0,
    )
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    weight_names = tuple(predictions[0].mean_fusion_weights)
    return NeuralPrediction(
        probabilities=probabilities.astype(np.float64),
        expected_return_bps=np.mean(
            [item.expected_return_bps for item in predictions],
            axis=0,
        ),
        predicted_range_bps=np.mean(
            [item.predicted_range_bps for item in predictions],
            axis=0,
        ),
        predicted_volatility_bps=np.mean(
            [item.predicted_volatility_bps for item in predictions],
            axis=0,
        ),
        mean_fusion_weights={
            name: float(
                np.mean(
                    [
                        item.mean_fusion_weights[name]
                        for item in predictions
                    ]
                )
            )
            for name in weight_names
        },
        inference_seconds=float(
            sum(item.inference_seconds for item in predictions)
        ),
    )


def _continuous_metrics(records: pd.DataFrame) -> dict[str, float]:
    return {
        "range_mae_bps": float(
            (
                records["future_range_bps"]
                - records["predicted_range_bps"]
            )
            .abs()
            .mean()
        ),
        "volatility_mae_bps": float(
            (
                records["future_realized_vol_bps"]
                - records["predicted_volatility_bps"]
            )
            .abs()
            .mean()
        ),
    }


def _aggregate_scalar(
    values: list[float],
    *,
    higher: bool,
) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "worst": float(
            min(values) if higher else max(values)
        ),
        "folds": values,
    }


def _aggregate_neural(
    folds: list[dict[str, Any]],
) -> dict[str, Any]:
    specs = {
        "accuracy": ("classification", "accuracy", True),
        "balanced_accuracy": (
            "classification",
            "balanced_accuracy",
            True,
        ),
        "macro_f1": ("classification", "macro_f1", True),
        "brier": (
            "classification",
            "multiclass_brier",
            False,
        ),
        "log_loss": ("classification", "log_loss", False),
        "return_mae_bps": (
            "classification",
            "return_mae_bps",
            False,
        ),
        "calibration_error": (
            "classification",
            "expected_calibration_error",
            False,
        ),
        "net_bps": (
            "base_backtest",
            "cumulative_net_return_bps",
            True,
        ),
        "stress_net_bps": (
            "stress_backtest",
            "cumulative_net_return_bps",
            True,
        ),
        "trades": (
            "base_backtest",
            "executed_trade_count",
            True,
        ),
        "drawdown_bps": (
            "base_backtest",
            "max_drawdown_bps",
            False,
        ),
        "turnover": (
            "base_backtest",
            "turnover",
            False,
        ),
        "exposure_fraction": (
            "base_backtest",
            "exposure_fraction",
            False,
        ),
        "range_mae_bps": (
            "continuous",
            "range_mae_bps",
            False,
        ),
        "volatility_mae_bps": (
            "continuous",
            "volatility_mae_bps",
            False,
        ),
    }
    result: dict[str, Any] = {}
    for name, (section, metric, higher) in specs.items():
        result[name] = _aggregate_scalar(
            [
                float(fold[section][metric])
                for fold in folds
            ],
            higher=higher,
        )
    return result


def _predictive_gate(
    neural: dict[str, Any],
    baseline: dict[str, Any],
) -> bool:
    """Require neural skill without a probability-quality regression."""

    return bool(
        neural["macro_f1"]["mean"]
        > baseline["macro_f1"]["mean"]
        and neural["macro_f1"]["worst"]
        >= baseline["macro_f1"]["worst"]
        and neural["brier"]["mean"]
        <= baseline["brier"]["mean"]
        and neural["log_loss"]["mean"]
        <= baseline["log_loss"]["mean"]
    )


def _economic_gate(
    neural: dict[str, Any],
    baseline: dict[str, Any],
) -> bool:
    return bool(
        neural["trades"]["worst"] >= 20
        and neural["net_bps"]["worst"] > 0
        and neural["stress_net_bps"]["mean"] > 0
        and neural["net_bps"]["worst"]
        >= baseline["net_bps"]["worst"]
    )


def _compare_to_phase7(
    neural: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    predictive = _predictive_gate(neural, baseline)
    economic = predictive and _economic_gate(neural, baseline)
    deltas = {
        "macro_f1_mean": (
            neural["macro_f1"]["mean"]
            - baseline["macro_f1"]["mean"]
        ),
        "macro_f1_worst": (
            neural["macro_f1"]["worst"]
            - baseline["macro_f1"]["worst"]
        ),
        "brier_mean": (
            neural["brier"]["mean"]
            - baseline["brier"]["mean"]
        ),
        "log_loss_mean": (
            neural["log_loss"]["mean"]
            - baseline["log_loss"]["mean"]
        ),
        "return_mae_mean_bps": (
            neural["return_mae_bps"]["mean"]
            - baseline["return_mae_bps"]["mean"]
        ),
    }
    return {
        "predictive_admission": predictive,
        "economic_promotion": economic,
        "decision": (
            "review_neural_challenger"
            if predictive
            else "keep_phase7_champion"
        ),
        "trading_champion": None,
        "deltas_vs_phase7_champion": deltas,
    }


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
        raise ValueError(
            f"phase-8 outer fold {fold.name} has empty required coverage"
        )
    return train, calibration, test


def _evaluate_fold(
    table: pd.DataFrame,
    sequences: Phase8SequenceBuildResult,
    fold: WalkForwardFold,
    horizon: int,
    timeframes: tuple[str, ...],
    config: Phase8Config,
    reference: Phase7Reference,
    directory: Path,
) -> dict[str, Any]:
    train, calibration, test = _fold_rows(table, fold)
    del calibration
    test_digest = sample_id_digest(test["sample_id"])
    expected_digest = reference.test_digest(
        horizon,
        fold.name,
    )
    if test_digest != expected_digest:
        raise ValueError(
            "phase-8 test samples differ from frozen phase-7 "
            f"reference: {horizon}/{fold.name}"
        )

    inner_by_name: dict[str, list[NeuralPrediction]] = {
        inner.name: [] for inner in fold.inner_folds
    }
    selected_epochs: dict[int, int] = {}
    training_audit: dict[str, Any] = {
        "seeds": {},
        "timeframes": list(timeframes),
    }
    for seed in config.seeds:
        seed_epochs: list[int] = []
        seed_record: dict[str, Any] = {"inner": {}}
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
                raise ValueError(
                    "phase-8 inner fold is empty: "
                    f"{fold.name}/{inner.name}"
                )
            result = train_neural_model(
                sequences,
                table,
                _rows(inner_train),
                _rows(validation),
                timeframes,
                config,
                seed=seed,
            )
            prediction = predict_neural_model(
                result,
                sequences,
                _rows(validation),
                timeframes,
                config,
            )
            inner_by_name[inner.name].append(prediction)
            seed_epochs.append(result.best_epoch)
            seed_dir = (
                directory
                / "inner_models"
                / str(seed)
                / inner.name
            )
            save_checkpoint(
                result,
                seed_dir / "checkpoint.pt",
            )
            write_json_atomic(
                seed_dir / "history.json",
                {"epochs": list(result.history)},
            )
            seed_record["inner"][inner.name] = {
                "best_epoch": result.best_epoch,
                "fit_seconds": result.fit_seconds,
                "parameter_count": result.parameter_count,
                "device": result.device,
                "mixed_precision_used": (
                    result.mixed_precision_used
                ),
                "validation_macro_f1": _metrics(
                    _prediction_records(
                        validation,
                        prediction,
                    )
                )["macro_f1"],
            }
        chosen_epoch = int(
            round(median(seed_epochs))
        )
        selected_epochs[seed] = max(
            1,
            chosen_epoch,
        )
        seed_record["selected_final_epoch"] = (
            selected_epochs[seed]
        )
        training_audit["seeds"][str(seed)] = (
            seed_record
        )

    inner_records: list[pd.DataFrame] = []
    for inner in fold.inner_folds:
        validation = select_block(
            table,
            inner.validation,
            purge=False,
        )
        ensemble = _mean_predictions(
            inner_by_name[inner.name]
        )
        records = _prediction_records(
            validation,
            ensemble,
        )
        inner_records.append(records)
        write_parquet_atomic(
            directory
            / f"selected_inner_{inner.name}.parquet",
            records,
        )
    policy, policy_audit = select_policy(
        inner_records,
        config.minimum_policy_trades,
    )

    final_seed_predictions: list[
        NeuralPrediction
    ] = []
    seed_outer_metrics: dict[str, Any] = {}
    for seed in config.seeds:
        result = train_neural_model(
            sequences,
            table,
            _rows(train),
            None,
            timeframes,
            config,
            seed=seed,
            forced_epochs=selected_epochs[seed],
        )
        prediction = predict_neural_model(
            result,
            sequences,
            _rows(test),
            timeframes,
            config,
        )
        final_seed_predictions.append(prediction)
        seed_dir = (
            directory
            / "final_models"
            / str(seed)
        )
        save_checkpoint(
            result,
            seed_dir / "checkpoint.pt",
        )
        write_json_atomic(
            seed_dir / "history.json",
            {"epochs": list(result.history)},
        )
        seed_records = _prediction_records(
            test,
            prediction,
        )
        seed_outer_metrics[str(seed)] = {
            "classification": _metrics(seed_records),
            "continuous": _continuous_metrics(
                seed_records
            ),
            "fit_seconds": result.fit_seconds,
            "inference_seconds": (
                prediction.inference_seconds
            ),
            "fusion_weights": (
                prediction.mean_fusion_weights
            ),
        }

    ensemble = _mean_predictions(
        final_seed_predictions
    )
    records = _prediction_records(
        test,
        ensemble,
    )
    evaluation = _evaluate_predictions(
        records,
        policy,
        directory / "ensemble",
    )
    continuous = _continuous_metrics(records)
    write_parquet_atomic(
        directory
        / "ensemble"
        / "outer_predictions_with_aux.parquet",
        records,
    )
    write_json_atomic(
        directory / "training_audit.json",
        {
            **training_audit,
            "policy_candidates": policy_audit,
            "selected_policy": {
                "confidence_threshold": (
                    policy.confidence_threshold
                ),
                "min_expected_net_bps": (
                    policy.min_expected_net_bps
                ),
            },
            "test_digest": test_digest,
            "seed_outer_metrics": (
                seed_outer_metrics
            ),
            "ensemble_fusion_weights": (
                ensemble.mean_fusion_weights
            ),
        },
    )
    return {
        "classification": (
            evaluation["classification"]
        ),
        "base_backtest": (
            evaluation["base_backtest"]
        ),
        "stress_backtest": (
            evaluation["stress_backtest"]
        ),
        "policy": evaluation["policy"],
        "sample_digest": (
            evaluation["sample_digest"]
        ),
        "continuous": continuous,
        "seed_outer_metrics": (
            seed_outer_metrics
        ),
        "selected_epochs": {
            str(seed): selected_epochs[seed]
            for seed in config.seeds
        },
    }


def _render_summary(
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Phase 8 — compact multi-timeframe neural challenger",
        "",
        f"Protocol: `{summary['protocol']}`",
        "",
        "The final holdout remains closed. Every neural outer test sample is digest-matched",
        "to the frozen v0.2 phase-7 reference before evaluation.",
        "",
        "| Horizon | Phase-7 fallback | Predictive admission | Economic promotion |",
        "|---:|---|---|---|",
    ]
    for horizon, record in summary[
        "horizons"
    ].items():
        baseline = record["phase7_champion"]
        comparison = record["comparison"]
        lines.append(
            f"| {horizon}m | "
            f"{baseline['variant']}/{baseline['family']} | "
            f"{comparison['predictive_admission']} | "
            f"{comparison['economic_promotion']} |"
        )
    lines.extend(
        [
            "",
            "Neural promotion is development-only. No phase-8 result activates paper or live trading.",
            "Probabilities remain uncalibrated until phase 11.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_phase8(
    config_path: str | Path,
) -> Path:
    path = Path(config_path).resolve(strict=True)
    config = load_phase8_config(path)
    project = load_project_config(
        path.parent / config.data_config
    )
    root = project.config_path.parent.parent
    expected_code_version = get_git_code_version(
        root
    )
    _require_clean_code_version(
        expected_code_version
    )
    resolve_device(config.device)

    reference = load_phase7_reference(
        root,
        config.phase7_reference_directory,
        path.parent
        / config.phase7_champion_config,
    )
    preflight_development_inputs(project)
    validate_existing_mvp_data(
        project.config_path
    )

    candles: dict[str, pd.DataFrame] = {}
    manifests: dict[str, Any] = {}
    for timeframe in config.timeframes:
        frame, manifest = _load_curated(
            root,
            project,
            timeframe,
        )
        candles[timeframe] = frame
        manifests[timeframe] = manifest
    data_version = content_version(
        {
            "curated": {
                name: manifests[
                    name
                ].dataset_version
                for name in config.timeframes
            },
            "phase7_reference": (
                reference.champions.run_id
            ),
        }
    )
    registry = RunRegistry(
        root / config.output_directory,
        path,
        data_version,
        code_root=root,
    )
    record = registry.start_run()
    output = record.output_directory
    print(
        f"Phase-8 run: {output}",
        flush=True,
    )
    try:
        if (
            record.code_version
            != expected_code_version
        ):
            raise RuntimeError(
                "Git identity changed during phase-8 preflight: "
                f"{expected_code_version!r} -> "
                f"{record.code_version!r}"
            )
        write_json_atomic(
            output / "resolved_config.json",
            config.model_dump(mode="json"),
        )
        write_json_atomic(
            output / "runtime.json",
            {
                "torch": torch.__version__,
                "cuda_available": (
                    torch.cuda.is_available()
                ),
                "cuda_version": torch.version.cuda,
                "device": str(
                    resolve_device(config.device)
                ),
                "cuda_device_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            },
        )
        versions = {
            name: importlib.metadata.version(
                name
            )
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "torch",
                "pyarrow",
            )
        }
        write_json_atomic(
            output / "dependencies.json",
            versions,
        )
        write_text_atomic(
            output / "protocol.md",
            (
                root
                / "docs/research_protocol_phase8.md"
            ).read_text(encoding="utf-8"),
        )
        for source in sorted(
            (
                root
                / "src"
                / "gold_forecasting"
            ).rglob("*.py")
        ):
            write_text_atomic(
                output
                / "source_snapshot"
                / source.relative_to(
                    root / "src"
                ),
                source.read_text(
                    encoding="utf-8"
                ),
            )
        write_text_atomic(
            output / "requirements.lock",
            (
                root / "requirements.lock"
            ).read_text(encoding="utf-8"),
        )
        write_json_atomic(
            output / "source_manifests.json",
            {
                name: manifests[
                    name
                ].model_dump(mode="json")
                for name in config.timeframes
            },
        )
        write_json_atomic(
            output / "phase7_reference.json",
            {
                "run_id": (
                    reference.champions.run_id
                ),
                "code_version": (
                    reference.champions.code_version
                ),
                "champions": {
                    str(horizon): (
                        champion.model_dump(
                            mode="json"
                        )
                    )
                    for horizon, champion in (
                        reference.champions.research_champions.items()
                    )
                },
            },
        )

        anchor = build_mvp_features(
            candles["3min"],
            project.features,
        ).features
        anchor = anchor.sort_values(
            [
                "instrument",
                "source",
                "prediction_time_utc",
            ],
            kind="stable",
        ).reset_index(drop=True)
        anchor["_sequence_row"] = np.arange(
            len(anchor),
            dtype=np.int64,
        )
        sequence_store = build_phase8_sequences(
            candles,
            anchor,
            {
                name: config.sequence_lengths[
                    name
                ]
                for name in config.timeframes
            },
        )
        write_json_atomic(
            output
            / "sequence_diagnostics.json",
            sequence_store.diagnostics,
        )

        folds = make_walk_forward_folds(
            test_years=config.test_years
        )
        write_json_atomic(
            output / "schedule.json",
            {
                "gap_minutes": GAP_MINUTES,
                "folds": [
                    fold.as_record()
                    for fold in folds
                ],
            },
        )
        results: dict[str, Any] = {}
        for horizon in config.horizons:
            print(
                f"Phase 8 horizon {horizon} minutes",
                flush=True,
            )
            label_result = (
                build_horizon_labels(
                    candles["1min"],
                    anchor,
                    horizon_minutes=horizon,
                )
            )
            table = anchor.merge(
                label_result.labels,
                on=[
                    "instrument",
                    "source",
                    "prediction_time_utc",
                ],
                validate="one_to_one",
            ).sort_values(
                [
                    "instrument",
                    "source",
                    "prediction_time_utc",
                ],
                kind="stable",
            )
            table = table.reset_index(drop=True)
            horizon_sequences = (
                subset_phase8_sequences(
                    sequence_store,
                    table[
                        "_sequence_row"
                    ].to_numpy(
                        dtype=np.int64
                    ),
                )
            )
            table = table.drop(
                columns="_sequence_row"
            )
            table["sample_id"] = [
                hashlib.sha256(
                    (
                        f"{instrument}|{source}|"
                        f"{stamp.isoformat()}|"
                        f"{horizon}"
                    ).encode()
                ).hexdigest()
                for instrument, source, stamp in (
                    table[
                        [
                            "instrument",
                            "source",
                            "prediction_time_utc",
                        ]
                    ].itertuples(
                        index=False,
                        name=None,
                    )
                )
            ]
            validate_development_frame(table)
            horizon_directory = (
                output
                / f"horizon_{horizon}"
            )
            write_parquet_atomic(
                horizon_directory
                / "model_table.parquet",
                table,
            )

            selected_timeframes = (
                config.horizon_timeframes[
                    horizon
                ]
            )
            availability = np.column_stack(
                [
                    horizon_sequences.by_timeframe[
                        name
                    ].available
                    for name in selected_timeframes
                ]
            )
            if (
                (~availability)
                .all(axis=1)
                .any()
            ):
                raise ValueError(
                    f"horizon {horizon} has rows "
                    "with no available configured sequence"
                )
            write_json_atomic(
                horizon_directory
                / "sequence_coverage.json",
                {
                    "rows": len(table),
                    "timeframes": list(
                        selected_timeframes
                    ),
                    "available_fraction": {
                        name: float(
                            horizon_sequences.by_timeframe[
                                name
                            ].available.mean()
                        )
                        for name in selected_timeframes
                    },
                },
            )

            fold_results = [
                _evaluate_fold(
                    table,
                    horizon_sequences,
                    fold,
                    horizon,
                    selected_timeframes,
                    config,
                    reference,
                    horizon_directory
                    / fold.name,
                )
                for fold in folds
            ]
            neural = _aggregate_neural(
                fold_results
            )
            baseline = (
                reference.aggregate_metrics(
                    horizon
                )
            )
            comparison = _compare_to_phase7(
                neural,
                baseline,
            )
            champion = reference.champion(
                horizon
            )
            result = {
                "neural": neural,
                "phase7_champion": (
                    champion.model_dump(
                        mode="json"
                    )
                ),
                "phase7_metrics": baseline,
                "comparison": comparison,
                "label_coverage": {
                    "candidates": (
                        label_result.candidate_count
                    ),
                    "eligible": (
                        label_result.output_row_count
                    ),
                    "dropped_missing_path": (
                        label_result.dropped_missing_path
                    ),
                },
            }
            results[str(horizon)] = result
            write_json_atomic(
                horizon_directory
                / "phase8_summary.json",
                result,
            )
            write_json_atomic(
                output / "progress.json",
                {
                    "completed_horizons": (
                        list(results)
                    )
                },
            )

        summary = {
            "protocol": config.protocol_version,
            "data_version": data_version,
            "phase7_reference_run": (
                reference.champions.run_id
            ),
            "test_years": list(
                config.test_years
            ),
            "seeds": list(config.seeds),
            "holdout_opened": False,
            "horizons": results,
            "interpretation": (
                "Development-only compact neural challenger; "
                "uncalibrated; no paper or live orders."
            ),
        }
        write_json_atomic(
            output / "summary.json",
            summary,
        )
        write_text_atomic(
            output / "summary.md",
            _render_summary(summary),
        )
        files = [
            file_digest(
                item,
                relative_to=output,
            ).model_dump(mode="json")
            for item in sorted(
                output.rglob("*")
            )
            if item.is_file()
            and item.name
            not in {
                "run.json",
                "completion.json",
            }
        ]
        write_json_atomic(
            output / "completion.json",
            {
                "files": files,
                "version": content_version(
                    {"files": files}
                ),
            },
        )
        registry.finish_run(
            record,
            status="succeeded",
            metadata={
                "summary": str(
                    output / "summary.json"
                )
            },
        )
    except BaseException as exc:
        registry.finish_run(
            record,
            status="failed",
            error=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )
        raise
    return output


def verify_phase8(
    output: str | Path,
) -> dict[str, Any]:
    result = verify_benchmark(output)
    root = Path(output).resolve(strict=True)
    summary = json.loads(
        (root / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    if summary.get("protocol") != "phase8-v1":
        raise ValueError(
            "run is not a phase8-v1 result"
        )
    if (
        summary.get("holdout_opened")
        is not False
    ):
        raise ValueError(
            "phase-8 run must not open the final holdout"
        )
    result["protocol"] = summary["protocol"]
    result["phase7_reference_run"] = (
        summary.get("phase7_reference_run")
    )
    return result


__all__ = [
    "run_phase8",
    "verify_phase8",
]
