"""Formal real-data Phase-9 benchmark orchestration.

The formal run is opened only after the complete guarded real-data preflight has passed.
No Phase-9 result opens the reserved 2025+ holdout or changes the frozen Phase-7/8
reference policies.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gold_forecasting.artifacts import (
    content_version,
    write_json_atomic,
    write_parquet_atomic,
    write_text_atomic,
)
from gold_forecasting.benchmark.data_guard import preflight_development_inputs
from gold_forecasting.config import load_project_config
from gold_forecasting.data_pipeline import validate_existing_mvp_data
from gold_forecasting.datasets.mvp import _load_curated
from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.evaluation.walk_forward import select_block
from gold_forecasting.features import build_mvp_features
from gold_forecasting.phase8.pipeline import _validate_locked_runtime_dependencies
from gold_forecasting.phase8.sequences import (
    Phase8SequenceBuildResult,
    build_phase8_sequences,
)
from gold_forecasting.phase8.training import resolve_device
from gold_forecasting.phase9.artifacts import (
    finalize_phase9_manifest,
    write_phase9_fold_artifacts,
    write_phase9_run_contract,
)
from gold_forecasting.phase9.config import Phase9Config, load_phase9_config
from gold_forecasting.phase9.dataset import (
    Phase9DatasetBuildResult,
    build_phase9_dataset,
)
from gold_forecasting.phase9.orchestration import (
    GAP_MINUTES,
    Phase9FoldComparison,
    build_phase9_schedule,
    evaluate_phase9_fold,
    validate_phase9_alignment,
)
from gold_forecasting.phase9.preflight import run_phase9_preflight
from gold_forecasting.phase9.reference import (
    Phase9ReferenceBundle,
    load_phase9_reference_bundle,
)
from gold_forecasting.registry import RunRegistry, get_git_code_version


class Phase9PipelineError(ValueError):
    """Raised when a formal Phase-9 run cannot preserve the preflight contract."""


@dataclass(frozen=True, slots=True)
class _PreparedPhase9:
    config_path: Path
    config: Phase9Config
    root: Path
    code_version: str
    data_version: str
    preflight: dict[str, Any]
    references: Phase9ReferenceBundle
    candles: dict[str, pd.DataFrame]
    manifests: dict[str, Any]
    dataset: Phase9DatasetBuildResult
    sequences: Phase8SequenceBuildResult
    folds: tuple[Any, ...]


def _require_clean_code_version(code_version: str) -> None:
    if (
        code_version in {"unavailable", "uncommitted"}
        or code_version.endswith("+dirty")
    ):
        raise Phase9PipelineError(
            "formal Phase-9 runs require a clean committed Git working tree; "
            f"found {code_version!r}"
        )


def _sequence_coverage(
    sequences: Phase8SequenceBuildResult,
    timeframes: tuple[str, ...],
) -> dict[str, float]:
    return {
        name: float(sequences.by_timeframe[name].available.mean())
        for name in timeframes
    }


def _fold_structure(
    table: pd.DataFrame,
    fold: Any,
) -> dict[str, Any]:
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
    )
    if train.empty or calibration.empty or test.empty:
        raise Phase9PipelineError(
            f"formal Phase-9 fold has empty required coverage: {fold.name}"
        )
    return {
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "train_digest": sample_id_digest(train["sample_id"]),
        "calibration_digest": sample_id_digest(calibration["sample_id"]),
        "test_digest": sample_id_digest(test["sample_id"]),
        "gap_minutes": GAP_MINUTES,
    }


def _require_mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Phase9PipelineError(f"{context} is missing or malformed")
    return value


def _assert_preflight_parity(
    prepared_code_version: str,
    data_version: str,
    references: Phase9ReferenceBundle,
    dataset: Phase9DatasetBuildResult,
    sequences: Phase8SequenceBuildResult,
    folds: tuple[Any, ...],
    config: Phase9Config,
    preflight: dict[str, Any],
) -> None:
    if (
        preflight.get("status") != "passed"
        or preflight.get("formal_run_opened") is not False
        or preflight.get("holdout_opened") is not False
        or preflight.get("protocol") != "phase9-v1"
    ):
        raise Phase9PipelineError(
            "formal Phase-9 run requires a passed closed-holdout preflight"
        )
    if preflight.get("code_version") != prepared_code_version:
        raise Phase9PipelineError(
            "Git identity changed after the Phase-9 preflight"
        )
    if preflight.get("data_version") != data_version:
        raise Phase9PipelineError(
            "Phase-9 data/reference version changed after preflight"
        )
    if preflight.get("test_years") != list(config.test_years):
        raise Phase9PipelineError(
            "Phase-9 preflight changed the frozen outer years"
        )
    if preflight.get("gap_minutes") != GAP_MINUTES:
        raise Phase9PipelineError(
            "Phase-9 preflight changed the common 181-minute gap"
        )

    phase7 = _require_mapping(
        preflight.get("phase7_reference"),
        context="Phase-7 preflight reference",
    )
    phase8 = _require_mapping(
        preflight.get("phase8_reference"),
        context="Phase-8 preflight reference",
    )
    expected_phase7 = {
        "run_id": references.phase7.run_id,
        "code_version": references.phase7.code_version,
        "completion_version": references.phase7.completion_version,
    }
    expected_phase8 = {
        "run_id": references.phase8.run_id,
        "code_version": references.phase8.code_version,
        "completion_version": references.phase8.completion_version,
    }
    if phase7 != expected_phase7 or phase8 != expected_phase8:
        raise Phase9PipelineError(
            "frozen reference identity changed after the Phase-9 preflight"
        )

    preflight_dataset = _require_mapping(
        preflight.get("dataset"),
        context="Phase-9 preflight dataset",
    )
    if preflight_dataset.get("sample_digest") != dataset.diagnostics.get(
        "sample_digest"
    ):
        raise Phase9PipelineError(
            "Phase-9 common sample universe changed after preflight"
        )
    if preflight_dataset.get("common_eligible") != dataset.diagnostics.get(
        "common_eligible"
    ):
        raise Phase9PipelineError(
            "Phase-9 common sample count changed after preflight"
        )

    coverage = _require_mapping(
        preflight.get("sequence_coverage"),
        context="Phase-9 preflight sequence coverage",
    )
    if coverage != _sequence_coverage(sequences, config.timeframes):
        raise Phase9PipelineError(
            "Phase-9 sequence coverage changed after preflight"
        )

    preflight_folds = _require_mapping(
        preflight.get("folds"),
        context="Phase-9 preflight folds",
    )
    for fold in folds:
        if preflight_folds.get(fold.name) != _fold_structure(
            dataset.table,
            fold,
        ):
            raise Phase9PipelineError(
                f"Phase-9 fold structure changed after preflight: {fold.name}"
            )


def _prepare_formal_phase9(
    config_path: str | Path,
) -> _PreparedPhase9:
    path = Path(config_path).resolve(strict=True)

    # This call must complete before RunRegistry.start_run(). A failed preflight
    # therefore cannot create a formal benchmark directory.
    preflight = run_phase9_preflight(
        path,
        report_path=None,
    )

    config = load_phase9_config(path)
    project = load_project_config(
        path.parent / config.data_config
    )
    root = project.config_path.parent.parent
    code_version = get_git_code_version(root)
    _require_clean_code_version(code_version)
    if code_version != preflight.get("code_version"):
        raise Phase9PipelineError(
            "Git identity changed between Phase-9 preflight and formal preparation"
        )

    champion_config = path.parent / config.phase7_champion_config
    references = load_phase9_reference_bundle(
        root,
        config,
        champion_config,
    )

    preflight_development_inputs(project)
    validate_existing_mvp_data(project.config_path)

    candles: dict[str, pd.DataFrame] = {}
    manifests: dict[str, Any] = {}
    preflight_versions = _require_mapping(
        preflight.get("curated_versions"),
        context="Phase-9 preflight curated versions",
    )
    for timeframe in config.timeframes:
        frame, manifest = _load_curated(
            root,
            project,
            timeframe,
        )
        if preflight_versions.get(timeframe) != manifest.dataset_version:
            raise Phase9PipelineError(
                "curated dataset changed after Phase-9 preflight: "
                f"{timeframe}"
            )
        candles[timeframe] = frame
        manifests[timeframe] = manifest

    curated_versions = {
        name: manifests[name].dataset_version
        for name in config.timeframes
    }
    data_version = content_version(
        {
            "curated": curated_versions,
            "phase7_reference": references.phase7.completion_version,
            "phase8_reference": references.phase8.completion_version,
        }
    )

    anchor = build_mvp_features(
        candles["3min"],
        project.features,
    ).features
    key = [
        "instrument",
        "source",
        "prediction_time_utc",
    ]
    candidates = anchor.loc[:, key].sort_values(
        key,
        kind="stable",
    ).reset_index(drop=True)
    dataset = build_phase9_dataset(
        candles["1min"],
        candles["3min"],
        candidates,
        neutral_threshold_bps=config.neutral_threshold_bps,
    )
    sequences = build_phase8_sequences(
        candles,
        dataset.table.loc[:, key],
        config.sequence_lengths,
    )
    validate_phase9_alignment(
        dataset.table,
        sequences,
    )
    folds = build_phase9_schedule(config)
    _assert_preflight_parity(
        code_version,
        data_version,
        references,
        dataset,
        sequences,
        folds,
        config,
        preflight,
    )
    return _PreparedPhase9(
        config_path=path,
        config=config,
        root=root,
        code_version=code_version,
        data_version=data_version,
        preflight=preflight,
        references=references,
        candles=candles,
        manifests=manifests,
        dataset=dataset,
        sequences=sequences,
        folds=folds,
    )


def _aggregate_scalar(
    values: list[float],
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    if not values or not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise Phase9PipelineError(
            "Phase-9 aggregate metric contains no finite fold values"
        )
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "worst": float(
            min(values)
            if higher_is_better
            else max(values)
        ),
        "folds": values,
    }


def _path_fold_scalars(
    evaluation: dict[str, Any],
) -> dict[str, float]:
    path = _require_mapping(
        evaluation.get("path"),
        context="Phase-9 path evaluation",
    )
    per_step = _require_mapping(
        path.get("per_step"),
        context="Phase-9 per-step path metrics",
    )
    aggregate = _require_mapping(
        path.get("aggregate"),
        context="Phase-9 aggregate path metrics",
    )
    step_metrics: list[dict[str, Any]] = []
    for step in per_step.values():
        step_mapping = _require_mapping(
            step,
            context="Phase-9 path step",
        )
        step_metrics.extend(
            _require_mapping(
                component,
                context="Phase-9 path component",
            )
            for component in step_mapping.values()
        )
    aggregate_metrics = [
        _require_mapping(
            component,
            context="Phase-9 aggregate component",
        )
        for component in aggregate.values()
    ]
    reconstructed = _require_mapping(
        path.get("reconstructed_ohlc_mae_bps"),
        context="Phase-9 reconstructed OHLC metrics",
    )
    aggregate_ohlc = _require_mapping(
        path.get("aggregate_reconstructed_ohlc_mae_bps"),
        context="Phase-9 aggregate reconstructed OHLC metrics",
    )
    consistency_ohlc = _require_mapping(
        path.get("path_vs_direct_aggregate_ohlc_mae_bps"),
        context="Phase-9 path/direct aggregate consistency metrics",
    )
    return {
        "path_component_median_mae": float(
            np.mean(
                [
                    float(metric["median_mae"])
                    for metric in step_metrics
                ]
            )
        ),
        "path_component_coverage": float(
            np.mean(
                [
                    float(metric["q10_q90_coverage"])
                    for metric in step_metrics
                ]
            )
        ),
        "path_component_interval_width": float(
            np.mean(
                [
                    float(metric["mean_interval_width"])
                    for metric in step_metrics
                ]
            )
        ),
        "aggregate_component_median_mae": float(
            np.mean(
                [
                    float(metric["median_mae"])
                    for metric in aggregate_metrics
                ]
            )
        ),
        "aggregate_component_coverage": float(
            np.mean(
                [
                    float(metric["q10_q90_coverage"])
                    for metric in aggregate_metrics
                ]
            )
        ),
        "aggregate_component_interval_width": float(
            np.mean(
                [
                    float(metric["mean_interval_width"])
                    for metric in aggregate_metrics
                ]
            )
        ),
        "cumulative_15m_return_mae_bps": float(
            path["cumulative_15m_return_mae_bps"]
        ),
        "reconstructed_open_mae_bps": float(reconstructed["open"]),
        "reconstructed_high_mae_bps": float(reconstructed["high"]),
        "reconstructed_low_mae_bps": float(reconstructed["low"]),
        "reconstructed_close_mae_bps": float(reconstructed["close"]),
        "aggregate_open_mae_bps": float(aggregate_ohlc["open"]),
        "aggregate_high_mae_bps": float(aggregate_ohlc["high"]),
        "aggregate_low_mae_bps": float(aggregate_ohlc["low"]),
        "aggregate_close_mae_bps": float(aggregate_ohlc["close"]),
        "aggregate_range_mae_bps": float(
            path["aggregate_reconstructed_range_mae_bps"]
        ),
        "path_direct_open_consistency_mae_bps": float(
            consistency_ohlc["open"]
        ),
        "path_direct_high_consistency_mae_bps": float(
            consistency_ohlc["high"]
        ),
        "path_direct_low_consistency_mae_bps": float(
            consistency_ohlc["low"]
        ),
        "path_direct_close_consistency_mae_bps": float(
            consistency_ohlc["close"]
        ),
        "path_direct_range_consistency_mae_bps": float(
            path["path_vs_direct_aggregate_range_mae_bps"]
        ),
        "path_implied_direction_accuracy": float(
            _require_mapping(
                path.get("path_implied_direction"),
                context="Phase-9 path-implied direction metrics",
            )["accuracy"]
        ),
        "path_implied_direction_macro_f1": float(
            _require_mapping(
                path.get("path_implied_direction"),
                context="Phase-9 path-implied direction metrics",
            )["macro_f1"]
        ),
    }


def _aggregate_variant(
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    specs: dict[str, tuple[str, str, bool]] = {
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
        "base_net_bps": (
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
        "exposure_fraction": (
            "base_backtest",
            "exposure_fraction",
            False,
        ),
    }
    result: dict[str, Any] = {}
    for name, (section, metric, higher) in specs.items():
        result[name] = _aggregate_scalar(
            [
                float(
                    _require_mapping(
                        evaluation.get(section),
                        context=f"Phase-9 {section}",
                    )[metric]
                )
                for evaluation in evaluations
            ],
            higher_is_better=higher,
        )

    path_scalars = [
        _path_fold_scalars(evaluation)
        for evaluation in evaluations
    ]
    higher_path = {
        "path_component_coverage",
        "aggregate_component_coverage",
        "path_implied_direction_accuracy",
        "path_implied_direction_macro_f1",
    }
    for name in path_scalars[0]:
        result[name] = _aggregate_scalar(
            [
                float(record[name])
                for record in path_scalars
            ],
            higher_is_better=name in higher_path,
        )
    return result


def _aggregate_reference(
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    specs: dict[str, tuple[str, str, bool]] = {
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
        "base_net_bps": (
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
        "exposure_fraction": (
            "base_backtest",
            "exposure_fraction",
            False,
        ),
    }
    result: dict[str, Any] = {}
    for name, (section, metric, higher) in specs.items():
        result[name] = _aggregate_scalar(
            [
                float(
                    _require_mapping(
                        evaluation.get(section),
                        context=f"frozen reference {section}",
                    )[metric]
                )
                for evaluation in evaluations
            ],
            higher_is_better=higher,
        )
    return result


def _comparison_to_reference(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, float]:
    return {
        "macro_f1_mean_delta": float(
            candidate["macro_f1"]["mean"]
            - reference["macro_f1"]["mean"]
        ),
        "macro_f1_worst_delta": float(
            candidate["macro_f1"]["worst"]
            - reference["macro_f1"]["worst"]
        ),
        "brier_mean_delta": float(
            candidate["brier"]["mean"]
            - reference["brier"]["mean"]
        ),
        "log_loss_mean_delta": float(
            candidate["log_loss"]["mean"]
            - reference["log_loss"]["mean"]
        ),
        "return_mae_mean_delta_bps": float(
            candidate["return_mae_bps"]["mean"]
            - reference["return_mae_bps"]["mean"]
        ),
        "base_net_bps_mean_delta": float(
            candidate["base_net_bps"]["mean"]
            - reference["base_net_bps"]["mean"]
        ),
        "stress_net_bps_mean_delta": float(
            candidate["stress_net_bps"]["mean"]
            - reference["stress_net_bps"]["mean"]
        ),
    }


def _render_summary(
    summary: dict[str, Any],
) -> str:
    aggregate = summary["aggregate"]
    lines = [
        "# Phase 9 — direct future-candle-path benchmark",
        "",
        "Protocol: `phase9-v1`",
        "",
        "The final 2025+ holdout remained closed. Frozen Phase-7 and Phase-8 "
        "predictions were restricted to the exact Phase-9 common sample universe "
        "without refitting or policy reselection.",
        "",
        "| Candidate | Mean macro-F1 | Mean Brier | Mean log loss | "
        "Mean cumulative path return MAE | Mean path q10-q90 coverage | "
        "Mean base net bps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ("direct", "recursive"):
        metrics = aggregate[variant]
        lines.append(
            f"| {variant} | {metrics['macro_f1']['mean']:.6f} | "
            f"{metrics['brier']['mean']:.6f} | "
            f"{metrics['log_loss']['mean']:.6f} | "
            f"{metrics['cumulative_15m_return_mae_bps']['mean']:.6f} | "
            f"{metrics['path_component_coverage']['mean']:.6f} | "
            f"{metrics['base_net_bps']['mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "No automatic Phase-9 promotion is performed by this runner. "
            "The frozen protocol contains qualitative multi-fold promotion gates "
            "but no numeric post-hoc threshold may be invented after seeing results.",
            "A separate verified promotion review must decide whether future-path "
            "learning is retained as research architecture.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_phase9(
    config_path: str | Path,
) -> Path:
    """Run the canonical Phase-9 benchmark after a fresh fail-closed preflight."""

    prepared = _prepare_formal_phase9(config_path)
    config = prepared.config
    root = prepared.root

    locked_runtime = _validate_locked_runtime_dependencies(root)
    if config.deterministic_algorithms:
        os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG",
            ":4096:8",
        )
    device = resolve_device(config.device)

    registry = RunRegistry(
        root / config.output_directory,
        prepared.config_path,
        prepared.data_version,
        code_root=root,
    )
    record = registry.start_run()
    output = record.output_directory
    print(f"Phase-9 run: {output}", flush=True)

    try:
        if record.code_version != prepared.code_version:
            raise Phase9PipelineError(
                "Git identity changed while opening the formal Phase-9 run"
            )

        runtime = {
            "run_mode": "formal_benchmark",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "deterministic_algorithms": config.deterministic_algorithms,
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        }
        write_phase9_run_contract(
            output,
            config=config,
            folds=prepared.folds,
            phase8_reference_run=prepared.references.phase8.run_id,
            code_version=prepared.code_version,
            runtime=runtime,
            requirements_lock=(
                root / "requirements.lock"
            ).read_text(encoding="utf-8"),
            neural_lock=(
                root / "requirements-neural.lock"
            ).read_text(encoding="utf-8"),
            reference_metadata={
                "phase7_reference_run": prepared.references.phase7.run_id,
                "phase7_reference_code": prepared.references.phase7.code_version,
                "phase7_reference_completion": (
                    prepared.references.phase7.completion_version
                ),
                "phase8_reference_code": prepared.references.phase8.code_version,
                "phase8_reference_completion": (
                    prepared.references.phase8.completion_version
                ),
                "preflight_data_version": prepared.data_version,
            },
        )
        write_json_atomic(
            output / "dependencies.json",
            {
                name: importlib.metadata.version(name)
                for name in locked_runtime
            },
        )
        write_json_atomic(
            output / "preflight.json",
            prepared.preflight,
        )
        write_json_atomic(
            output / "dataset_diagnostics.json",
            prepared.dataset.diagnostics,
        )
        write_json_atomic(
            output / "sequence_diagnostics.json",
            prepared.sequences.diagnostics,
        )
        write_json_atomic(
            output / "source_manifests.json",
            {
                name: prepared.manifests[name].model_dump(mode="json")
                for name in config.timeframes
            },
        )
        write_parquet_atomic(
            output / "model_table.parquet",
            prepared.dataset.table,
        )
        write_text_atomic(
            output / "protocol.md",
            (
                root / "docs" / "research_protocol_phase9.md"
            ).read_text(encoding="utf-8"),
        )
        for source in sorted(
            (root / "src" / "gold_forecasting").rglob("*.py")
        ):
            write_text_atomic(
                output
                / "source_snapshot"
                / source.relative_to(root / "src"),
                source.read_text(encoding="utf-8"),
            )

        print(
            "Phase 9 preflight revalidated; starting canonical benchmark",
            flush=True,
        )
        fold_results: list[Phase9FoldComparison] = []
        fold_summary: dict[str, Any] = {}
        reference_evidence = _require_mapping(
            prepared.preflight.get("reference_parity"),
            context="Phase-9 frozen reference parity",
        )
        for fold in prepared.folds:
            print(f"Phase 9 fold {fold.name}", flush=True)
            result = evaluate_phase9_fold(
                prepared.dataset.table,
                prepared.sequences,
                fold,
                config,
            )
            fold_results.append(result)
            write_phase9_fold_artifacts(
                result,
                output / fold.name,
            )
            frozen = _require_mapping(
                reference_evidence.get(fold.name),
                context=f"Phase-9 frozen references {fold.name}",
            )
            write_json_atomic(
                output / fold.name / "frozen_references.json",
                frozen,
            )
            fold_summary[fold.name] = {
                "split": result.split_audit,
                "direct_vs_recursive": result.comparison,
                "variants": {
                    name: {
                        "evaluation": variant.summary["evaluation"],
                        "selected_epochs": variant.summary["training"][
                            "selected_epochs"
                        ],
                    }
                    for name, variant in result.variants.items()
                },
                "frozen_reference_test_digest": frozen["phase9_test_digest"],
            }
            write_json_atomic(
                output / "progress.json",
                {
                    "completed_folds": [
                        item.fold
                        for item in fold_results
                    ]
                },
            )

        variant_aggregates = {
            variant: _aggregate_variant(
                [
                    result.variants[variant].summary["evaluation"]
                    for result in fold_results
                ]
            )
            for variant in config.benchmark_variants
        }
        phase7_aggregate = _aggregate_reference(
            [
                _require_mapping(
                    reference_evidence[fold.name],
                    context=f"Phase-9 references {fold.name}",
                )["phase7"]
                for fold in prepared.folds
            ]
        )
        phase8_aggregate = _aggregate_reference(
            [
                _require_mapping(
                    reference_evidence[fold.name],
                    context=f"Phase-9 references {fold.name}",
                )["phase8"]
                for fold in prepared.folds
            ]
        )
        comparisons = {
            variant: {
                "vs_phase7": _comparison_to_reference(
                    variant_aggregates[variant],
                    phase7_aggregate,
                ),
                "vs_phase8": _comparison_to_reference(
                    variant_aggregates[variant],
                    phase8_aggregate,
                ),
            }
            for variant in config.benchmark_variants
        }

        aggregate_summary: dict[str, Any] = {
            str(name): metrics
            for name, metrics in variant_aggregates.items()
        }
        aggregate_summary["phase7_frozen"] = phase7_aggregate
        aggregate_summary["phase8_frozen"] = phase8_aggregate

        summary = {
            "protocol": "phase9-v1",
            "run_mode": "formal_benchmark",
            "code_version": prepared.code_version,
            "data_version": prepared.data_version,
            "holdout_opened": False,
            "test_years": list(config.test_years),
            "seeds": list(config.seeds),
            "phase7_reference_run": prepared.references.phase7.run_id,
            "phase8_reference_run": prepared.references.phase8.run_id,
            "dataset": prepared.dataset.diagnostics,
            "folds": fold_summary,
            "aggregate": aggregate_summary,
            "comparisons": comparisons,
            "promotion_review": {
                "state": "pending_post_benchmark_review",
                "automatic_promotion": False,
                "reason": (
                    "phase9-v1 freezes qualitative multi-fold promotion criteria "
                    "but no numeric post-hoc promotion threshold"
                ),
            },
            "interpretation": (
                "Development-only future-path challenger; probabilities remain "
                "uncalibrated; no paper or live trading activation."
            ),
        }
        write_text_atomic(
            output / "summary.md",
            _render_summary(summary),
        )
        completion = finalize_phase9_manifest(
            output,
            summary,
        )
        registry.finish_run(
            record,
            status="succeeded",
            metadata={
                "summary": str(output / "summary.json"),
                "completion_version": completion["version"],
                "promotion_review": "pending_post_benchmark_review",
            },
        )
    except BaseException as exc:
        registry.finish_run(
            record,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    return output


__all__ = [
    "Phase9PipelineError",
    "run_phase9",
]
