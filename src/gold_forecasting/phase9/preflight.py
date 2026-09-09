"""Real-data preflight for Phase 9 before the formal benchmark is exposed."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gold_forecasting.artifacts import content_version, file_digest, write_json_atomic
from gold_forecasting.benchmark.data_guard import preflight_development_inputs
from gold_forecasting.config import load_project_config
from gold_forecasting.data_pipeline import validate_existing_mvp_data
from gold_forecasting.datasets.mvp import _load_curated
from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.evaluation.walk_forward import select_block
from gold_forecasting.features import build_mvp_features
from gold_forecasting.phase8.pipeline import _validate_locked_runtime_dependencies
from gold_forecasting.phase8.sequences import build_phase8_sequences
from gold_forecasting.phase8.training import resolve_device
from gold_forecasting.phase9.config import Phase9Config, load_phase9_config
from gold_forecasting.phase9.dataset import build_phase9_dataset
from gold_forecasting.phase9.orchestration import (
    GAP_MINUTES,
    build_phase9_schedule,
    validate_phase9_alignment,
)
from gold_forecasting.phase9.parity import evaluate_frozen_reference_on_phase9_universe
from gold_forecasting.phase9.reference import (
    Phase9ReferenceBundle,
    Phase9ReferenceError,
    load_phase9_reference_bundle,
)
from gold_forecasting.phase9.targets import PATH_COMPONENTS, PATH_STEPS
from gold_forecasting.phase9.training import (
    Phase9Prediction,
    predict_phase9_model,
    save_phase9_checkpoint,
    train_phase9_model,
)
from gold_forecasting.registry import get_git_code_version


class Phase9PreflightError(ValueError):
    """Raised when a real-data Phase-9 preflight contract is not satisfied."""


def _require_clean_code_version(code_version: str) -> None:
    if (
        code_version in {"unavailable", "uncommitted"}
        or code_version.endswith("+dirty")
    ):
        raise Phase9PreflightError(
            "Phase-9 real-data preflight requires a clean committed Git working tree; "
            f"found {code_version!r}"
        )


def _sequence_coverage(
    sequences: Any,
    timeframes: tuple[str, ...],
) -> dict[str, float]:
    return {
        name: float(sequences.by_timeframe[name].available.mean())
        for name in timeframes
    }


def _fold_rows(
    table: pd.DataFrame,
    fold: Any,
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
        raise Phase9PreflightError(
            f"Phase-9 preflight found empty outer coverage: {fold.name}"
        )
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
            raise Phase9PreflightError(
                "Phase-9 preflight found empty inner coverage: "
                f"{fold.name}/{inner.name}"
            )
    return train, calibration, test


def _validate_phase8_data_identity(
    references: Phase9ReferenceBundle,
    manifests: dict[str, Any],
    timeframes: tuple[str, ...],
) -> dict[str, str]:
    frozen = references.phase8.read_json("source_manifests.json")
    result: dict[str, str] = {}
    for timeframe in timeframes:
        expected_record = frozen.get(timeframe)
        if not isinstance(expected_record, dict):
            raise Phase9PreflightError(
                f"frozen Phase-8 source manifest is missing {timeframe}"
            )
        expected = expected_record.get("dataset_version")
        actual = manifests[timeframe].dataset_version
        if not isinstance(expected, str) or expected != actual:
            raise Phase9PreflightError(
                "current curated data differs from the canonical Phase-8 input: "
                f"{timeframe}"
            )
        result[timeframe] = actual
    return result


def _reference_fold_evidence(
    references: Phase9ReferenceBundle,
    table: pd.DataFrame,
    fold: Any,
) -> dict[str, Any]:
    _, _, test = _fold_rows(table, fold)
    phase7 = references.phase7_fold(fold.name)
    phase8 = references.phase8_fold(fold.name)
    if phase7.original_sample_digest != phase8.original_sample_digest:
        raise Phase9PreflightError(
            f"Phase-7/8 original 15m sample digests differ: {fold.name}"
        )
    phase7_evaluation = evaluate_frozen_reference_on_phase9_universe(
        phase7.records,
        test,
        phase7.policy,
    )
    phase8_evaluation = evaluate_frozen_reference_on_phase9_universe(
        phase8.records,
        test,
        phase8.policy,
    )
    expected_digest = sample_id_digest(test["sample_id"])
    if (
        phase7_evaluation["sample_digest"] != expected_digest
        or phase8_evaluation["sample_digest"] != expected_digest
    ):
        raise Phase9PreflightError(
            f"frozen reference common-universe digest mismatch: {fold.name}"
        )
    return {
        "phase9_test_rows": len(test),
        "phase9_test_digest": expected_digest,
        "phase7_original_digest": phase7.original_sample_digest,
        "phase8_original_digest": phase8.original_sample_digest,
        "phase7": phase7_evaluation,
        "phase8": phase8_evaluation,
    }


def _validate_prediction_invariants(
    prediction: Phase9Prediction,
    *,
    expected_rows: int,
) -> dict[str, float]:
    probabilities = prediction.probabilities
    path = prediction.path_quantiles_log_bps
    aggregate = prediction.aggregate_quantiles_log_bps
    expected_path_shape = (
        expected_rows,
        PATH_STEPS,
        len(PATH_COMPONENTS),
        3,
    )
    expected_aggregate_shape = (
        expected_rows,
        len(PATH_COMPONENTS),
        3,
    )
    if probabilities.shape != (expected_rows, 3):
        raise Phase9PreflightError(
            f"Phase-9 smoke probabilities have wrong shape: {probabilities.shape}"
        )
    if path.shape != expected_path_shape:
        raise Phase9PreflightError(
            f"Phase-9 smoke path quantiles have wrong shape: {path.shape}"
        )
    if aggregate.shape != expected_aggregate_shape:
        raise Phase9PreflightError(
            f"Phase-9 smoke aggregate quantiles have wrong shape: {aggregate.shape}"
        )
    arrays = (
        probabilities,
        prediction.expected_return_bps,
        prediction.predicted_range_bps,
        prediction.predicted_volatility_bps,
        path,
        aggregate,
    )
    if any(not np.isfinite(values).all() for values in arrays):
        raise Phase9PreflightError("Phase-9 smoke prediction contains non-finite values")
    probability_error = float(
        np.max(
            np.abs(
                probabilities.sum(axis=1)
                - 1.0
            )
        )
    )
    if probability_error > 1e-12:
        raise Phase9PreflightError(
            "Phase-9 smoke probabilities do not sum to one"
        )
    if (
        (path[..., 1] < path[..., 0]).any()
        or (path[..., 2] < path[..., 1]).any()
        or (aggregate[..., 1] < aggregate[..., 0]).any()
        or (aggregate[..., 2] < aggregate[..., 1]).any()
    ):
        raise Phase9PreflightError("Phase-9 smoke quantiles crossed")
    upper_index = PATH_COMPONENTS.index("upper_wick_log_bps")
    lower_index = PATH_COMPONENTS.index("lower_wick_log_bps")
    minimum_wick_q10 = float(
        min(
            path[:, :, upper_index, 0].min(),
            path[:, :, lower_index, 0].min(),
            aggregate[:, upper_index, 0].min(),
            aggregate[:, lower_index, 0].min(),
        )
    )
    if minimum_wick_q10 < -1e-12:
        raise Phase9PreflightError(
            "Phase-9 smoke produced a negative wick q10"
        )
    return {
        "probability_sum_max_abs_error": probability_error,
        "minimum_wick_q10_log_bps": minimum_wick_q10,
    }


def _runtime_smoke(
    root: Path,
    table: pd.DataFrame,
    sequences: Any,
    folds: tuple[Any, ...],
    config: Phase9Config,
) -> dict[str, Any]:
    first_inner = folds[0].inner_folds[0]
    inner_train = select_block(
        table,
        first_inner.train,
        gap_minutes=GAP_MINUTES,
    )
    candidate_rows = inner_train.index.to_numpy(dtype=np.int64)
    availability = np.column_stack(
        [
            sequences.by_timeframe[name].available[candidate_rows]
            for name in config.timeframes
        ]
    ).astype(bool)
    complete_rows = candidate_rows[availability.all(axis=1)]
    if len(complete_rows) < 64:
        raise Phase9PreflightError(
            "Phase-9 runtime smoke requires at least 64 complete multi-timeframe rows"
        )
    smoke_rows = complete_rows[-min(len(complete_rows), 512):]
    prediction_rows = smoke_rows[-min(len(smoke_rows), 128):]

    reports_root = root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    variants: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(
        prefix=".phase9-preflight-",
        dir=reports_root,
    ) as temporary:
        smoke_root = Path(temporary)
        for variant in config.benchmark_variants:
            training = train_phase9_model(
                sequences,
                table,
                smoke_rows,
                None,
                config,
                seed=config.seeds[0],
                forced_epochs=1,
                model_variant=variant,
            )
            prediction = predict_phase9_model(
                training,
                sequences,
                prediction_rows,
                config,
            )
            invariants = _validate_prediction_invariants(
                prediction,
                expected_rows=len(prediction_rows),
            )
            checkpoint = smoke_root / variant / "checkpoint.pt"
            save_phase9_checkpoint(
                training,
                checkpoint,
            )
            checkpoint_digest = file_digest(
                checkpoint,
                relative_to=smoke_root,
            ).model_dump(mode="json")
            payload = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            if (
                not isinstance(payload, dict)
                or payload.get("model_variant") != variant
                or payload.get("parameter_count") != training.parameter_count
            ):
                raise Phase9PreflightError(
                    f"Phase-9 checkpoint smoke verification failed: {variant}"
                )
            evidence_path = smoke_root / variant / "evidence.json"
            write_json_atomic(
                evidence_path,
                {
                    "variant": variant,
                    "rows": len(smoke_rows),
                    "prediction_rows": len(prediction_rows),
                    "parameter_count": training.parameter_count,
                    "optimizer_steps": training.optimizer_steps,
                    "amp_skipped_steps": training.amp_skipped_steps,
                    **invariants,
                },
            )
            evidence_digest = file_digest(
                evidence_path,
                relative_to=smoke_root,
            ).model_dump(mode="json")
            variants[variant] = {
                "fit_rows": len(smoke_rows),
                "prediction_rows": len(prediction_rows),
                "device": training.device,
                "mixed_precision_used": training.mixed_precision_used,
                "parameter_count": training.parameter_count,
                "optimizer_steps": training.optimizer_steps,
                "amp_skipped_steps": training.amp_skipped_steps,
                "fit_seconds": training.fit_seconds,
                "inference_seconds": prediction.inference_seconds,
                "checkpoint_digest": checkpoint_digest,
                "evidence_digest": evidence_digest,
                **invariants,
            }
    return {
        "status": "passed",
        "variants": variants,
    }


def run_phase9_preflight(
    config_path: str | Path,
    *,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate real data, references, runtime and short fits without a formal run."""

    path = Path(config_path).resolve(strict=True)
    config = load_phase9_config(path)
    project = load_project_config(
        path.parent / config.data_config
    )
    root = project.config_path.parent.parent
    code_version = get_git_code_version(root)
    _require_clean_code_version(code_version)

    locked_runtime = _validate_locked_runtime_dependencies(root)
    if config.deterministic_algorithms:
        os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG",
            ":4096:8",
        )
    device = resolve_device(config.device)

    champion_config = path.parent / config.phase7_champion_config
    try:
        references = load_phase9_reference_bundle(
            root,
            config,
            champion_config,
        )
    except Phase9ReferenceError as exc:
        raise Phase9PreflightError(str(exc)) from exc

    preflight_development_inputs(project)
    validate_existing_mvp_data(project.config_path)

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

    curated_versions = _validate_phase8_data_identity(
        references,
        manifests,
        config.timeframes,
    )
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

    fold_evidence: dict[str, Any] = {}
    fold_structure: dict[str, Any] = {}
    for fold in folds:
        train, calibration, test = _fold_rows(
            dataset.table,
            fold,
        )
        fold_structure[fold.name] = {
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "test_rows": len(test),
            "train_digest": sample_id_digest(train["sample_id"]),
            "calibration_digest": sample_id_digest(
                calibration["sample_id"]
            ),
            "test_digest": sample_id_digest(test["sample_id"]),
            "gap_minutes": GAP_MINUTES,
        }
        fold_evidence[fold.name] = _reference_fold_evidence(
            references,
            dataset.table,
            fold,
        )

    runtime_smoke = _runtime_smoke(
        root,
        dataset.table,
        sequences,
        folds,
        config,
    )
    report: dict[str, Any] = {
        "protocol": "phase9-v1",
        "status": "passed",
        "formal_run_opened": False,
        "holdout_opened": False,
        "code_version": code_version,
        "data_version": data_version,
        "test_years": list(config.test_years),
        "gap_minutes": GAP_MINUTES,
        "phase7_reference": {
            "run_id": references.phase7.run_id,
            "code_version": references.phase7.code_version,
            "completion_version": references.phase7.completion_version,
        },
        "phase8_reference": {
            "run_id": references.phase8.run_id,
            "code_version": references.phase8.code_version,
            "completion_version": references.phase8.completion_version,
        },
        "curated_versions": curated_versions,
        "dataset": dataset.diagnostics,
        "sequence_coverage": _sequence_coverage(
            sequences,
            config.timeframes,
        ),
        "folds": fold_structure,
        "reference_parity": fold_evidence,
        "runtime": {
            "locked_dependencies": locked_runtime,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "deterministic_algorithms": config.deterministic_algorithms,
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
        },
        "runtime_smoke": runtime_smoke,
    }
    if report_path is not None:
        destination = Path(report_path)
        if not destination.is_absolute():
            destination = root / destination
        destination = destination.resolve()
        expected_reports = (root / "reports").resolve()
        if (
            not destination.is_relative_to(expected_reports)
            or destination.suffix.lower() != ".json"
        ):
            raise Phase9PreflightError(
                "Phase-9 preflight report must be a JSON file under reports/"
            )
        write_json_atomic(
            destination,
            report,
        )
    return report


__all__ = [
    "Phase9PreflightError",
    "run_phase9_preflight",
]
