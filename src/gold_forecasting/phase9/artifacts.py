"""Atomic persistence and integrity verification for phase-9 research runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from gold_forecasting.artifacts import (
    content_version,
    file_digest,
    write_json_atomic,
    write_parquet_atomic,
    write_text_atomic,
)
from gold_forecasting.backtesting.v1 import DecisionPolicy, run_backtest_v1
from gold_forecasting.benchmark.pipeline import (
    BASE_COSTS,
    STRESS_COSTS,
    verify_benchmark,
)
from gold_forecasting.evaluation.walk_forward import WalkForwardFold
from gold_forecasting.phase9.config import Phase9Config
from gold_forecasting.phase9.orchestration import (
    GAP_MINUTES,
    Phase9FoldComparison,
    Phase9VariantFoldResult,
)
from gold_forecasting.phase9.targets import PATH_COMPONENTS, PATH_STEPS
from gold_forecasting.phase9.training import save_phase9_checkpoint

_QUANTILE_NAMES = ("q10", "q50", "q90")


class Phase9ArtifactError(ValueError):
    """Raised when persisted phase-9 evidence violates the artifact contract."""


def _path_prediction_frame(
    result: Phase9VariantFoldResult,
) -> pd.DataFrame:
    frame = result.outer_records.loc[
        :,
        [
            "sample_id",
            "instrument",
            "source",
            "prediction_time_utc",
        ],
    ].copy()
    path = result.outer_path_quantiles
    aggregate = result.outer_aggregate_quantiles
    if len(frame) != len(path) or len(frame) != len(aggregate):
        raise Phase9ArtifactError(
            "outer records and path quantiles are not row-aligned"
        )
    for step in range(PATH_STEPS):
        for component_index, component in enumerate(PATH_COMPONENTS):
            for quantile_index, quantile in enumerate(_QUANTILE_NAMES):
                frame[
                    f"step_{step + 1}_{component}_{quantile}"
                ] = path[
                    :,
                    step,
                    component_index,
                    quantile_index,
                ]
    for component_index, component in enumerate(PATH_COMPONENTS):
        for quantile_index, quantile in enumerate(_QUANTILE_NAMES):
            frame[
                f"aggregate_{component}_{quantile}"
            ] = aggregate[
                :,
                component_index,
                quantile_index,
            ]
    return frame


def write_phase9_run_contract(
    root: str | Path,
    *,
    config: Phase9Config,
    folds: tuple[WalkForwardFold, ...],
    phase8_reference_run: str,
    code_version: str,
    runtime: dict[str, Any],
    requirements_lock: str,
    neural_lock: str,
) -> None:
    destination = Path(root)
    if not phase8_reference_run:
        raise Phase9ArtifactError(
            "phase-9 run contract requires an explicit phase-8 reference"
        )
    write_json_atomic(
        destination / "resolved_config.json",
        config.model_dump(mode="json"),
    )
    write_json_atomic(
        destination / "schedule.json",
        {
            "gap_minutes": GAP_MINUTES,
            "folds": [
                fold.as_record()
                for fold in folds
            ],
        },
    )
    write_json_atomic(
        destination / "reference.json",
        {
            "phase8_reference_run": (
                phase8_reference_run
            ),
            "code_version": code_version,
        },
    )
    write_json_atomic(
        destination / "runtime.json",
        runtime,
    )
    write_text_atomic(
        destination / "requirements.lock",
        requirements_lock,
    )
    write_text_atomic(
        destination / "requirements-neural.lock",
        neural_lock,
    )


def write_phase9_fold_artifacts(
    result: Phase9FoldComparison,
    directory: str | Path,
) -> None:
    root = Path(directory)
    write_json_atomic(
        root / "split_audit.json",
        result.split_audit,
    )
    write_json_atomic(
        root / "comparison.json",
        result.comparison,
    )
    for variant, variant_result in result.variants.items():
        variant_root = root / variant
        write_json_atomic(
            variant_root / "summary.json",
            variant_result.summary,
        )
        write_parquet_atomic(
            variant_root / "outer_predictions.parquet",
            variant_result.outer_records,
        )
        write_parquet_atomic(
            variant_root / "path_quantiles.parquet",
            _path_prediction_frame(
                variant_result
            ),
        )
        for inner_name, records in variant_result.inner_records.items():
            write_parquet_atomic(
                variant_root
                / "inner_predictions"
                / f"{inner_name}.parquet",
                records,
            )
        for name, training in variant_result.checkpoints.items():
            checkpoint_root = (
                variant_root
                / "checkpoints"
                / Path(name)
            )
            save_phase9_checkpoint(
                training,
                checkpoint_root / "checkpoint.pt",
            )
            write_json_atomic(
                checkpoint_root / "history.json",
                {
                    "epochs": list(
                        training.history
                    ),
                    "best_epoch": (
                        training.best_epoch
                    ),
                    "model_variant": (
                        training.model_variant
                    ),
                },
            )

        policy = DecisionPolicy(
            **variant_result.summary[
                "evaluation"
            ]["policy"]
        )
        base = run_backtest_v1(
            variant_result.outer_records,
            costs=BASE_COSTS,
            policy=policy,
        )
        stress = run_backtest_v1(
            variant_result.outer_records,
            costs=STRESS_COSTS,
            policy=policy,
            decision_costs=BASE_COSTS,
        )
        write_parquet_atomic(
            variant_root / "base_decisions.parquet",
            base.decisions,
        )
        write_parquet_atomic(
            variant_root / "base_trades.parquet",
            base.trades,
        )
        write_parquet_atomic(
            variant_root / "stress_trades.parquet",
            stress.trades,
        )


def finalize_phase9_manifest(
    root: str | Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    destination = Path(root)
    if summary.get("protocol") != "phase9-v1":
        raise Phase9ArtifactError(
            "phase-9 summary must use protocol phase9-v1"
        )
    if summary.get("holdout_opened") is not False:
        raise Phase9ArtifactError(
            "phase-9 summary must keep the final holdout closed"
        )
    if summary.get("test_years") != [2022, 2023, 2024]:
        raise Phase9ArtifactError(
            "phase-9 summary changed the frozen outer years"
        )
    write_json_atomic(
        destination / "summary.json",
        summary,
    )
    files = [
        file_digest(
            item,
            relative_to=destination,
        ).model_dump(mode="json")
        for item in sorted(
            destination.rglob("*")
        )
        if (
            item.is_file()
            and item.name
            not in {
                "run.json",
                "completion.json",
            }
        )
    ]
    payload = {
        "files": files,
        "version": content_version(
            {"files": files}
        ),
    }
    write_json_atomic(
        destination / "completion.json",
        payload,
    )
    return payload


def verify_phase9(
    output: str | Path,
) -> dict[str, Any]:
    result = verify_benchmark(output)
    root = Path(output).resolve(strict=True)
    summary = json.loads(
        (root / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    if summary.get("protocol") != "phase9-v1":
        raise Phase9ArtifactError(
            "run is not a phase9-v1 result"
        )
    if summary.get("holdout_opened") is not False:
        raise Phase9ArtifactError(
            "phase-9 run opened the final holdout"
        )
    if summary.get("test_years") != [2022, 2023, 2024]:
        raise Phase9ArtifactError(
            "phase-9 run changed the frozen outer years"
        )
    schedule = json.loads(
        (root / "schedule.json").read_text(
            encoding="utf-8"
        )
    )
    if schedule.get("gap_minutes") != GAP_MINUTES:
        raise Phase9ArtifactError(
            "phase-9 run changed the common 181-minute gap"
        )
    reference = json.loads(
        (root / "reference.json").read_text(
            encoding="utf-8"
        )
    )
    reference_run = reference.get(
        "phase8_reference_run"
    )
    if not isinstance(reference_run, str) or not reference_run:
        raise Phase9ArtifactError(
            "phase-9 run has no phase-8 reference"
        )
    result["protocol"] = "phase9-v1"
    result["phase8_reference_run"] = (
        reference_run
    )
    return result


__all__ = [
    "Phase9ArtifactError",
    "finalize_phase9_manifest",
    "verify_phase9",
    "write_phase9_fold_artifacts",
    "write_phase9_run_contract",
]
