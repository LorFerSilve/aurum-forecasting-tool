"""Post-benchmark audits for persisted Phase-9 prediction artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gold_forecasting.artifacts import write_json_atomic
from gold_forecasting.phase9.artifacts import verify_phase9
from gold_forecasting.phase9.config import Phase9Config
from gold_forecasting.phase9.targets import PATH_STEPS


class Phase9AuditError(ValueError):
    """Raised when persisted Phase-9 artifacts cannot be audited safely."""


def _path_return_from_quantiles(
    frame: pd.DataFrame,
    *,
    clip_log_bps: float,
) -> np.ndarray:
    """Return q50 five-step close return in log-bps from persisted path columns."""

    if not np.isfinite(clip_log_bps) or clip_log_bps <= 0.0:
        raise Phase9AuditError("clip_log_bps must be finite and positive")
    result = np.zeros(len(frame), dtype=np.float64)
    for step in range(1, PATH_STEPS + 1):
        for component in ("gap_log_bps", "body_log_bps"):
            column = f"step_{step}_{component}_q50"
            if column not in frame:
                raise Phase9AuditError(
                    f"persisted path quantiles are missing {column}"
                )
            values = frame[column].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise Phase9AuditError(
                    f"persisted path quantiles contain non-finite values: {column}"
                )
            result += np.clip(
                values,
                -clip_log_bps,
                clip_log_bps,
            )
    return result


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Phase9AuditError(f"expected JSON object: {path}")
    return payload


def audit_phase9_path_return(
    run_directory: str | Path,
    *,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute cumulative q50 path-return MAE without retraining any model."""

    root = Path(run_directory).resolve(strict=True)
    verified = verify_phase9(root)
    if verified.get("run_mode") != "formal_benchmark":
        raise Phase9AuditError(
            "path-return audit requires a formal Phase-9 benchmark run"
        )

    config = Phase9Config.model_validate(
        _load_json(root / "resolved_config.json")
    )
    summary = _load_json(root / "summary.json")
    if summary.get("holdout_opened") is not False:
        raise Phase9AuditError(
            "path-return audit refuses a run that opened the final holdout"
        )

    model_table = pd.read_parquet(
        root / "model_table.parquet",
        columns=[
            "sample_id",
            "path_15m_close_return_log_bps",
        ],
    )
    if model_table.empty or model_table["sample_id"].duplicated().any():
        raise Phase9AuditError(
            "model_table sample IDs must be non-empty and unique"
        )

    required_path_columns = ["sample_id"]
    for step in range(1, PATH_STEPS + 1):
        required_path_columns.extend(
            [
                f"step_{step}_gap_log_bps_q50",
                f"step_{step}_body_log_bps_q50",
            ]
        )

    folds: dict[str, Any] = {}
    aggregate_values: dict[str, list[float]] = {
        variant: []
        for variant in config.benchmark_variants
    }
    for year in config.test_years:
        fold = f"test_{year}"
        fold_result: dict[str, Any] = {}
        for variant in config.benchmark_variants:
            path_file = (
                root
                / fold
                / variant
                / "path_quantiles.parquet"
            )
            path_frame = pd.read_parquet(
                path_file,
                columns=required_path_columns,
            )
            if path_frame.empty or path_frame["sample_id"].duplicated().any():
                raise Phase9AuditError(
                    f"{fold}/{variant} path sample IDs must be non-empty and unique"
                )
            joined = path_frame.merge(
                model_table,
                on="sample_id",
                how="left",
                validate="one_to_one",
            )
            if joined[
                "path_15m_close_return_log_bps"
            ].isna().any():
                raise Phase9AuditError(
                    f"{fold}/{variant} contains samples absent from model_table"
                )

            predicted = _path_return_from_quantiles(
                joined,
                clip_log_bps=config.reconstruction_clip_log_bps,
            )
            target = joined[
                "path_15m_close_return_log_bps"
            ].to_numpy(dtype=np.float64)
            if not np.isfinite(target).all():
                raise Phase9AuditError(
                    f"{fold}/{variant} path targets contain non-finite values"
                )
            corrected_mae = float(
                np.mean(np.abs(predicted - target))
            )
            recorded_mae = float(
                summary["folds"][fold]["variants"][variant][
                    "evaluation"
                ]["path"]["cumulative_15m_return_mae_bps"]
            )
            aggregate_values[variant].append(corrected_mae)
            fold_result[variant] = {
                "rows": len(joined),
                "recorded_aggregate_head_mae_bps": recorded_mae,
                "corrected_path_return_mae_bps": corrected_mae,
                "correction_delta_bps": corrected_mae - recorded_mae,
            }
        folds[fold] = fold_result

    aggregate: dict[str, Any] = {}
    for variant, values in aggregate_values.items():
        aggregate[variant] = {
            "folds": values,
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "worst": float(max(values)),
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "phase9-v1-path-return-audit-v1",
        "source_run_id": verified["run_id"],
        "source_completion_version": verified["version"],
        "source_verified_files": verified["verified_files"],
        "holdout_opened": False,
        "retraining_performed": False,
        "method": (
            "sum clipped q50 gap/body log-bps across the persisted five-step path; "
            "compare with path_15m_close_return_log_bps on exact persisted sample IDs"
        ),
        "folds": folds,
        "aggregate": aggregate,
    }
    if report_path is not None:
        write_json_atomic(report_path, report)
    return report


__all__ = [
    "Phase9AuditError",
    "_path_return_from_quantiles",
    "audit_phase9_path_return",
]
