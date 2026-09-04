"""Minimal vertical pipeline orchestration shared by the CLI and tests."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from gold_forecasting.config import ProjectConfig, load_project_config
from gold_forecasting.data_pipeline import (
    DataBuildError,
    build_mvp_data,
    validate_existing_mvp_data,
)
from gold_forecasting.datasets import DatasetBuildResult, build_mvp_dataset
from gold_forecasting.registry import RunRecord, RunRegistry
from gold_forecasting.research_mvp import MVPAnalysisResult, run_mvp_analysis
from gold_forecasting.runtime import configure_logging, seed_everything

LOGGER = logging.getLogger(__name__)

MVP_STAGES = (
    "data_import",
    "data_validate",
    "dataset_build",
    "train",
    "evaluate",
    "backtest",
    "predict",
)


class PipelineNotReadyError(RuntimeError):
    """Backward-compatible error type retained for earlier API consumers."""


@dataclass(frozen=True, slots=True)
class PreparedMVPArtifacts:
    dataset: DatasetBuildResult
    data_action: str


def _prepare_mvp_artifacts(config: ProjectConfig) -> PreparedMVPArtifacts:
    try:
        validate_existing_mvp_data(config.config_path)
        data_action = "reused"
    except DataBuildError as exc:
        if "missing curated manifest" not in str(exc):
            raise
        build_mvp_data(config.config_path)
        data_action = "built"
    dataset = build_mvp_dataset(config.config_path)
    return PreparedMVPArtifacts(dataset=dataset, data_action=data_action)


def _resolve_from_project(config: ProjectConfig, path: Path) -> Path:
    project_root = config.config_path.parent.parent
    return (path if path.is_absolute() else project_root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_components(config: ProjectConfig, record: RunRecord) -> dict[str, str]:
    snapshot_directory = record.output_directory / "config_components"
    snapshot_directory.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for name, source_path in sorted(config.component_paths.items()):
        destination = snapshot_directory / f"{name}{source_path.suffix.lower()}"
        shutil.copyfile(source_path, destination)
        hashes[name] = _sha256(destination)
    return hashes


def run_mvp_pipeline(config_path: Path, *, dry_run: bool = False) -> RunRecord:
    """Run the complete research MVP or its lightweight structural dry-run."""

    config = load_project_config(config_path)
    configure_logging(os.getenv("GOLD_FORECAST_LOG_LEVEL", "INFO"))
    seed_everything(config.root.run.seed)
    output_root = _resolve_from_project(config, config.root.run.output_root)
    prepared = None if dry_run else _prepare_mvp_artifacts(config)
    data_version = (
        config.root.run.data_version if prepared is None else prepared.dataset.dataset_version
    )
    registry = RunRegistry(
        output_root=output_root,
        config_path=config.config_path,
        data_version=data_version,
        code_root=config.config_path.parent.parent,
    )
    record = registry.start_run(dry_run=dry_run)

    try:
        component_hashes = _snapshot_components(config, record)
        if dry_run:
            stages = [
                {"name": stage, "status": "validated_placeholder"} for stage in MVP_STAGES
            ]
            finished = registry.finish_run(
                record,
                status="succeeded",
                metadata={
                    "mode": "phase_1_skeleton",
                    "instrument": config.instrument.instrument.id,
                    "protocol_version": config.root.protocol_version,
                    "component_config_sha256": component_hashes,
                    "stages": stages,
                },
            )
            LOGGER.info("Validated phase-1 MVP skeleton in run %s", finished.run_id)
            return finished

        if prepared is None:  # pragma: no cover - narrowed by dry_run above
            raise RuntimeError("real MVP artifacts were not prepared")
        analysis: MVPAnalysisResult = run_mvp_analysis(
            config.config_path,
            record.output_directory,
        )
        stage_details = {
            "data_import": prepared.data_action,
            "data_validate": "validated",
            "dataset_build": prepared.dataset.dataset_version,
            "train": analysis.training.bundle.model_version,
            "evaluate": f"{analysis.test_rows} aligned test rows",
            "backtest": (
                f"{analysis.backtest.metrics.executed_trade_count} executed trades"
            ),
            "predict": analysis.example_prediction.predicted_class,
        }
        stages = [
            {"name": stage, "status": "succeeded", "detail": stage_details[stage]}
            for stage in MVP_STAGES
        ]
        finished = registry.finish_run(
            record,
            status="succeeded",
            metadata={
                "mode": "research_mvp_v0.1",
                "instrument": config.instrument.instrument.id,
                "protocol_version": config.root.protocol_version,
                "component_config_sha256": component_hashes,
                "model_version": analysis.training.bundle.model_version,
                "test_rows": analysis.test_rows,
                "calibration_status": "preliminary",
                "stages": stages,
            },
        )
        LOGGER.info("Completed research MVP in run %s", finished.run_id)
        return finished
    except Exception as exc:
        with suppress(RuntimeError):
            registry.finish_run(record, status="failed", error=str(exc))
        raise


__all__ = ["MVP_STAGES", "PipelineNotReadyError", "run_mvp_pipeline"]
