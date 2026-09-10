"""Exploratory source ablation orchestration on fully verified local inputs."""

from __future__ import annotations

import importlib.metadata
import shutil
from pathlib import Path

from gold_forecasting.artifacts import (
    sha256_file,
    write_json_atomic,
    write_parquet_atomic,
    write_text_atomic,
)
from gold_forecasting.phase10.artifacts import complete_phase10_run, summarize_phase10
from gold_forecasting.phase10.real_preflight import prepare_phase10_inputs
from gold_forecasting.phase10.training import evaluate_context_fold
from gold_forecasting.registry import RunRegistry, get_git_code_version


def run_phase10(config_path: str | Path = "configs/phase10_silver_ablation.yaml") -> Path:
    """A registry run is opened only after every real-data preflight gate passes."""
    path = Path(config_path).resolve(strict=True)
    prepared = prepare_phase10_inputs(path)
    root, config = prepared.root, prepared.config
    profile = config.profile
    report = prepared.report
    if (
        prepared.source.source_id != profile.source_id
        or report.get("protocol") != profile.protocol
        or report.get("formal_benchmark_ready") is not False
        or report.get("strict_pit_source_ready") is not False
        or report.get("champion_changed") is not False
        or report.get("trading_activated") is not False
        or report.get("status") != "passed"
        or report.get("exploratory_ablation_ready") is not True
        or report.get("holdout_opened") is not False
        or report["code_version"] != get_git_code_version(root)
        or report["config_sha256"] != sha256_file(path)
        or report["source_config_sha256"] != sha256_file(path.parent / config.source_config)
    ):
        raise ValueError("Phase-10 preflight identity changed or mandatory gates are not green")
    output_root = root / config.output_directory
    if output_root.resolve() != output_root.absolute():
        raise ValueError("Phase-10 output root must not be redirected")
    registry = RunRegistry(output_root, path, report["data_version"], code_root=root)
    record = registry.start_run()
    output = record.output_directory
    print(f"Phase-10 exploratory run: {output}", flush=True)
    try:
        if record.code_version != report["code_version"]:
            raise ValueError("Git identity changed after Phase-10 preflight")
        write_json_atomic(output / "resolved_config.json", config.model_dump(mode="json"))
        write_json_atomic(output / "source_config.json", prepared.source.model_dump(mode="json"))
        write_json_atomic(output / "preflight.json", report)
        write_json_atomic(
            output / "schedule.json",
            {
                "gap_minutes": 181,
                "folds": [fold.as_record() for fold in prepared.folds],
            },
        )
        write_json_atomic(
            output / "dependencies.json",
            {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "pyarrow",
                    "joblib",
                    "xgboost",
                )
            },
        )
        write_text_atomic(output / "requirements.lock", (root / "requirements.lock").read_text())
        write_text_atomic(
            output / "protocol.md",
            (root / "docs/research_protocol_phase10.md").read_text("utf-8"),
        )
        for source in sorted((root / "src/gold_forecasting").rglob("*.py")):
            write_text_atomic(
                output / "source_snapshot" / source.relative_to(root / "src"),
                source.read_text("utf-8"),
            )
        for name, source in {
            "root": prepared.project.config_path,
            **prepared.project.component_paths,
        }.items():
            shutil.copyfile(source, _parent(output / "configs" / f"project_{name}.yaml"))
        for name in (config.features_config, config.source_config, config.phase7_champion_config):
            shutil.copyfile(path.parent / name, _parent(output / "configs" / Path(name).name))
        write_parquet_atomic(output / "features.parquet", prepared.table)
        reference = prepared.references.run
        shutil.copyfile(reference.root / "completion.json", output / "reference_completion.json")
        folds = {}
        for fold in prepared.folds:
            print(
                f"  {fold.name}: {profile.source_id} logistic selection and evaluation", flush=True
            )
            frozen = prepared.references.fold(fold.name)
            destination = output / "folds" / fold.name
            folds[fold.name] = evaluate_context_fold(
                prepared.table,
                prepared.feature_columns,
                fold,
                frozen.records,
                frozen.inner_records,
                frozen.policy,
                destination,
                profile=profile,
            )
            base = f"horizon_15/mvp/{fold.name}"
            copies = {
                "reference/outer_predictions.parquet": f"{base}/logistic/outer_predictions.parquet",
                "reference_split.json": f"{base}/split_audit.json",
                "reference_selection.json": f"{base}/logistic/selection.json",
                "reference_evaluation.json": f"{base}/logistic/evaluation.json",
                **{
                    f"reference_inner_{i}.parquet": f"{base}/logistic/selected_inner_{i}.parquet"
                    for i in range(len(fold.inner_folds))
                },
            }
            for target, original in copies.items():
                shutil.copyfile(reference.verified_path(original), destination / target)
        summary = {
            "protocol": config.protocol_version,
            "run_mode": "exploratory_modeled_latency",
            "run_id": record.run_id,
            "code_version": record.code_version,
            "data_version": report["data_version"],
            "test_years": list(config.test_years),
            "gap_minutes": 181,
            "horizon_minutes": 15,
            "holdout_opened": False,
            "champion_promotion": False,
            "trading_activation": False,
            "baseline": {
                "run": config.phase7_reference_run,
                "completion": config.phase7_reference_completion,
                "variant": "mvp",
                "family": "logistic",
            },
            "folds": folds,
            "comparison": summarize_phase10(list(folds.values()), profile=profile),
        }
        write_json_atomic(output / "summary.json", summary)
        write_text_atomic(
            output / "summary.md",
            (
                f"# Phase 10 exploratory {profile.title} ablation\n\n"
                f"Protocol: `{config.protocol_version}`.\n\n"
                f"Decision: `{summary['comparison']['decision']}`.\n\n"
                "Modeled latency is an assumption, not historical release evidence. "
                "The frozen Phase-7 15m mvp/logistic champion is retained. "
                "No promotion or trading activation; the 2025+ holdout remains closed.\n"
            ),
        )
        if get_git_code_version(root) != record.code_version:
            raise ValueError("Git identity changed during Phase-10 evaluation")
        # Seal while the registry still says "running".  The completion inventory
        # deliberately excludes run.json, whose terminal status is written next.
        complete_phase10_run(output)
    except Exception as exc:
        registry.finish_run(record, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    registry.finish_run(record, metadata={"protocol": config.protocol_version})
    return output


def _parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["run_phase10"]
