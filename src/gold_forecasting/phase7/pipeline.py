"""Guarded phase-7 feature ablations on the frozen phase-6 evaluation machinery."""

from __future__ import annotations

import hashlib
import importlib.metadata
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
from gold_forecasting.benchmark.config import BenchmarkConfig
from gold_forecasting.benchmark.data_guard import preflight_development_inputs
from gold_forecasting.benchmark.pipeline import (
    FAMILIES,
    aggregate_horizon,
    evaluate_fold,
    verify_benchmark,
)
from gold_forecasting.config import load_project_config
from gold_forecasting.data_pipeline import validate_existing_mvp_data
from gold_forecasting.datasets.mvp import _load_curated
from gold_forecasting.evaluation.walk_forward import (
    make_walk_forward_folds,
    select_block,
    validate_development_frame,
)
from gold_forecasting.features.phase7 import (
    Phase7FeatureBuildResult,
    build_phase7_features,
    load_phase7_feature_config,
)
from gold_forecasting.labels.multihorizon import build_horizon_labels
from gold_forecasting.phase7.config import Phase7Config, load_phase7_config
from gold_forecasting.registry import RunRegistry

_TIMEFRAMES = ("1min", "3min", "5min", "15min", "30min", "1h", "3h")


def _benchmark_config(config: Phase7Config) -> BenchmarkConfig:
    """Reuse the exact phase-6 tuning/backtest contract for phase-7 challengers."""
    return BenchmarkConfig(
        horizons=config.horizons,
        test_years=config.test_years,
        seed=config.seed,
        xgb_max_rounds=config.xgb_max_rounds,
        xgb_early_stopping_rounds=config.xgb_early_stopping_rounds,
        minimum_policy_trades=config.minimum_policy_trades,
        output_directory=config.output_directory,
    )


def _feature_distribution(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name in feature_names:
        train_values = train[name].astype("float64")
        test_values = test[name].astype("float64")
        train_std = float(train_values.std(ddof=0))
        records[name] = {
            "train_mean": float(train_values.mean()),
            "train_std": train_std,
            "train_p05": float(train_values.quantile(0.05)),
            "train_p50": float(train_values.quantile(0.50)),
            "train_p95": float(train_values.quantile(0.95)),
            "test_mean": float(test_values.mean()),
            "test_std": float(test_values.std(ddof=0)),
            "test_p05": float(test_values.quantile(0.05)),
            "test_p50": float(test_values.quantile(0.50)),
            "test_p95": float(test_values.quantile(0.95)),
            "standardized_mean_shift": float(
                (test_values.mean() - train_values.mean()) / max(train_std, 1e-12)
            ),
        }
    return records


def _predictive_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return bool(
        candidate["macro_f1"]["mean"] > baseline["macro_f1"]["mean"]
        and candidate["macro_f1"]["worst"] >= baseline["macro_f1"]["worst"]
        and candidate["brier"]["mean"] <= baseline["brier"]["mean"]
    )


def _economic_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return bool(
        candidate["trades"]["worst"] >= 20
        and candidate["net_bps"]["worst"] > 0
        and candidate["stress_net_bps"]["mean"] > 0
        and candidate["net_bps"]["worst"] >= baseline["net_bps"]["worst"]
    )


def _compare_variants(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare every model+feature challenger to the common MVP/reference baseline."""
    baseline = variants["mvp"]["models"]["reference"]
    predictive: list[dict[str, str]] = []
    trading: list[dict[str, str]] = []
    deltas: dict[str, Any] = {}
    variant_order = tuple(variants)
    for variant in variant_order:
        deltas[variant] = {}
        for family in FAMILIES:
            metrics = variants[variant]["models"][family]
            deltas[variant][family] = {
                "macro_f1_mean_delta": (
                    metrics["macro_f1"]["mean"] - baseline["macro_f1"]["mean"]
                ),
                "macro_f1_worst_delta": (
                    metrics["macro_f1"]["worst"] - baseline["macro_f1"]["worst"]
                ),
                "brier_mean_delta": metrics["brier"]["mean"] - baseline["brier"]["mean"],
                "log_loss_mean_delta": (
                    metrics["log_loss"]["mean"] - baseline["log_loss"]["mean"]
                ),
                "return_mae_mean_delta_bps": (
                    metrics["return_mae_bps"]["mean"] - baseline["return_mae_bps"]["mean"]
                ),
            }
            if variant == "mvp" and family == "reference":
                continue
            if _predictive_gate(metrics, baseline):
                candidate = {"variant": variant, "family": family}
                predictive.append(candidate)
                if _economic_gate(metrics, baseline):
                    trading.append(candidate)

    ranked = sorted(
        predictive,
        key=lambda item: (
            -variants[item["variant"]]["models"][item["family"]]["macro_f1"]["mean"],
            variants[item["variant"]]["models"][item["family"]]["brier"]["mean"],
            item["variant"],
            item["family"],
        ),
    )
    research_champion = (
        ranked[0] if ranked else {"variant": "mvp", "family": "reference"}
    )
    return {
        "baseline": {"variant": "mvp", "family": "reference"},
        "predictive_admission_candidates": predictive,
        "economic_promotion_candidates": trading,
        "research_champion_for_review": research_champion,
        "trading_champion": None,
        "decision": "review_predictive_challenger" if predictive else "keep_mvp_features",
        "deltas_vs_mvp_reference": deltas,
        "note": (
            "Feature admission is a development-only research decision. "
            "No phase-7 result activates live or paper trading."
        ),
    }


def _render_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 7 — richer price-only feature ablations",
        "",
        f"Protocol: `{summary['protocol']}`",
        "",
        "The final holdout remains closed. Every ablation uses the same common causal",
        "feature universe, walk-forward folds, model-selection rules and execution costs.",
        "",
        "## Horizon decisions",
        "",
        "| Horizon | Research decision | Predictive candidates | Economic candidates |",
        "|---:|---|---:|---:|",
    ]
    for horizon, record in summary["horizons"].items():
        comparison = record["comparison"]
        lines.append(
            f"| {horizon} min | {comparison['decision']} | "
            f"{len(comparison['predictive_admission_candidates'])} | "
            f"{len(comparison['economic_promotion_candidates'])} |"
        )
    lines.extend(
        [
            "",
            "Microstructure features remain disabled because the current HistData source",
            "does not provide reliable historical ask, spread or tick-count observations.",
            "",
            "Probabilities remain uncalibrated; full OOF calibration belongs to phase 11.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_phase7(config_path: str | Path) -> Path:
    path = Path(config_path).resolve(strict=True)
    config = load_phase7_config(path)
    project = load_project_config(path.parent / config.data_config)
    feature_config = load_phase7_feature_config(path.parent / config.features_config)
    benchmark_config = _benchmark_config(config)
    root = project.config_path.parent.parent

    preflight_development_inputs(project)
    validate_existing_mvp_data(project.config_path)

    minutes, minute_manifest = _load_curated(root, project, "1min")
    candles: dict[str, pd.DataFrame] = {"1min": minutes}
    manifests: dict[str, Any] = {"1min": minute_manifest}
    for timeframe in _TIMEFRAMES:
        if timeframe == "1min":
            continue
        frame, manifest = _load_curated(root, project, timeframe)
        candles[timeframe] = frame
        manifests[timeframe] = manifest

    data_version = content_version(
        {name: manifest.dataset_version for name, manifest in manifests.items()}
    )
    registry = RunRegistry(root / config.output_directory, path, data_version, code_root=root)
    record = registry.start_run()
    output = record.output_directory
    print(f"Phase-7 run: {output}", flush=True)

    try:
        if (
            record.code_version in {"unavailable", "uncommitted"}
            or record.code_version.endswith("+dirty")
        ):
            raise RuntimeError(
                "formal phase-7 runs require a clean committed Git working tree; "
                f"found {record.code_version!r}"
            )
        write_json_atomic(output / "resolved_config.json", config.model_dump(mode="json"))
        write_json_atomic(
            output / "resolved_feature_config.json", feature_config.model_dump(mode="json")
        )
        write_text_atomic(
            output / "configs" / "phase7.yaml", path.read_text(encoding="utf-8")
        )
        write_text_atomic(
            output / "configs" / "features_phase7.yaml",
            (path.parent / config.features_config).read_text(encoding="utf-8"),
        )
        for name, source in {"root": project.config_path, **project.component_paths}.items():
            write_text_atomic(
                output / "configs" / f"project_{name}.yaml",
                source.read_text(encoding="utf-8"),
            )
        protocol_path = root / "docs" / "research_protocol_phase7.md"
        write_text_atomic(
            output / "protocol.md", protocol_path.read_text(encoding="utf-8")
        )
        versions = {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "xgboost",
                "pyarrow",
                "joblib",
            )
        }
        write_json_atomic(output / "dependencies.json", versions)
        for source in sorted((root / "src" / "gold_forecasting").rglob("*.py")):
            write_text_atomic(
                output / "source_snapshot" / source.relative_to(root / "src"),
                source.read_text(encoding="utf-8"),
            )
        write_text_atomic(
            output / "requirements.lock",
            (root / "requirements.lock").read_text(encoding="utf-8"),
        )
        write_json_atomic(
            output / "source_manifests.json",
            {
                name: manifest.model_dump(mode="json")
                for name, manifest in manifests.items()
            },
        )

        built: Phase7FeatureBuildResult = build_phase7_features(
            candles, project.features, feature_config
        )
        write_json_atomic(
            output / "feature_catalog.json", built.catalog.model_dump(mode="json")
        )
        write_json_atomic(output / "feature_diagnostics.json", built.diagnostics)
        write_json_atomic(
            output / "ablation_manifest.json",
            {
                "variant_order": list(built.catalog.variants),
                "variants": {
                    name: list(features)
                    for name, features in built.catalog.variants.items()
                },
                "common_row_count": len(built.features),
                "unavailable_groups": built.catalog.unavailable_groups,
            },
        )

        folds = make_walk_forward_folds(test_years=config.test_years)
        results: dict[str, Any] = {}
        for horizon in config.horizons:
            print(f"Phase 7 horizon {horizon} minutes", flush=True)
            label_result = build_horizon_labels(
                minutes, built.features, horizon_minutes=horizon
            )
            table = built.features.merge(
                label_result.labels,
                on=["instrument", "source", "prediction_time_utc"],
                validate="one_to_one",
            )
            table["sample_id"] = [
                hashlib.sha256(
                    f"{instrument}|{source}|{stamp.isoformat()}|{horizon}".encode()
                ).hexdigest()
                for instrument, source, stamp in table[
                    ["instrument", "source", "prediction_time_utc"]
                ].itertuples(index=False, name=None)
            ]
            validate_development_frame(table)
            horizon_directory = output / f"horizon_{horizon}"
            write_parquet_atomic(horizon_directory / "model_table.parquet", table)

            distribution: dict[str, Any] = {}
            for fold in folds:
                train = select_block(table, fold.train, gap_minutes=181)
                test = select_block(table, fold.test, purge=False)
                distribution[fold.name] = _feature_distribution(
                    train, test, built.catalog.feature_names
                )
            write_json_atomic(
                horizon_directory / "feature_distributions.json", distribution
            )

            variant_results: dict[str, Any] = {}
            for variant, feature_names in built.catalog.variants.items():
                print(f"  ablation {variant}", flush=True)
                destination = horizon_directory / variant
                summaries = [
                    evaluate_fold(
                        table,
                        feature_names,
                        fold,
                        benchmark_config,
                        destination / fold.name,
                    )
                    for fold in folds
                ]
                aggregate = aggregate_horizon(summaries)
                aggregate["feature_count"] = len(feature_names)
                aggregate["feature_spec_version"] = built.catalog.feature_spec_version
                variant_results[variant] = aggregate
                write_json_atomic(destination / "variant_summary.json", aggregate)

            comparison = _compare_variants(variant_results)
            result = {
                "variants": variant_results,
                "comparison": comparison,
                "label_coverage": {
                    "candidates": label_result.candidate_count,
                    "eligible": label_result.output_row_count,
                    "dropped_missing_path": label_result.dropped_missing_path,
                },
            }
            results[str(horizon)] = result
            write_json_atomic(horizon_directory / "phase7_summary.json", result)
            write_json_atomic(
                output / "progress.json", {"completed_horizons": list(results)}
            )

        summary = {
            "protocol": config.protocol_version,
            "data_version": data_version,
            "feature_spec_version": built.catalog.feature_spec_version,
            "test_years": list(config.test_years),
            "holdout_opened": False,
            "horizons": results,
            "interpretation": (
                "Development-only feature ablation; uncalibrated; no live or paper orders."
            ),
        }
        write_json_atomic(output / "summary.json", summary)
        write_text_atomic(output / "summary.md", _render_summary(summary))
        files = [
            file_digest(item, relative_to=output).model_dump(mode="json")
            for item in sorted(output.rglob("*"))
            if item.is_file() and item.name not in {"run.json", "completion.json"}
        ]
        write_json_atomic(
            output / "completion.json",
            {"files": files, "version": content_version({"files": files})},
        )
        registry.finish_run(
            record,
            status="succeeded",
            metadata={"summary": str(output / "summary.json")},
        )
    except BaseException as exc:
        registry.finish_run(
            record, status="failed", error=f"{type(exc).__name__}: {exc}"
        )
        raise
    return output


def verify_phase7(output: str | Path) -> dict[str, Any]:
    result = verify_benchmark(output)
    root = Path(output).resolve(strict=True)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("protocol") != "phase7-v1":
        raise ValueError("run is not a phase7-v1 result")
    if summary.get("holdout_opened") is not False:
        raise ValueError("phase-7 run must not open the final holdout")
    result["protocol"] = summary["protocol"]
    return result


__all__ = ["run_phase7", "verify_phase7"]
