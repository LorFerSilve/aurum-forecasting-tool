"""Reproducible nested development evaluation without changing MVP artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from gold_forecasting.artifacts import (
    content_version,
    file_digest,
    write_json_atomic,
    write_parquet_atomic,
    write_text_atomic,
)
from gold_forecasting.backtesting.v1 import BacktestCosts, DecisionPolicy, run_backtest_v1
from gold_forecasting.benchmark.config import BenchmarkConfig, load_benchmark_config
from gold_forecasting.benchmark.data_guard import preflight_development_inputs
from gold_forecasting.benchmark.models import (
    FittedModel,
    ModelSpec,
    candidate_specs,
    class_return_means,
    fit_model,
    prediction_records,
)
from gold_forecasting.benchmark.reporting import render_benchmark_report
from gold_forecasting.classification import CLASS_ORDER
from gold_forecasting.config import load_project_config
from gold_forecasting.data_pipeline import validate_existing_mvp_data
from gold_forecasting.datasets.mvp import _load_curated
from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.evaluation.metrics import compute_classification_metrics
from gold_forecasting.evaluation.walk_forward import (
    WalkForwardFold,
    make_walk_forward_folds,
    select_block,
    validate_development_frame,
)
from gold_forecasting.features import build_mvp_features
from gold_forecasting.labels.multihorizon import build_horizon_labels
from gold_forecasting.models.baselines import build_mvp_baseline_predictions
from gold_forecasting.registry import RunRegistry

GAP_MINUTES = 181
FAMILIES = ("reference", "logistic", "ridge", "xgboost")
POLICIES = (DecisionPolicy(0.5, 0), DecisionPolicy(0.6, 2), DecisionPolicy(0.7, 4))
CASH_POLICY = DecisionPolicy(1.0, 1e12)
BASE_COSTS = BacktestCosts()
STRESS_COSTS = BacktestCosts(spread_bps=4.5)
PROBABILITY_COLUMNS = [f"p_{label}" for label in CLASS_ORDER]


def _audit_records(frame: pd.DataFrame) -> pd.DataFrame:
    excluded = {"split"}
    # Feature values remain in the hashed model table, not repeated for every policy.
    columns = [
        name
        for name in frame
        if isinstance(name, str)
        and name not in excluded
        and (
            name.endswith("_utc")
            or name.endswith("_version")
            or "hash" in name
            or name.startswith(("entry_", "exit_", "target_", "future_"))
            or name
            in {
                "sample_id",
                "instrument",
                "source",
                "horizon_minutes",
                "arithmetic_return_bps",
                "fold",
            }
        )
    ]
    return frame.loc[:, columns].copy()


def _save_model(model: FittedModel, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        joblib.dump(model, temporary, compress=3)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _metrics(records: pd.DataFrame) -> dict[str, Any]:
    result = compute_classification_metrics(
        records["target_class"], records[PROBABILITY_COLUMNS].to_numpy()
    ).to_dict()
    errors = records["arithmetic_return_bps"] - records["expected_return_bps"]
    result["return_mae_bps"] = float(errors.abs().mean())
    result["return_rmse_bps"] = float(np.sqrt(np.mean(errors**2)))
    # Reliability bins are descriptive, not a fitted calibrator.
    matrix = records[PROBABILITY_COLUMNS].to_numpy()
    confidence = matrix.max(axis=1)
    correct = matrix.argmax(axis=1) == records["target_class_id"].to_numpy()
    reliability = []
    bin_index = np.minimum((confidence * 10).astype(int), 9)
    for index in range(10):
        lower = index / 10
        mask = bin_index == index
        if mask.any():
            reliability.append(
                {
                    "lower": float(lower),
                    "rows": int(mask.sum()),
                    "mean_confidence": float(confidence[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                }
            )
    result["reliability_bins"] = reliability
    result["expected_calibration_error"] = float(
        sum(row["rows"] * abs(row["mean_confidence"] - row["accuracy"]) for row in reliability)
        / len(records)
    )
    return result


def select_policy(
    inner_predictions: list[pd.DataFrame],
    minimum_trades: int,
) -> tuple[DecisionPolicy, list[dict[str, Any]]]:
    """Accept validation records only; caller freezes this before outer prediction."""
    attempts: list[dict[str, Any]] = []
    winner = CASH_POLICY
    best = 0.0
    for policy in POLICIES:
        metrics = [
            run_backtest_v1(frame, costs=BASE_COSTS, policy=policy).metrics
            for frame in inner_predictions
        ]
        mean_net = float(np.mean([m["cumulative_net_return_bps"] for m in metrics]))
        enough = all(m["executed_trade_count"] >= minimum_trades for m in metrics)  # type: ignore[operator]
        attempts.append(
            {
                "policy": asdict(policy),
                "folds": metrics,
                "mean_net_bps": mean_net,
                "minimum_trades_met": enough,
            }
        )
        if enough and mean_net > best:
            winner, best = policy, mean_net
    return winner, attempts


def _select_family(
    table: pd.DataFrame,
    names: tuple[str, ...],
    fold: WalkForwardFold,
    family: str,
    config: BenchmarkConfig,
    directory: Path,
    *,
    allowed_feature_names: frozenset[str] | None = None,
) -> tuple[ModelSpec, int | None, DecisionPolicy, dict[str, Any]]:
    specs = (ModelSpec("reference", 0.1),) if family == "reference" else candidate_specs(family)
    audits: list[dict[str, Any]] = []
    predictions: dict[str, list[pd.DataFrame]] = {}
    for spec in specs:
        scores: list[dict[str, Any]] = []
        records: list[pd.DataFrame] = []
        rounds: list[int | None] = []
        for inner in fold.inner_folds:
            train = select_block(table, inner.train, gap_minutes=GAP_MINUTES)
            validation = select_block(table, inner.validation, purge=False)
            if train.empty or validation.empty:
                raise ValueError(f"empty inner train/validation: {fold.name}/{inner.name}")
            started = time.perf_counter()
            model = fit_model(
                train,
                names,
                spec,
                config,
                validation=validation if family == "xgboost" else None,
                allowed_feature_names=allowed_feature_names,
            )
            probabilities, expected = model.predict(validation)
            predicted = prediction_records(_audit_records(validation), probabilities, expected)
            records.append(predicted)
            rounds.append(model.best_rounds)
            _save_model(model, directory / "inner_models" / f"{spec.name}-{inner.name}.joblib")
            scores.append(
                {
                    "inner_fold": inner.name,
                    "metrics": _metrics(predicted),
                    "fit_seconds": time.perf_counter() - started,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "train_sample_digest": sample_id_digest(train["sample_id"]),
                    "validation_sample_digest": sample_id_digest(validation["sample_id"]),
                    "best_rounds": model.best_rounds,
                }
            )
        mean_f1 = float(np.mean([s["metrics"]["macro_f1"] for s in scores]))
        mean_loss = float(np.mean([s["metrics"]["log_loss"] for s in scores]))
        mean_mae = float(np.mean([s["metrics"]["return_mae_bps"] for s in scores]))
        audits.append(
            {
                "spec": asdict(spec),
                "name": spec.name,
                "folds": scores,
                "mean_macro_f1": mean_f1,
                "mean_log_loss": mean_loss,
                "mean_return_mae_bps": mean_mae,
                "rounds": rounds,
            }
        )
        predictions[spec.name] = records
    ranked = sorted(
        audits,
        key=lambda a: (
            (a["mean_return_mae_bps"],)
            if family == "ridge"
            else (-a["mean_macro_f1"], a["mean_log_loss"])
        ),
    )
    selected = ranked[0]
    selected_spec = ModelSpec(**selected["spec"])
    best_rounds = max(1, int(np.median(selected["rounds"]))) if family == "xgboost" else None
    policy, policy_audit = select_policy(
        predictions[selected_spec.name], config.minimum_policy_trades
    )
    for index, frame in enumerate(predictions[selected_spec.name]):
        write_parquet_atomic(directory / f"selected_inner_{index}.parquet", frame)
    audit = {
        "candidates": audits,
        "selected_spec": asdict(selected_spec),
        "selected_rounds": best_rounds,
        "selected_policy": asdict(policy),
        "policy_candidates": policy_audit,
        "calibration_status": "reserved_not_fitted",
    }
    write_json_atomic(directory / "selection.json", audit)
    return selected_spec, best_rounds, policy, audit


def _evaluate_predictions(
    records: pd.DataFrame,
    policy: DecisionPolicy,
    directory: Path,
) -> dict[str, Any]:
    write_parquet_atomic(directory / "outer_predictions.parquet", records)
    base = run_backtest_v1(records, costs=BASE_COSTS, policy=policy)
    stress = run_backtest_v1(records, costs=STRESS_COSTS, policy=policy, decision_costs=BASE_COSTS)
    write_parquet_atomic(directory / "base_decisions.parquet", base.decisions)
    write_parquet_atomic(directory / "base_trades.parquet", base.trades)
    write_parquet_atomic(directory / "stress_trades.parquet", stress.trades)
    result = {
        "classification": _metrics(records),
        "base_backtest": base.metrics,
        "stress_backtest": stress.metrics,
        "policy": asdict(policy),
        "sample_digest": sample_id_digest(records["sample_id"]),
    }
    write_json_atomic(directory / "evaluation.json", result)
    return result


def evaluate_fold(
    table: pd.DataFrame,
    names: tuple[str, ...],
    fold: WalkForwardFold,
    config: BenchmarkConfig,
    directory: Path,
    *,
    allowed_feature_names: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Evaluate all candidates on identical rows; never fit the calibration block."""
    validate_development_frame(table)
    train = select_block(table, fold.train, gap_minutes=GAP_MINUTES)
    test = select_block(table, fold.test, purge=False).assign(fold=fold.name)
    calibration = select_block(table, fold.calibration, purge=False)
    if train.empty or test.empty or calibration.empty:
        raise ValueError("train, reserved calibration and test all require eligible rows")
    audit = {
        "fold": fold.name,
        "train_rows": len(train),
        "test_rows": len(test),
        "calibration_rows_reserved": len(calibration),
        "gap_minutes": GAP_MINUTES,
        "train_digest": sample_id_digest(train["sample_id"]),
        "test_digest": sample_id_digest(test["sample_id"]),
        "calibration_digest": sample_id_digest(calibration["sample_id"]),
    }
    write_json_atomic(directory / "split_audit.json", audit)
    results: dict[str, Any] = {}
    for family in FAMILIES:
        print(f"  {fold.name}: {family} tuning and evaluation", flush=True)
        destination = directory / family
        spec, rounds, policy, _ = _select_family(
            table,
            names,
            fold,
            family,
            config,
            destination,
            allowed_feature_names=allowed_feature_names,
        )
        model = fit_model(
            train,
            names,
            spec,
            config,
            rounds=rounds,
            allowed_feature_names=allowed_feature_names,
        )
        _save_model(model, destination / "final_model.joblib")
        probabilities, expected = model.predict(test)
        records = prediction_records(_audit_records(test), probabilities, expected)
        # Every final bundle must reproduce its own predictions after deserialization.
        loaded = joblib.load(destination / "final_model.joblib")
        reloaded_probabilities, reloaded_expected = loaded.predict(test)
        if not (
            np.array_equal(probabilities, reloaded_probabilities)
            and np.array_equal(expected, reloaded_expected)
        ):
            raise ValueError("saved model prediction parity failed")
        results[family] = _evaluate_predictions(records, policy, destination)
    class_means = class_return_means(train)
    naive = build_mvp_baseline_predictions(train["target_class"], test)
    probabilities_by_name = {name: batch.probabilities for name, batch in naive.items()}
    frequencies = train["target_class"].value_counts(normalize=True)
    priors = np.array([frequencies.get(label, 0.0) for label in CLASS_ORDER])
    probabilities_by_name["empirical_priors"] = np.tile(priors, (len(test), 1))
    for name, probabilities in probabilities_by_name.items():
        records = prediction_records(
            _audit_records(test), probabilities, probabilities @ class_means
        )
        # Naive references use a fixed policy, with no validation optimization.
        results[name] = _evaluate_predictions(records, POLICIES[0], directory / name)
    volatility = test["realized_volatility_20_bps"].to_numpy() * np.sqrt(
        float(test["horizon_minutes"].iloc[0]) / 60
    )
    continuous = {
        "zero_return_mae_bps": float(test["arithmetic_return_bps"].abs().mean()),
        "historical_volatility_mae_bps": float(
            np.mean(np.abs(test["future_realized_vol_bps"].to_numpy() - volatility))
        ),
        "training_median_range_mae_bps": float(
            (test["future_range_bps"] - train["future_range_bps"].median()).abs().mean()
        ),
    }
    if len({value["sample_digest"] for value in results.values()}) != 1:
        raise ValueError("candidate test sample alignment failed")
    summary = {"split": audit, "models": results, "continuous_baselines": continuous}
    write_json_atomic(directory / "fold_summary.json", summary)
    return summary


def aggregate_model_metrics(folds: list[dict[str, Any]]) -> dict[str, Any]:
    """Shared equal-weight mean/median/worst metrics for any fixed model comparison."""
    metrics = {
        "accuracy": ("classification", "accuracy", True),
        "balanced_accuracy": ("classification", "balanced_accuracy", True),
        "macro_f1": ("classification", "macro_f1", True),
        "brier": ("classification", "multiclass_brier", False),
        "log_loss": ("classification", "log_loss", False),
        "return_mae_bps": ("classification", "return_mae_bps", False),
        "calibration_error": ("classification", "expected_calibration_error", False),
        "net_bps": ("base_backtest", "cumulative_net_return_bps", True),
        "stress_net_bps": ("stress_backtest", "cumulative_net_return_bps", True),
        "trades": ("base_backtest", "executed_trade_count", True),
        "drawdown_bps": ("base_backtest", "max_drawdown_bps", False),
        "turnover": ("base_backtest", "turnover", False),
        "exposure_fraction": ("base_backtest", "exposure_fraction", False),
    }
    results: dict[str, Any] = {}
    for family in folds[0]["models"]:
        summaries = {}
        for name, (section, metric, higher) in metrics.items():
            values = [fold["models"][family][section][metric] for fold in folds]
            summaries[name] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "worst": float(min(values) if higher else max(values)),
                "folds": values,
            }
        results[family] = summaries
    return results


def aggregate_horizon(folds: list[dict[str, Any]]) -> dict[str, Any]:
    results = aggregate_model_metrics(folds)
    reference = results["reference"]
    eligible = []
    for family in ("logistic", "ridge", "xgboost"):
        value = results[family]
        if (
            value["macro_f1"]["mean"] > reference["macro_f1"]["mean"]
            and value["macro_f1"]["worst"] >= reference["macro_f1"]["worst"]
            and value["brier"]["mean"] <= reference["brier"]["mean"]
            and value["trades"]["worst"] >= 20
            and value["net_bps"]["worst"] > 0
            and value["stress_net_bps"]["mean"] > 0
            and value["net_bps"]["worst"] >= reference["net_bps"]["worst"]
        ):
            eligible.append(family)
    return {
        "models": results,
        "research_champion": "reference",
        "decision": "keep_champion",
        "promotion_eligible_for_review": eligible,
        "trading_champion": None,
        "note": "Development comparisons only; no model is activated for trading.",
    }


def run_benchmark(config_path: str | Path) -> Path:
    path = Path(config_path).resolve(strict=True)
    config = load_benchmark_config(path)
    project = load_project_config(path.parent / config.data_config)
    root = project.config_path.parent.parent
    # Validate manifests and source hashes before using any curated observations.
    preflight_development_inputs(project)
    validate_existing_mvp_data(project.config_path)
    minutes, minute_manifest = _load_curated(root, project, "1min")
    candles, candle_manifest = _load_curated(root, project, "3min")
    data_version = content_version(
        {"minute": minute_manifest.dataset_version, "features": candle_manifest.dataset_version}
    )
    registry = RunRegistry(root / config.output_directory, path, data_version, code_root=root)
    record = registry.start_run()
    output = record.output_directory
    print(f"Benchmark run: {output}", flush=True)
    try:
        write_json_atomic(output / "resolved_config.json", config.model_dump(mode="json"))
        for name, source in {"root": project.config_path, **project.component_paths}.items():
            write_text_atomic(
                output / "configs" / f"{name}.yaml", source.read_text(encoding="utf-8")
            )
        write_text_atomic(
            output / "protocol.md",
            (root / "docs/research_protocol_phase6.md").read_text(encoding="utf-8"),
        )
        versions = {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "pyarrow", "joblib")
        }
        write_json_atomic(output / "dependencies.json", versions)
        for source in sorted((root / "src/gold_forecasting").rglob("*.py")):
            write_text_atomic(
                output / "source_snapshot" / source.relative_to(root / "src"),
                source.read_text(encoding="utf-8"),
            )
        write_text_atomic(
            output / "requirements.lock", (root / "requirements.lock").read_text(encoding="utf-8")
        )
        write_json_atomic(
            output / "source_manifests.json",
            {
                "1min": minute_manifest.model_dump(mode="json"),
                "3min": candle_manifest.model_dump(mode="json"),
            },
        )
        built = build_mvp_features(candles, project.features)
        names = built.catalog.feature_names
        write_json_atomic(output / "feature_catalog.json", built.catalog.model_dump(mode="json"))
        folds = make_walk_forward_folds(test_years=config.test_years)
        write_json_atomic(
            output / "schedule.json",
            {"gap_minutes": GAP_MINUTES, "folds": [fold.as_record() for fold in folds]},
        )
        results: dict[str, Any] = {}
        for horizon in config.horizons:
            print(f"Building and evaluating horizon {horizon} minutes", flush=True)
            label_result = build_horizon_labels(minutes, built.features, horizon_minutes=horizon)
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
            directory = output / f"horizon_{horizon}"
            write_parquet_atomic(directory / "model_table.parquet", table)
            summaries = [
                evaluate_fold(table, names, fold, config, directory / fold.name) for fold in folds
            ]
            result = aggregate_horizon(summaries)
            result["label_coverage"] = {
                "candidates": label_result.candidate_count,
                "eligible": label_result.output_row_count,
                "dropped_missing_path": label_result.dropped_missing_path,
            }
            results[str(horizon)] = result
            write_json_atomic(directory / "horizon_summary.json", result)
            write_json_atomic(output / "progress.json", {"completed_horizons": list(results)})
        summary = {
            "protocol": config.protocol_version,
            "full_protocol_scope": config.full_protocol_scope,
            "test_years": list(config.test_years),
            "data_version": data_version,
            "holdout_opened": False,
            "horizons": results,
            "interpretation": "Development benchmark; uncalibrated; no live orders.",
        }
        write_json_atomic(output / "summary.json", summary)
        write_text_atomic(output / "summary.md", render_benchmark_report(summary))
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
            record, status="succeeded", metadata={"summary": str(output / "summary.json")}
        )
    except BaseException as exc:
        registry.finish_run(record, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    return output


def verify_benchmark(output: str | Path) -> dict[str, Any]:
    """Reject incomplete runs and detect any modified persisted artifact."""
    root = Path(output).resolve(strict=True)
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if run["status"] != "succeeded":
        raise ValueError("benchmark run did not succeed")
    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    if completion["version"] != content_version({"files": completion["files"]}):
        raise ValueError("completion manifest version mismatch")
    for entry in completion["files"]:
        target = (root / entry["path"]).resolve(strict=True)
        if not target.is_relative_to(root):
            raise ValueError("artifact escapes run directory")
        if file_digest(target, relative_to=root).model_dump(mode="json") != entry:
            raise ValueError(f"artifact digest mismatch: {entry['path']}")
    return {
        "run_id": run["run_id"],
        "verified_files": len(completion["files"]),
        "version": completion["version"],
    }
