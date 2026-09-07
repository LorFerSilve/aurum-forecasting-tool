"""Read-only reproduction of frozen final fits and inner-selected policies.

This verifies the original artifact checksums and exact dependency versions,
then uses the current implementation to refit the recorded final candidates.
It does not load pickle/joblib bundles, fetch data, regenerate features/labels,
or rerun all inner hyperparameter candidates. Exact output parity is evidence
for this bounded reproduction, not independent profitability confirmation.
"""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gold_forecasting.artifacts import content_version
from gold_forecasting.backtesting.v1 import DecisionPolicy
from gold_forecasting.benchmark.config import BenchmarkConfig, load_benchmark_config
from gold_forecasting.benchmark.models import ModelSpec, candidate_specs, fit_model
from gold_forecasting.benchmark.pipeline import (
    FAMILIES,
    GAP_MINUTES,
    PROBABILITY_COLUMNS,
    select_policy,
    verify_benchmark,
)
from gold_forecasting.classification import CLASS_ORDER
from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.evaluation.walk_forward import (
    WalkForwardFold,
    make_walk_forward_folds,
    select_block,
    validate_development_frame,
)
from gold_forecasting.features import FeatureCatalog

_REQUIRED_DEPENDENCIES = frozenset(
    {"numpy", "pandas", "scipy", "scikit-learn", "xgboost", "pyarrow", "joblib"}
)


class BenchmarkReproductionError(ValueError):
    """The saved benchmark cannot be reproduced under its frozen contract."""


class _VerifiedArtifacts:
    """Permit reads only from files included in the verified completion manifest."""

    def __init__(self, root: Path) -> None:
        self.root = root
        completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
        self.paths = {entry["path"] for entry in completion["files"]}

    def path(self, relative: str) -> Path:
        if relative not in self.paths:
            raise BenchmarkReproductionError(f"required artifact is not hash-verified: {relative}")
        target = (self.root / relative).resolve(strict=True)
        if not target.is_relative_to(self.root):
            raise BenchmarkReproductionError("required artifact escapes the run directory")
        return target

    def mapping(self, relative: str) -> dict[str, Any]:
        payload = json.loads(self.path(relative).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise BenchmarkReproductionError(f"artifact must contain a JSON object: {relative}")
        return payload

    def frame(self, relative: str) -> pd.DataFrame:
        frame = pd.read_parquet(self.path(relative))
        validate_development_frame(frame)
        return frame


def _matching_dependencies(recorded: dict[str, Any]) -> dict[str, str]:
    missing = _REQUIRED_DEPENDENCIES.difference(recorded)
    if missing:
        raise BenchmarkReproductionError(f"dependency snapshot is incomplete: {sorted(missing)}")
    installed: dict[str, str] = {}
    for name, saved in sorted(recorded.items()):
        if not isinstance(saved, str) or not saved:
            raise BenchmarkReproductionError(f"invalid recorded dependency version: {name}")
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise BenchmarkReproductionError(
                f"recorded dependency is not installed: {name}"
            ) from exc
        if installed[name] != saved:
            raise BenchmarkReproductionError(
                f"dependency version mismatch for {name}: "
                f"saved={saved}, installed={installed[name]}"
            )
    return installed


def _assert_sample_ids(expected: pd.DataFrame, saved: pd.DataFrame, *, context: str) -> None:
    if "sample_id" not in saved or not np.array_equal(
        expected["sample_id"].to_numpy(), saved["sample_id"].to_numpy()
    ):
        raise BenchmarkReproductionError(f"ordered sample ID parity failed: {context}")


def _check_split(
    table: pd.DataFrame,
    fold: WalkForwardFold,
    audit: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = select_block(table, fold.train, gap_minutes=GAP_MINUTES)
    test = select_block(table, fold.test, purge=False).assign(fold=fold.name)
    calibration = select_block(table, fold.calibration, purge=False)
    expected = {
        "fold": fold.name,
        "train_rows": len(train),
        "test_rows": len(test),
        "calibration_rows_reserved": len(calibration),
        "gap_minutes": GAP_MINUTES,
        "train_digest": sample_id_digest(train["sample_id"]),
        "test_digest": sample_id_digest(test["sample_id"]),
        "calibration_digest": sample_id_digest(calibration["sample_id"]),
    }
    if train.empty or test.empty or calibration.empty or audit != expected:
        raise BenchmarkReproductionError(f"frozen split audit parity failed: {fold.name}")
    return train, test


def _check_policy(
    artifacts: _VerifiedArtifacts,
    relative: str,
    table: pd.DataFrame,
    fold: WalkForwardFold,
    selection: dict[str, Any],
    config: BenchmarkConfig,
) -> DecisionPolicy:
    inner_predictions = []
    for index, inner in enumerate(fold.inner_folds):
        expected = select_block(table, inner.validation, purge=False)
        saved = artifacts.frame(f"{relative}/selected_inner_{index}.parquet")
        _assert_sample_ids(expected, saved, context=f"{relative}/{inner.name}")
        inner_predictions.append(saved)
    policy, attempts = select_policy(inner_predictions, config.minimum_policy_trades)
    if asdict(policy) != selection["selected_policy"]:
        raise BenchmarkReproductionError(f"inner-selected policy parity failed: {relative}")
    if content_version({"attempts": attempts}) != content_version(
        {"attempts": selection["policy_candidates"]}
    ):
        raise BenchmarkReproductionError(f"inner policy audit parity failed: {relative}")
    return policy


def _check_candidate_selection(
    selection: dict[str, Any],
    family: str,
    fold: WalkForwardFold,
    config: BenchmarkConfig,
    *,
    context: str,
) -> tuple[ModelSpec, int | None]:
    """Replay the frozen ranking from saved scores, without claiming to refit them."""
    expected_specs = (
        (ModelSpec("reference", 0.1),) if family == "reference" else candidate_specs(family)
    )
    candidates = selection["candidates"]
    if [candidate["spec"] for candidate in candidates] != [asdict(spec) for spec in expected_specs]:
        raise BenchmarkReproductionError(f"frozen candidate inventory parity failed: {context}")
    for candidate in candidates:
        scores = candidate["folds"]
        if [score["inner_fold"] for score in scores] != [inner.name for inner in fold.inner_folds]:
            raise BenchmarkReproductionError(f"candidate inner fold parity failed: {context}")
        for metric in ("macro_f1", "log_loss", "return_mae_bps"):
            mean = float(np.mean([score["metrics"][metric] for score in scores]))
            if not np.isfinite(mean) or mean != candidate[f"mean_{metric}"]:
                raise BenchmarkReproductionError(f"candidate mean score parity failed: {context}")
        rounds = [score["best_rounds"] for score in scores]
        valid_rounds = (
            all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 1 <= value <= config.xgb_max_rounds
                for value in rounds
            )
            if family == "xgboost"
            else all(value is None for value in rounds)
        )
        if not valid_rounds or rounds != candidate["rounds"]:
            raise BenchmarkReproductionError(f"candidate round audit parity failed: {context}")
    winner = min(
        candidates,
        key=lambda candidate: (
            (candidate["mean_return_mae_bps"],)
            if family == "ridge"
            else (-candidate["mean_macro_f1"], candidate["mean_log_loss"])
        ),
    )
    selected_rounds = max(1, int(np.median(winner["rounds"]))) if family == "xgboost" else None
    if (
        selection["selected_spec"] != winner["spec"]
        or selection["selected_rounds"] != selected_rounds
    ):
        raise BenchmarkReproductionError(f"inner-selected model audit parity failed: {context}")
    return ModelSpec(**winner["spec"]), selected_rounds


def reproduce_benchmark(run_directory: str | Path) -> dict[str, Any]:
    """Verify all artifacts, then exactly refit each recorded final candidate.

    The returned JSON-compatible report belongs outside the original run.
    This function performs no writes and never accesses the original market
    data directories or reserved holdout. It fails before fitting if captured
    dependencies, configuration, data versions, or the schedule disagree.
    """

    root = Path(run_directory).resolve(strict=True)
    verification = verify_benchmark(root)
    artifacts = _VerifiedArtifacts(root)
    dependencies = _matching_dependencies(artifacts.mapping("dependencies.json"))
    config = BenchmarkConfig.model_validate(artifacts.mapping("resolved_config.json"))
    if load_benchmark_config(artifacts.path("config.yaml")) != config:
        raise BenchmarkReproductionError("frozen and resolved configuration disagree")
    catalog = FeatureCatalog.model_validate(artifacts.mapping("feature_catalog.json"))
    names = catalog.feature_names
    folds = make_walk_forward_folds(test_years=config.test_years)
    schedule = {"gap_minutes": GAP_MINUTES, "folds": [fold.as_record() for fold in folds]}
    if artifacts.mapping("schedule.json") != schedule:
        raise BenchmarkReproductionError("saved and current frozen schedules disagree")
    manifests = artifacts.mapping("source_manifests.json")
    data_version = content_version(
        {
            "minute": manifests["1min"]["dataset_version"],
            "features": manifests["3min"]["dataset_version"],
        }
    )
    summary = artifacts.mapping("summary.json")
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if data_version != summary["data_version"] or data_version != run["data_version"]:
        raise BenchmarkReproductionError("source manifests and recorded data versions disagree")
    if summary["protocol"] != config.protocol_version or summary["holdout_opened"] is not False:
        raise BenchmarkReproductionError("saved summary violates the development-only protocol")
    if set(summary["horizons"]) != {str(horizon) for horizon in config.horizons}:
        raise BenchmarkReproductionError("saved and configured horizon inventories disagree")

    parities: list[dict[str, Any]] = []
    for horizon in config.horizons:
        horizon_root = f"horizon_{horizon}"
        table = artifacts.frame(f"{horizon_root}/model_table.parquet")
        if (
            table.empty
            or "sample_id" not in table
            or table["sample_id"].duplicated().any()
            or not table["horizon_minutes"].eq(horizon).all()
        ):
            raise BenchmarkReproductionError(f"invalid saved model table: {horizon_root}")
        for fold in folds:
            fold_root = f"{horizon_root}/{fold.name}"
            train, test = _check_split(
                table, fold, artifacts.mapping(f"{fold_root}/split_audit.json")
            )
            for family in FAMILIES:
                relative = f"{fold_root}/{family}"
                selection = artifacts.mapping(f"{relative}/selection.json")
                if selection["calibration_status"] != "reserved_not_fitted":
                    raise BenchmarkReproductionError(f"calibration status is unsafe: {relative}")
                spec, rounds = _check_candidate_selection(
                    selection, family, fold, config, context=relative
                )
                policy = _check_policy(artifacts, relative, table, fold, selection, config)
                saved = artifacts.frame(f"{relative}/outer_predictions.parquet")
                _assert_sample_ids(test, saved, context=relative)
                print(f"Reproducing {horizon}m / {fold.name} / {family}", flush=True)
                model = fit_model(train, names, spec, config, rounds=rounds)
                probabilities, expected = model.predict(test)
                if not np.array_equal(
                    probabilities, saved[PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
                ):
                    raise BenchmarkReproductionError(f"probability parity failed: {relative}")
                if not np.array_equal(expected, saved["expected_return_bps"].to_numpy()):
                    raise BenchmarkReproductionError(f"expected return parity failed: {relative}")
                labels = np.asarray(CLASS_ORDER)[probabilities.argmax(axis=1)]
                if not np.array_equal(labels, saved["predicted_class"].to_numpy()):
                    raise BenchmarkReproductionError(f"predicted class parity failed: {relative}")
                if model.preprocessor.train_sample_digest != sample_id_digest(train["sample_id"]):
                    raise BenchmarkReproductionError(
                        f"train-only preprocessing parity failed: {relative}"
                    )
                parities.append(
                    {
                        "horizon_minutes": horizon,
                        "fold": fold.name,
                        "family": family,
                        "selected_spec": asdict(spec),
                        "selected_rounds": rounds,
                        "selected_policy": asdict(policy),
                        "train_sample_digest": model.preprocessor.train_sample_digest,
                        "test_sample_digest": sample_id_digest(test["sample_id"]),
                        "test_rows": len(test),
                        "exact_prediction_parity": True,
                        "inner_policy_parity": True,
                    }
                )
    return {
        "schema_version": 1,
        "run_id": verification["run_id"],
        "completion_version": verification["version"],
        "verified_files": verification["verified_files"],
        "protocol": config.protocol_version,
        "data_version": data_version,
        "dependency_versions": dependencies,
        "final_fit_parity_count": len(parities),
        "inner_policy_parity_count": len(parities),
        "holdout_opened": False,
        "original_run_modified": False,
        "parities": parities,
        "scope": {
            "artifact_hashes_verified": True,
            "final_candidates_refitted": True,
            "inner_policies_reselected_from_saved_predictions": True,
            "model_selection_replayed_from_saved_scores": True,
            "all_inner_candidates_refitted": False,
            "features_and_labels_regenerated": False,
            "implementation": "current_installed_code_with_matching_dependency_versions",
            "interpretation": "Reproduction of development artifacts, not fresh holdout evidence.",
        },
    }


__all__ = ["BenchmarkReproductionError", "reproduce_benchmark"]
