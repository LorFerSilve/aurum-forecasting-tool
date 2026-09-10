"""Synthetic-only dollar integration: frozen folds, fallback, sealing and replay."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from gold_forecasting.artifacts import content_version, file_digest, sha256_file
from gold_forecasting.backtesting.v1 import DecisionPolicy
from gold_forecasting.benchmark.models import prediction_records
from gold_forecasting.benchmark.pipeline import PROBABILITY_COLUMNS
from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.evaluation.walk_forward import make_walk_forward_folds, select_block
from gold_forecasting.phase10 import artifacts
from gold_forecasting.phase10.config import Phase10Config, load_phase10_config
from gold_forecasting.phase10.contracts import load_source
from gold_forecasting.phase10.profiles import DOLLAR_PROFILE, SILVER_PROFILE, profile_for_protocol
from gold_forecasting.phase10.real_preflight import (
    Phase10PreflightError,
    inspect_context_metadata,
    validate_modeled_context_source,
)
from gold_forecasting.phase10.training import (
    context_usable_mask,
    evaluate_context_fold,
    route_context_predictions,
)
from tests.test_phase10_artifact_lifecycle import _identity_fixture
from tests.test_phase10_training import PRICE_NAMES, synthetic_table


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _table() -> pd.DataFrame:
    table = synthetic_table()
    table = table.drop(columns=[name for name in table if "silver" in name])
    count = np.arange(len(table), dtype=float)
    table["gold_close"] = 2000 + count / 20 + np.sin(count / 3)
    table["dollar_value"] = 1.1 + count / 100000 + np.sin(count / 5) / 1000
    table["dollar_age_seconds"] = 60.0
    table["dollar_is_missing"] = table.index % 149 == 0
    table["dollar_is_stale"] = table.index % 137 == 0
    table = DOLLAR_PROFILE.build_features(table)
    table["dollar_usable"] = context_usable_mask(table, profile=DOLLAR_PROFILE)
    return table


def _reference(rows: pd.DataFrame) -> pd.DataFrame:
    # Fixed artificial outputs, never a refit or substitute for the real Phase-7 run.
    probabilities = np.tile([0.234567890123, 0.531234567891, 0.234197541986], (len(rows), 1))
    return prediction_records(rows, probabilities, np.full(len(rows), -0.017345678901))


def _evaluate(table: pd.DataFrame, directory: Path, year: int = 2022) -> dict:
    fold = next(item for item in make_walk_forward_folds() if item.name == f"test_{year}")
    return evaluate_context_fold(
        table,
        PRICE_NAMES,
        fold,
        _reference(select_block(table, fold.test, purge=False)),
        tuple(
            _reference(select_block(table, item.validation, purge=False))
            for item in fold.inner_folds
        ),
        DecisionPolicy(),
        directory,
        profile=DOLLAR_PROFILE,
    )


def test_profiles_preserve_silver_configuration_and_bind_dollar_source(tmp_path: Path) -> None:
    silver = load_phase10_config("configs/phase10_silver_ablation.yaml")
    dollar = load_phase10_config("configs/phase10_dollar_ablation.yaml")
    assert silver == Phase10Config()
    assert "profile" not in silver.model_dump()
    assert silver.profile == SILVER_PROFILE
    assert dollar.profile == DOLLAR_PROFILE
    assert dollar.gap_minutes == 181 and dollar.test_years == (2022, 2023, 2024)
    source = load_source("configs/phase10_dollar_exploratory.yaml")
    validate_modeled_context_source(source, DOLLAR_PROFILE)
    with pytest.raises(Phase10PreflightError, match="source mismatch"):
        inspect_context_metadata(tmp_path, silver, source)
    with pytest.raises(Phase10PreflightError, match="source mismatch"):
        validate_modeled_context_source(
            source.model_copy(update={"availability_basis": "provider_timestamp"}), DOLLAR_PROFILE
        )
    with pytest.raises(ValueError, match="unsupported frozen"):
        profile_for_protocol("phase10-dollar-strict-v1")


@pytest.mark.parametrize("year", [2022, 2023, 2024])
def test_dollar_nested_folds_train_only_and_exact_fallback(tmp_path: Path, year: int) -> None:
    table = _table()
    result = _evaluate(table, tmp_path, year)
    selected = json.loads((tmp_path / "selection.json").read_text())
    assert [(item["spec"]["value"], item["class_weight"]) for item in selected["candidates"]] == [
        (0.1, "balanced"),
        (1.0, "balanced"),
        (10.0, "balanced"),
    ]
    assert selected["calibration_status"] == "reserved_not_fitted"
    reference = pd.read_parquet(tmp_path / "reference/outer_predictions.parquet")
    dollar = pd.read_parquet(tmp_path / "dollar/outer_predictions.parquet")
    fallback = dollar.used_price_only_fallback
    assert fallback.any() and (~fallback).any()
    pd.testing.assert_series_equal(reference.sample_id, dollar.sample_id)
    pd.testing.assert_frame_equal(
        reference.loc[fallback, PROBABILITY_COLUMNS],
        dollar.loc[fallback, PROBABILITY_COLUMNS],
        check_exact=True,
    )
    assert set(result["models"]) == {"reference", "dollar"}
    fold = next(item for item in make_walk_forward_folds() if item.name == f"test_{year}")
    train = select_block(table, fold.train, gap_minutes=181)
    supported = train.loc[context_usable_mask(train, profile=DOLLAR_PROFILE)]
    fitted = joblib.load(tmp_path / "final_model.joblib")
    assert fitted.preprocessor.train_sample_digest == sample_id_digest(supported.sample_id)
    assert fitted.preprocessor.train_row_count == len(supported) < len(train)
    assert set(fitted.preprocessor.feature_names) == set(PRICE_NAMES) | set(
        DOLLAR_PROFILE.model_features
    )


def test_dollar_outer_and_calibration_do_not_select_model_or_policy(tmp_path: Path) -> None:
    table = _table()
    _evaluate(table, tmp_path / "a")
    changed = table.copy()
    later = changed.prediction_time_utc.ge("2021-10-01T00:00:00Z")
    changed.loc[later, [*PRICE_NAMES, *DOLLAR_PROFILE.model_features]] *= 1000
    _evaluate(changed, tmp_path / "b")
    for name in ("selection.json", "training_audit.json"):
        assert json.loads((tmp_path / "a" / name).read_text()) == json.loads(
            (tmp_path / "b" / name).read_text()
        )


def test_dollar_unusable_context_and_changed_fold_fail_closed(tmp_path: Path) -> None:
    table = _table()
    reference = _reference(table)
    missing = table.copy()
    missing["dollar_is_missing"] = True
    missing = missing.drop(columns="dollar_usable")
    routed = route_context_predictions(missing, reference, None, profile=DOLLAR_PROFILE)
    assert routed.used_price_only_fallback.all()
    pd.testing.assert_frame_equal(reference[PROBABILITY_COLUMNS], routed[PROBABILITY_COLUMNS])
    with pytest.raises(ValueError, match="frozen 2022/2023/2024"):
        evaluate_context_fold(
            table,
            PRICE_NAMES,
            replace(make_walk_forward_folds()[0], name="changed"),
            reference,
            (),
            DecisionPolicy(),
            tmp_path,
            profile=DOLLAR_PROFILE,
        )


def _sealed_synthetic_run(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Build a complete fixture with a separate, explicitly synthetic reference pin.

    The production reference authentication algorithm is exercised unchanged;
    only its expected reference digest is scoped to the fabricated fixture. No
    real reference run, data preflight or historical market benchmark is opened.
    """
    root.mkdir()
    _identity_fixture(root, status="running")
    config = load_phase10_config("configs/phase10_dollar_ablation.yaml")
    _write(root / "resolved_config.json", config.model_dump(mode="json"))
    _write(root / "config.yaml", config.model_dump(mode="json"))
    source_path = root / "configs" / config.source_config
    shutil.copyfile(Path("configs") / config.source_config, source_path)
    # The silver identity helper's obsolete source file is not part of this contract.
    (root / "configs/phase10_silver_exploratory.yaml").unlink()
    source = load_source(source_path).model_dump(mode="json")
    _write(root / "source_config.json", source)
    preflight = json.loads((root / "preflight.json").read_text())
    preflight.update(
        protocol=DOLLAR_PROFILE.protocol,
        source=source,
        config_sha256=sha256_file(root / "config.yaml"),
        source_config_sha256=sha256_file(source_path),
    )
    _write(root / "preflight.json", preflight)
    run = json.loads((root / "run.json").read_text())
    run["config_sha256"] = sha256_file(root / "config.yaml")
    _write(root / "run.json", run)
    table = _table()
    table.to_parquet(root / "features.parquet", index=False)
    folds = {}
    reference_files = []
    for fold in make_walk_forward_folds():
        directory = root / "folds" / fold.name
        folds[fold.name] = _evaluate(table, directory, int(fold.name[-4:]))
        for target, origin in (
            ("reference_split.json", "split_audit.json"),
            ("reference_selection.json", "selection.json"),
            ("reference_evaluation.json", "reference/evaluation.json"),
        ):
            shutil.copyfile(directory / origin, directory / target)
        for local, original in {
            "reference/outer_predictions.parquet": "logistic/outer_predictions.parquet",
            "reference_inner_0.parquet": "logistic/selected_inner_0.parquet",
            "reference_inner_1.parquet": "logistic/selected_inner_1.parquet",
            "reference_split.json": "split_audit.json",
            "reference_selection.json": "logistic/selection.json",
            "reference_evaluation.json": "logistic/evaluation.json",
        }.items():
            entry = file_digest(directory / local, relative_to=root).model_dump(mode="json")
            entry["path"] = f"horizon_15/mvp/{fold.name}/{original}"
            reference_files.append(entry)
    completion = {"files": reference_files, "version": content_version({"files": reference_files})}
    _write(root / "reference_completion.json", completion)
    original_authenticate = artifacts._authenticated_reference

    def authenticate_fixture(path: Path, inventory: dict) -> None:
        with monkeypatch.context() as scoped:
            scoped.setattr(artifacts, "PHASE7_REFERENCE_COMPLETION", completion["version"])
            original_authenticate(path, inventory)

    monkeypatch.setattr(artifacts, "_authenticated_reference", authenticate_fixture)
    summary = json.loads((root / "summary.json").read_text())
    summary.update(
        protocol=DOLLAR_PROFILE.protocol,
        folds=folds,
        comparison=artifacts.summarize_phase10(list(folds.values()), profile=DOLLAR_PROFILE),
    )
    _write(root / "summary.json", summary)
    for name in artifacts._ROOT_FILES | {
        "configs/project_root.yaml",
        "configs/project_instrument.yaml",
        "configs/project_features.yaml",
        "configs/project_labels.yaml",
        "configs/project_costs.yaml",
        "configs/project_splits.yaml",
        "configs/project_model.yaml",
        "configs/project_backtest.yaml",
        "configs/features_phase7.yaml",
        "configs/phase7_champion.yaml",
        "source_snapshot/gold_forecasting/synthetic_fixture.py",
    }:
        path = root / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic-only fixture", encoding="utf-8")
    artifacts.complete_phase10_run(root)
    run["status"] = "succeeded"
    _write(root / "run.json", run)


def test_dollar_completion_and_validator_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "synthetic-dollar-run"
    _sealed_synthetic_run(root, monkeypatch)
    result = artifacts.verify_phase10_run(root)
    assert result["protocol"] == DOLLAR_PROFILE.protocol
    assert result["verified_files"] > 80
    assert result["champion_promotion"] is False
    assert result["trading_activation"] is False
    assert result["holdout_opened"] is False
    summary = json.loads((root / "summary.json").read_text())
    summary["champion_promotion"] = True
    _write(root / "summary.json", summary)
    inventory = artifacts._inventory(root)
    files = list(inventory.values())
    _write(root / "completion.json", {"files": files, "version": content_version({"files": files})})
    with pytest.raises(artifacts.Phase10ArtifactError, match="champion_promotion"):
        artifacts.verify_phase10_run(root)
