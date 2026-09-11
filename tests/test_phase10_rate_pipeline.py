"""Synthetic-only rate integration: frozen folds, fallback, sealing and replay."""

from __future__ import annotations

import hashlib
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
from gold_forecasting.classification import CLASS_ORDER
from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.evaluation.walk_forward import make_walk_forward_folds, select_block
from gold_forecasting.phase10 import artifacts
from gold_forecasting.phase10.config import load_phase10_config
from gold_forecasting.phase10.contracts import load_source
from gold_forecasting.phase10.profiles import RATE_PROFILE
from gold_forecasting.phase10.rate_fred import FRED_RATE_FILENAME, import_fred_dfii10
from gold_forecasting.phase10.real_preflight import (
    Phase10PreflightError,
    inspect_context_metadata,
    load_verified_context,
    validate_modeled_context_source,
)
from gold_forecasting.phase10.training import (
    context_usable_mask,
    evaluate_context_fold,
    route_context_predictions,
)
from tests.test_phase10_artifact_lifecycle import _identity_fixture
from tests.test_phase10_training import PRICE_NAMES


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _table() -> pd.DataFrame:
    rng = np.random.default_rng(10103)
    times = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range(
                    f"{year}-01-22T22:00:00Z",
                    f"{year}-12-20T22:00:00Z",
                    freq="B",
                ).to_numpy()
                for year in range(2020, 2025)
            ]
        )
    )
    n = len(times)
    classes = np.arange(n, dtype=np.int64) % 3
    returns = (classes - 1) * 20.0
    observed = times.normalize() - pd.Timedelta(days=1)
    table = pd.DataFrame(
        {
            "instrument": "XAU_USD",
            "source": "synthetic",
            "prediction_time_utc": times,
            "entry_time_utc": times + pd.Timedelta(minutes=1),
            "label_end_time_utc": times + pd.Timedelta(minutes=16),
            "feature_available_at_utc": times,
            "horizon_minutes": 15,
            "target_class": np.asarray(CLASS_ORDER)[classes],
            "target_class_id": classes,
            "arithmetic_return_bps": returns,
            "entry_bid_open": 2000.0,
            "exit_bid_open": 2000.0 * (1 + returns / 10000),
            "rate_observation_id": [f"synthetic-rate-{stamp:%Y-%m-%d}" for stamp in observed],
            "rate_observed_at_utc": observed,
            "rate_value": 1.0 + 0.4 * np.sin(np.arange(n) / 19.0),
            "rate_is_missing": False,
            "rate_is_stale": False,
            "rate_age_seconds": (times - observed).total_seconds(),
        }
    )
    table["sample_id"] = [
        hashlib.sha256(f"XAU_USD|synthetic|{stamp.isoformat()}|15".encode()).hexdigest()
        for stamp in times
    ]
    for name in PRICE_NAMES:
        table[name] = rng.normal(0, 1, n)
    # Force sparse, auditable fallback in every outer year without filtering gold rows.
    table.loc[table.index % 101 == 0, "rate_is_stale"] = True
    table = RATE_PROFILE.build_features(table)
    table["rate_usable"] = context_usable_mask(table, profile=RATE_PROFILE)
    return table


def _reference(rows: pd.DataFrame) -> pd.DataFrame:
    probabilities = np.tile([0.234567890123, 0.531234567891, 0.234197541986], (len(rows), 1))
    return prediction_records(rows, probabilities, np.full(len(rows), -0.017345678901))


def _evaluate(table: pd.DataFrame, directory: Path, year: int) -> dict:
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
        profile=RATE_PROFILE,
    )


def _fred_fixture(root: Path) -> tuple[object, object]:
    config = load_phase10_config("configs/phase10_rate_ablation.yaml")
    source = load_source("configs/phase10_rate_exploratory.yaml")
    raw = root / config.archive_directory
    raw.mkdir(parents=True)
    rows = [
        ("2020-01-02", "0.10"),
        ("2021-01-04", "0.20"),
        ("2022-01-03", "0.30"),
        ("2023-01-03", "0.40"),
        ("2024-01-02", "0.50"),
        ("2024-12-31", "0.60"),
    ]
    source_file = raw / FRED_RATE_FILENAME
    source_file.write_text(
        "observation_date,DFII10\n"
        + "\n".join(f"{stamp},{value}" for stamp, value in rows)
        + "\n",
        encoding="utf-8",
    )
    import_fred_dfii10(
        source_file,
        (root / config.bundle_path).parent,
        source,
        ingested_at_utc=pd.Timestamp("2026-09-10T00:00:00Z"),
    )
    return config, source


def test_rate_real_metadata_guard_and_authenticated_reparse(tmp_path: Path) -> None:
    config, source = _fred_fixture(tmp_path)
    evidence = inspect_context_metadata(tmp_path, config, source)
    observations, replayed = load_verified_context(
        tmp_path,
        config,
        source,
        metadata=evidence,
    )
    assert replayed == evidence
    assert len(observations) == 6
    assert observations.raw_sha256.nunique() == 1
    assert evidence["source_file"]["series_id"] == "DFII10"
    assert evidence["source_file"]["availability_is_historical_evidence"] is False

    artifacts_path = (tmp_path / config.bundle_path).parent / "source_artifacts.json"
    payload = json.loads(artifacts_path.read_text())
    payload["source_file"]["release_schedule"] = "09:00 UTC"
    artifacts_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Phase10PreflightError, match="frozen contract"):
        inspect_context_metadata(tmp_path, config, source)


def test_rate_profile_and_source_contract_are_frozen() -> None:
    config = load_phase10_config("configs/phase10_rate_ablation.yaml")
    assert config.profile == RATE_PROFILE
    assert config.test_years == (2022, 2023, 2024)
    assert config.gap_minutes == 181
    source = load_source("configs/phase10_rate_exploratory.yaml")
    validate_modeled_context_source(source, RATE_PROFILE)
    with pytest.raises(Phase10PreflightError, match="frozen modeled"):
        validate_modeled_context_source(
            source.model_copy(update={"availability_basis": "provider_timestamp"}),
            RATE_PROFILE,
        )


@pytest.mark.parametrize("year", [2022, 2023, 2024])
def test_rate_nested_folds_train_only_and_exact_fallback(tmp_path: Path, year: int) -> None:
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
    rate = pd.read_parquet(tmp_path / "rate/outer_predictions.parquet")
    fallback = rate.used_price_only_fallback
    assert fallback.any() and (~fallback).any()
    pd.testing.assert_series_equal(reference.sample_id, rate.sample_id)
    pd.testing.assert_frame_equal(
        reference.loc[fallback, PROBABILITY_COLUMNS],
        rate.loc[fallback, PROBABILITY_COLUMNS],
        check_exact=True,
    )
    assert set(result["models"]) == {"reference", "rate"}
    fold = next(item for item in make_walk_forward_folds() if item.name == f"test_{year}")
    train = select_block(table, fold.train, gap_minutes=181)
    supported = train.loc[context_usable_mask(train, profile=RATE_PROFILE)]
    fitted = joblib.load(tmp_path / "final_model.joblib")
    assert fitted.preprocessor.train_sample_digest == sample_id_digest(supported.sample_id)
    assert fitted.preprocessor.train_row_count == len(supported) < len(train)
    assert set(fitted.preprocessor.feature_names) == set(PRICE_NAMES) | set(
        RATE_PROFILE.model_features
    )


def test_rate_outer_and_calibration_cannot_change_selection(tmp_path: Path) -> None:
    table = _table()
    _evaluate(table, tmp_path / "a", 2022)
    changed = table.copy()
    later = changed.prediction_time_utc.ge("2021-10-01T00:00:00Z")
    changed.loc[later, [*PRICE_NAMES, *RATE_PROFILE.model_features]] *= 1000
    _evaluate(changed, tmp_path / "b", 2022)
    for name in ("selection.json", "training_audit.json"):
        assert json.loads((tmp_path / "a" / name).read_text()) == json.loads(
            (tmp_path / "b" / name).read_text()
        )


def test_rate_unusable_context_and_changed_fold_fail_closed(tmp_path: Path) -> None:
    table = _table()
    reference = _reference(table)
    missing = table.copy()
    missing["rate_is_missing"] = True
    missing = missing.drop(columns="rate_usable")
    routed = route_context_predictions(missing, reference, None, profile=RATE_PROFILE)
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
            profile=RATE_PROFILE,
        )


def _sealed_synthetic_run(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root.mkdir()
    _identity_fixture(root, status="running")
    config = load_phase10_config("configs/phase10_rate_ablation.yaml")
    _write(root / "resolved_config.json", config.model_dump(mode="json"))
    _write(root / "config.yaml", config.model_dump(mode="json"))
    source_path = root / "configs" / config.source_config
    shutil.copyfile(Path("configs") / config.source_config, source_path)
    (root / "configs/phase10_silver_exploratory.yaml").unlink()
    source = load_source(source_path).model_dump(mode="json")
    _write(root / "source_config.json", source)
    preflight = json.loads((root / "preflight.json").read_text())
    preflight.update(
        protocol=RATE_PROFILE.protocol,
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
        protocol=RATE_PROFILE.protocol,
        folds=folds,
        comparison=artifacts.summarize_phase10(list(folds.values()), profile=RATE_PROFILE),
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


def test_rate_completion_and_validator_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "synthetic-rate-run"
    _sealed_synthetic_run(root, monkeypatch)
    result = artifacts.verify_phase10_run(root)
    assert result["protocol"] == RATE_PROFILE.protocol
    assert result["verified_files"] > 80
    assert result["champion_promotion"] is False
    assert result["trading_activation"] is False
    assert result["holdout_opened"] is False
    summary = json.loads((root / "summary.json").read_text())
    summary["holdout_opened"] = True
    _write(root / "summary.json", summary)
    inventory = artifacts._inventory(root)
    files = list(inventory.values())
    _write(root / "completion.json", {"files": files, "version": content_version({"files": files})})
    with pytest.raises(artifacts.Phase10ArtifactError, match="holdout_opened"):
        artifacts.verify_phase10_run(root)
