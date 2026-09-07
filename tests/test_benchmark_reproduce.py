from __future__ import annotations

import importlib.metadata
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gold_forecasting.artifacts import (
    content_version,
    file_digest,
    write_json_atomic,
    write_parquet_atomic,
)
from gold_forecasting.benchmark import reproduce
from gold_forecasting.benchmark.config import BenchmarkConfig
from gold_forecasting.benchmark.pipeline import GAP_MINUTES, evaluate_fold
from gold_forecasting.benchmark.reproduce import (
    BenchmarkReproductionError,
    reproduce_benchmark,
)
from gold_forecasting.evaluation.walk_forward import WalkForwardError, make_walk_forward_folds
from gold_forecasting.features.mvp import FeatureCatalog, FeatureDefinition
from tests.test_benchmark_pipeline import FEATURE_NAMES, _synthetic_table


def _completion(directory: Path) -> None:
    files = [
        file_digest(path, relative_to=directory).model_dump(mode="json")
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in {"run.json", "completion.json"}
    ]
    write_json_atomic(
        directory / "completion.json",
        {"files": files, "version": content_version({"files": files})},
    )


def _change_json(directory: Path, relative: str, updates: dict[str, Any]) -> None:
    path = directory / relative
    current = json.loads(path.read_text(encoding="utf-8"))
    current.update(updates)
    write_json_atomic(path, current)
    _completion(directory)


@pytest.fixture(scope="module")
def saved_benchmark(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("saved-phase6-reproduction")
    table = _synthetic_table()
    config = BenchmarkConfig(
        horizons=(15,),
        test_years=(2022,),
        xgb_max_rounds=2,
        xgb_early_stopping_rounds=1,
        minimum_policy_trades=1,
    )
    folds = make_walk_forward_folds(config.test_years)
    manifest_versions = {
        "1min": {"dataset_version": "sha256:" + "1" * 64},
        "3min": {"dataset_version": "sha256:" + "3" * 64},
    }
    data_version = content_version(
        {
            "minute": manifest_versions["1min"]["dataset_version"],
            "features": manifest_versions["3min"]["dataset_version"],
        }
    )
    # JSON is valid YAML, preserving the frozen root config without a writer dependency.
    write_json_atomic(directory / "config.yaml", config.model_dump(mode="json"))
    write_json_atomic(directory / "resolved_config.json", config.model_dump(mode="json"))
    write_json_atomic(directory / "source_manifests.json", manifest_versions)
    write_json_atomic(
        directory / "dependencies.json",
        {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "pyarrow", "joblib")
        },
    )
    catalog = FeatureCatalog(
        feature_spec_version=content_version({"synthetic_feature_names": FEATURE_NAMES}),
        definitions=tuple(
            FeatureDefinition(
                name=name,
                formula="synthetic causal fixture",
                lookback_candles=1,
                lookback_minutes=3,
            )
            for name in FEATURE_NAMES
        ),
    )
    write_json_atomic(directory / "feature_catalog.json", catalog.model_dump(mode="json"))
    write_json_atomic(
        directory / "schedule.json",
        {"gap_minutes": GAP_MINUTES, "folds": [fold.as_record() for fold in folds]},
    )
    write_parquet_atomic(directory / "horizon_15/model_table.parquet", table)
    evaluate_fold(table, FEATURE_NAMES, folds[0], config, directory / "horizon_15/test_2022")
    write_json_atomic(
        directory / "summary.json",
        {
            "protocol": config.protocol_version,
            "holdout_opened": False,
            "data_version": data_version,
            "horizons": {"15": {}},
        },
    )
    write_json_atomic(
        directory / "run.json",
        {"run_id": "synthetic-reproduction", "status": "succeeded", "data_version": data_version},
    )
    _completion(directory)
    return directory


@pytest.fixture
def copied_benchmark(saved_benchmark: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(saved_benchmark, tmp_path / "copied-run"))


def _inventory(directory: Path) -> list[tuple[str, str, int]]:
    return [
        (
            path.relative_to(directory).as_posix(),
            file_digest(path).sha256,
            path.stat().st_mtime_ns,
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def _must_not_fit(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("unsafe metadata or observations reached model fitting")


def test_reproduction_exactly_refits_final_candidates_without_modifying_run(
    saved_benchmark: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _inventory(saved_benchmark)
    actual_fit = reproduce.fit_model
    fits: list[tuple[str, int | None]] = []

    def record_fit(*args: Any, **kwargs: Any) -> Any:
        assert kwargs.get("validation") is None
        fits.append((args[2].family, kwargs.get("rounds")))
        return actual_fit(*args, **kwargs)

    monkeypatch.setattr(reproduce, "fit_model", record_fit)
    report = reproduce_benchmark(saved_benchmark)
    assert report["run_id"] == "synthetic-reproduction"
    assert report["final_fit_parity_count"] == 4
    assert report["inner_policy_parity_count"] == 4
    assert report["holdout_opened"] is False
    assert report["original_run_modified"] is False
    assert [family for family, _ in fits] == ["reference", "logistic", "ridge", "xgboost"]
    assert [rounds for _, rounds in fits[:3]] == [None, None, None]
    assert fits[3][1] in (1, 2)
    assert all(item["exact_prediction_parity"] for item in report["parities"])
    assert all(item["test_rows"] == 48 for item in report["parities"])
    assert report["scope"]["all_inner_candidates_refitted"] is False
    assert report["scope"]["model_selection_replayed_from_saved_scores"] is True
    assert report["scope"]["features_and_labels_regenerated"] is False
    assert _inventory(saved_benchmark) == before
    assert json.loads(json.dumps(report)) == report


def test_reproduction_verifies_artifacts_before_reading_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed_verification(path: Path) -> dict[str, Any]:
        assert path == tmp_path
        raise ValueError("hash verification gate")

    monkeypatch.setattr(reproduce, "verify_benchmark", failed_verification)
    monkeypatch.setattr(reproduce, "fit_model", _must_not_fit)
    with pytest.raises(ValueError, match="hash verification gate"):
        reproduce_benchmark(tmp_path)


@pytest.mark.parametrize(
    "relative,updates,expected",
    [
        ("dependencies.json", {"numpy": "0.0.not-installed"}, "dependency version mismatch"),
        ("resolved_config.json", {"horizons": [3]}, "configuration disagree"),
        ("schedule.json", {"gap_minutes": 31}, "schedules disagree"),
        ("summary.json", {"data_version": "sha256:" + "0" * 64}, "data versions disagree"),
        ("summary.json", {"holdout_opened": True}, "development-only protocol"),
        ("summary.json", {"horizons": {}}, "horizon inventories disagree"),
    ],
)
def test_reproduction_rejects_incompatible_metadata_before_fitting(
    copied_benchmark: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
    updates: dict[str, Any],
    expected: str,
) -> None:
    _change_json(copied_benchmark, relative, updates)
    monkeypatch.setattr(reproduce, "fit_model", _must_not_fit)
    with pytest.raises(BenchmarkReproductionError, match=expected):
        reproduce_benchmark(copied_benchmark)


def test_reproduction_rejects_incomplete_dependency_snapshot(
    copied_benchmark: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = copied_benchmark / "dependencies.json"
    dependencies = json.loads(path.read_text(encoding="utf-8"))
    del dependencies["numpy"]
    write_json_atomic(path, dependencies)
    _completion(copied_benchmark)
    monkeypatch.setattr(reproduce, "fit_model", _must_not_fit)
    with pytest.raises(BenchmarkReproductionError, match="dependency snapshot is incomplete"):
        reproduce_benchmark(copied_benchmark)


def test_reproduction_rejects_reserved_holdout_rows_before_fitting(
    copied_benchmark: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = copied_benchmark / "horizon_15/model_table.parquet"
    table = pd.read_parquet(path)
    holdout = table.iloc[:1].copy()
    prediction = pd.Timestamp("2025-01-01T00:00:00Z")
    for column, delta in (
        ("prediction_time_utc", 0),
        ("entry_time_utc", 1),
        ("label_end_time_utc", 16),
        ("feature_available_at_utc", 0),
        ("feature_window_start_utc", -60),
    ):
        holdout[column] = prediction + pd.Timedelta(minutes=delta)
    holdout["sample_id"] = "forbidden-holdout"
    write_parquet_atomic(path, pd.concat([table, holdout], ignore_index=True))
    _completion(copied_benchmark)
    monkeypatch.setattr(reproduce, "fit_model", _must_not_fit)
    with pytest.raises(WalkForwardError, match="reserved holdout"):
        reproduce_benchmark(copied_benchmark)


@pytest.mark.parametrize("name", ["outer_predictions", "selected_inner_0"])
def test_reproduction_rejects_changed_sample_order_before_fitting(
    copied_benchmark: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    path = copied_benchmark / f"horizon_15/test_2022/reference/{name}.parquet"
    records = pd.read_parquet(path)
    write_parquet_atomic(path, records.iloc[::-1])
    _completion(copied_benchmark)
    monkeypatch.setattr(reproduce, "fit_model", _must_not_fit)
    with pytest.raises(BenchmarkReproductionError, match="ordered sample ID parity"):
        reproduce_benchmark(copied_benchmark)


@pytest.mark.parametrize(
    "column,expected",
    [("p_up", "probability parity"), ("expected_return_bps", "expected return parity")],
)
def test_reproduction_demands_exact_float_parity(
    copied_benchmark: Path, column: str, expected: str
) -> None:
    path = copied_benchmark / "horizon_15/test_2022/reference/outer_predictions.parquet"
    records = pd.read_parquet(path)
    value = records.loc[0, column]
    records.loc[0, column] = np.nextafter(value, np.inf)
    write_parquet_atomic(path, records)
    _completion(copied_benchmark)
    with pytest.raises(BenchmarkReproductionError, match=expected):
        reproduce_benchmark(copied_benchmark)


def test_reproduction_rechecks_inner_policy_selection(
    copied_benchmark: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = "horizon_15/test_2022/reference/selection.json"
    _change_json(
        copied_benchmark,
        relative,
        {"selected_policy": {"confidence_threshold": 1.0, "min_expected_net_bps": 1e11}},
    )
    monkeypatch.setattr(reproduce, "fit_model", _must_not_fit)
    with pytest.raises(BenchmarkReproductionError, match="inner-selected policy parity"):
        reproduce_benchmark(copied_benchmark)


def test_reproduction_rechecks_inner_policy_candidate_audit(
    copied_benchmark: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = copied_benchmark / "horizon_15/test_2022/reference/selection.json"
    selection = json.loads(path.read_text(encoding="utf-8"))
    selection["policy_candidates"][0]["mean_net_bps"] += 1
    write_json_atomic(path, selection)
    _completion(copied_benchmark)
    monkeypatch.setattr(reproduce, "fit_model", _must_not_fit)
    with pytest.raises(BenchmarkReproductionError, match="inner policy audit parity"):
        reproduce_benchmark(copied_benchmark)


def test_reproduction_requires_every_read_input_to_be_hash_verified(
    copied_benchmark: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = copied_benchmark / "completion.json"
    completion = json.loads(path.read_text(encoding="utf-8"))
    files = [entry for entry in completion["files"] if entry["path"] != "dependencies.json"]
    write_json_atomic(path, {"files": files, "version": content_version({"files": files})})
    # Isolate this defense from any stronger inventory validation in verify_benchmark.
    monkeypatch.setattr(reproduce, "verify_benchmark", lambda _: {})
    monkeypatch.setattr(reproduce, "fit_model", _must_not_fit)
    with pytest.raises(BenchmarkReproductionError, match="required artifact is not hash-verified"):
        reproduce_benchmark(copied_benchmark)


@pytest.mark.parametrize(
    "change,expected",
    [
        ("inventory", "frozen candidate inventory parity"),
        ("inner_fold", "candidate inner fold parity"),
        ("mean", "candidate mean score parity"),
        ("round_audit", "candidate round audit parity"),
        ("selected_rounds", "inner-selected model audit parity"),
        ("selected_spec", "inner-selected model audit parity"),
    ],
)
def test_reproduction_checks_frozen_model_selection_audit(
    saved_benchmark: Path, change: str, expected: str
) -> None:
    config = BenchmarkConfig.model_validate_json(
        (saved_benchmark / "resolved_config.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (saved_benchmark / "horizon_15/test_2022/xgboost/selection.json").read_text(
            encoding="utf-8"
        )
    )
    if change == "inventory":
        selection["candidates"].pop()
    elif change == "inner_fold":
        selection["candidates"][0]["folds"][0]["inner_fold"] = "test_2022"
    elif change == "mean":
        selection["candidates"][0]["mean_macro_f1"] += 0.01
    elif change == "round_audit":
        selection["candidates"][0]["rounds"][0] = 0
    elif change == "selected_rounds":
        selection["selected_rounds"] = 3 - selection["selected_rounds"]
    elif change == "selected_spec":
        selected_value = selection["selected_spec"]["value"]
        selection["selected_spec"]["value"] = 2.0 if selected_value != 2.0 else 3.0
    with pytest.raises(BenchmarkReproductionError, match=expected):
        reproduce._check_candidate_selection(
            selection,
            "xgboost",
            make_walk_forward_folds(config.test_years)[0],
            config,
            context="synthetic candidate audit",
        )
