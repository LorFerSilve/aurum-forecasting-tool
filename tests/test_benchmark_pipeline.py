from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest

from gold_forecasting.artifacts import content_version, file_digest, write_json_atomic
from gold_forecasting.benchmark.config import BenchmarkConfig
from gold_forecasting.benchmark.pipeline import (
    CASH_POLICY,
    FAMILIES,
    POLICIES,
    _metrics,
    evaluate_fold,
    select_policy,
    verify_benchmark,
)
from gold_forecasting.classification import CLASS_ORDER
from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.evaluation.walk_forward import make_walk_forward_folds, select_block

FEATURE_NAMES = (
    "close_log_return_1_bps",
    "momentum_5_bps",
    "realized_volatility_20_bps",
    "candle_body_bps",
)
ALL_MODELS = {
    *FAMILIES,
    "most_frequent",
    "always_up",
    "always_down",
    "last_candle_direction",
    "momentum",
    "mean_reversion",
    "empirical_priors",
}


def _synthetic_table() -> pd.DataFrame:
    starts = (
        "2020-06-01T00:00:00Z",
        "2021-01-15T00:00:00Z",
        "2021-04-15T00:00:00Z",
        "2021-07-15T00:00:00Z",
        "2021-10-15T00:00:00Z",
        "2022-01-15T00:00:00Z",
        "2022-06-15T00:00:00Z",
    )
    frames = []
    for block_number, start in enumerate(starts):
        times = pd.date_range(start, periods=24, freq="30min")
        row = np.arange(len(times))
        targets = row % 3
        signal = (targets - 1) * 10.0 + np.sin(row) * 0.2
        returns = np.array([-18.0, 0.5, 20.0])[targets] + np.cos(row) * 0.2
        entry = 1500.0 + row + block_number * 50
        frames.append(
            pd.DataFrame(
                {
                    "sample_id": [f"block_{block_number}_row_{number}" for number in row],
                    "instrument": "XAUUSD",
                    "source": "synthetic",
                    "prediction_time_utc": times,
                    "entry_time_utc": times + pd.Timedelta(minutes=1),
                    "label_end_time_utc": times + pd.Timedelta(minutes=16),
                    "feature_available_at_utc": times,
                    "feature_window_start_utc": times - pd.Timedelta(minutes=60),
                    "horizon_minutes": 15,
                    "entry_bid_open": entry,
                    "exit_bid_open": entry * (1.0 + returns / 10_000),
                    "arithmetic_return_bps": returns,
                    "future_return_bps": np.log1p(returns / 10_000) * 10_000,
                    "future_range_bps": np.abs(returns) + 3,
                    "future_realized_vol_bps": np.sqrt(np.abs(returns) + 1),
                    "target_class": np.array(CLASS_ORDER)[targets],
                    "target_class_id": targets,
                    "close_log_return_1_bps": signal,
                    "momentum_5_bps": signal * 2,
                    "realized_volatility_20_bps": np.sqrt(np.abs(signal) + 1),
                    "candle_body_bps": signal + np.cos(row) * 0.3,
                    "label_spec_version": "synthetic-phase6-v1",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _mutate_future_values(table: pd.DataFrame) -> pd.DataFrame:
    changed = table.copy(deep=True)
    future = changed["prediction_time_utc"].ge(pd.Timestamp("2021-10-01T00:00:00Z"))
    changed.loc[future, list(FEATURE_NAMES)] = changed.loc[future, list(FEATURE_NAMES)] * -7 + 1000
    changed.loc[future, "arithmetic_return_bps"] *= -5
    changed.loc[future, "entry_bid_open"] *= 2
    changed.loc[future, "exit_bid_open"] = changed.loc[future, "entry_bid_open"] * (
        1 + changed.loc[future, "arithmetic_return_bps"] / 10_000
    )
    changed.loc[future, "future_return_bps"] = (
        np.log1p(changed.loc[future, "arithmetic_return_bps"] / 10_000) * 10_000
    )
    changed.loc[future, ["future_range_bps", "future_realized_vol_bps"]] *= 3
    labels = np.select(
        [changed["future_return_bps"].lt(-6), changed["future_return_bps"].gt(6)],
        [0, 2],
        default=1,
    )
    changed["target_class_id"] = labels
    changed["target_class"] = np.array(CLASS_ORDER)[labels]
    return changed


@pytest.fixture(scope="module")
def evaluated_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    directory = tmp_path_factory.mktemp("phase6-synthetic-fold")
    table = _synthetic_table()
    changed = _mutate_future_values(table)
    original = table.copy(deep=True)
    config = BenchmarkConfig(
        horizons=(15,),
        test_years=(2022,),
        xgb_max_rounds=2,
        xgb_early_stopping_rounds=1,
        minimum_policy_trades=1,
    )
    fold = make_walk_forward_folds((2022,))[0]
    baseline = evaluate_fold(table, FEATURE_NAMES, fold, config, directory / "baseline")
    mutated = evaluate_fold(changed, FEATURE_NAMES, fold, config, directory / "mutated")
    pd.testing.assert_frame_equal(table, original)
    return {
        "directory": directory,
        "table": table,
        "changed_table": changed,
        "baseline": baseline,
        "mutated": mutated,
        "fold": fold,
    }


def test_evaluate_fold_uses_identical_samples_and_persists_auditable_outputs(
    evaluated_runs: dict[str, Any],
) -> None:
    results = evaluated_runs["baseline"]
    expected = select_block(evaluated_runs["table"], evaluated_runs["fold"].test, purge=False)
    ids = expected["sample_id"].tolist()
    digest = sample_id_digest(expected["sample_id"])
    assert set(results["models"]) == ALL_MODELS
    assert results["split"]["test_digest"] == digest
    assert results["split"]["test_rows"] == 48
    assert results["split"]["train_rows"] == 96
    assert results["split"]["calibration_rows_reserved"] == 24
    for name, metrics in results["models"].items():
        destination = evaluated_runs["directory"] / "baseline" / name
        assert metrics["sample_digest"] == digest
        records = pd.read_parquet(destination / "outer_predictions.parquet")
        assert records["sample_id"].tolist() == ids
        assert records["fold"].unique().tolist() == ["test_2022"]
        assert metrics["base_backtest"]["record_count"] == len(ids)
        assert metrics["stress_backtest"]["record_count"] == len(ids)
        assert np.isfinite(records[[f"p_{label}" for label in CLASS_ORDER]].to_numpy()).all()
        for artifact in (
            "evaluation.json",
            "base_decisions.parquet",
            "base_trades.parquet",
            "stress_trades.parquet",
        ):
            assert (destination / artifact).is_file()
        if name in FAMILIES:
            for artifact in ("selection.json", "final_model.joblib", "selected_inner_0.parquet"):
                assert (destination / artifact).is_file()
    assert set(results["continuous_baselines"]) == {
        "zero_return_mae_bps",
        "historical_volatility_mae_bps",
        "training_median_range_mae_bps",
    }
    persisted = json.loads(
        (evaluated_runs["directory"] / "baseline" / "fold_summary.json").read_text(encoding="utf-8")
    )
    assert persisted == json.loads(json.dumps(results))


def test_calibration_and_outer_mutation_does_not_change_models_or_policy_selection(
    evaluated_runs: dict[str, Any],
) -> None:
    assert evaluated_runs["baseline"]["split"] == evaluated_runs["mutated"]["split"]
    probe = select_block(evaluated_runs["table"], evaluated_runs["fold"].test, purge=False)
    for family in FAMILIES:
        original_path = evaluated_runs["directory"] / "baseline" / family
        mutated_path = evaluated_runs["directory"] / "mutated" / family
        first = json.loads((original_path / "selection.json").read_text(encoding="utf-8"))
        second = json.loads((mutated_path / "selection.json").read_text(encoding="utf-8"))
        for key in ("selected_spec", "selected_rounds", "selected_policy", "policy_candidates"):
            assert first[key] == second[key]
        assert first["calibration_status"] == second["calibration_status"] == "reserved_not_fitted"
        for left, right in zip(first["candidates"], second["candidates"], strict=True):
            for first_fold, second_fold in zip(left["folds"], right["folds"], strict=True):
                assert first_fold["train_sample_digest"] == second_fold["train_sample_digest"]
                assert (
                    first_fold["validation_sample_digest"]
                    == second_fold["validation_sample_digest"]
                )
                assert first_fold["metrics"] == second_fold["metrics"]
        first_model = joblib.load(original_path / "final_model.joblib")
        second_model = joblib.load(mutated_path / "final_model.joblib")
        assert first_model.preprocessor.train_sample_digest == (
            second_model.preprocessor.train_sample_digest
        )
        for first_values, second_values in zip(
            first_model.predict(probe), second_model.predict(probe), strict=True
        ):
            np.testing.assert_array_equal(first_values, second_values)
        for index in (0, 1):
            pd.testing.assert_frame_equal(
                pd.read_parquet(original_path / f"selected_inner_{index}.parquet"),
                pd.read_parquet(mutated_path / f"selected_inner_{index}.parquet"),
            )
    assert (
        evaluated_runs["baseline"]["models"]["ridge"]["classification"]
        != (evaluated_runs["mutated"]["models"]["ridge"]["classification"])
    )


def _policy_records(*, actual_return: float, confidence: float = 0.85) -> pd.DataFrame:
    records = _synthetic_table().iloc[:6].copy()
    records["predicted_class"] = "up"
    records["p_down"] = (1 - confidence) / 2
    records["p_neutral"] = (1 - confidence) / 2
    records["p_up"] = confidence
    records["expected_return_bps"] = 20.0
    records["exit_bid_open"] = records["entry_bid_open"] * (1 + actual_return / 10_000)
    return records


def test_policy_selection_falls_back_to_cash_for_negative_inner_net_returns() -> None:
    policy, audit = select_policy([_policy_records(actual_return=-20)] * 2, minimum_trades=1)
    assert policy == CASH_POLICY
    assert len(audit) == len(POLICIES)
    assert all(attempt["minimum_trades_met"] for attempt in audit)
    assert all(attempt["mean_net_bps"] < 0 for attempt in audit)


def test_policy_selection_requires_enough_trades_in_every_inner_fold() -> None:
    eligible = _policy_records(actual_return=20)
    policy, audit = select_policy([eligible, eligible.iloc[:1]], minimum_trades=2)
    assert policy == CASH_POLICY
    assert all(attempt["mean_net_bps"] > 0 for attempt in audit)
    assert all(not attempt["minimum_trades_met"] for attempt in audit)


def test_policy_selection_uses_fixed_candidate_order_to_break_ties() -> None:
    policy, audit = select_policy([_policy_records(actual_return=20)] * 2, minimum_trades=1)
    assert policy == POLICIES[0]
    assert audit[0]["policy"] == asdict(POLICIES[0])
    assert len({attempt["mean_net_bps"] for attempt in audit}) == 1


def test_reliability_bins_partition_rows_including_exact_boundaries() -> None:
    records = _policy_records(actual_return=20).iloc[:5].copy()
    confidence = np.array([0.5, 0.6, 0.8, 0.9, 1.0])
    records["p_down"] = (1 - confidence) / 2
    records["p_neutral"] = (1 - confidence) / 2
    records["p_up"] = confidence
    metrics = _metrics(records)
    assert sum(bin_record["rows"] for bin_record in metrics["reliability_bins"]) == len(records)
    correct = records["target_class_id"].eq(2).to_numpy()
    expected_error = float(np.mean(np.abs(confidence - correct)))
    assert metrics["expected_calibration_error"] == pytest.approx(expected_error)


def _verified_run(directory: Path, *, status: str = "succeeded") -> Path:
    directory.mkdir()
    write_json_atomic(directory / "run.json", {"run_id": "synthetic", "status": status})
    write_json_atomic(directory / "summary.json", {"holdout_opened": False, "value": 1})
    files = [file_digest(directory / "summary.json", relative_to=directory).model_dump(mode="json")]
    write_json_atomic(
        directory / "completion.json",
        {"files": files, "version": content_version({"files": files})},
    )
    return directory


def test_verify_benchmark_accepts_intact_artifact_manifest(tmp_path: Path) -> None:
    directory = _verified_run(tmp_path / "run")
    result = verify_benchmark(directory)
    assert result["run_id"] == "synthetic"
    assert result["verified_files"] == 1


@pytest.mark.parametrize("status", ["running", "failed"])
def test_verify_benchmark_rejects_incomplete_or_failed_run(tmp_path: Path, status: str) -> None:
    directory = _verified_run(tmp_path / "run", status=status)
    with pytest.raises(ValueError, match="did not succeed"):
        verify_benchmark(directory)


def test_verify_benchmark_rejects_changed_artifact(tmp_path: Path) -> None:
    directory = _verified_run(tmp_path / "run")
    write_json_atomic(directory / "summary.json", {"holdout_opened": False, "value": 2})
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        verify_benchmark(directory)


def test_verify_benchmark_rejects_modified_completion_version(tmp_path: Path) -> None:
    directory = _verified_run(tmp_path / "run")
    completion_path = directory / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["version"] = "sha256:" + "0" * 64
    write_json_atomic(completion_path, completion)
    with pytest.raises(ValueError, match="completion manifest version mismatch"):
        verify_benchmark(directory)


def test_verify_benchmark_rejects_missing_artifact(tmp_path: Path) -> None:
    directory = _verified_run(tmp_path / "run")
    (directory / "summary.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_benchmark(directory)


def test_verify_benchmark_rejects_path_outside_run(tmp_path: Path) -> None:
    directory = _verified_run(tmp_path / "run")
    outside = tmp_path / "outside.json"
    write_json_atomic(outside, {"outside": True})
    digest = file_digest(outside).model_dump(mode="json")
    digest["path"] = "../outside.json"
    files = [digest]
    write_json_atomic(
        directory / "completion.json",
        {"files": files, "version": content_version({"files": files})},
    )
    with pytest.raises(ValueError, match="artifact escapes"):
        verify_benchmark(directory)
