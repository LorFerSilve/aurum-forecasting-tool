"""Actual logistic integration with sparse development fixtures and adversarial routing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from gold_forecasting.backtesting.v1 import DecisionPolicy
from gold_forecasting.benchmark.config import BenchmarkConfig
from gold_forecasting.benchmark.models import (
    ALLOWED_FEATURE_NAMES,
    ModelSpec,
    fit_model,
    prediction_records,
)
from gold_forecasting.benchmark.pipeline import PROBABILITY_COLUMNS
from gold_forecasting.classification import CLASS_ORDER
from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.evaluation.walk_forward import make_walk_forward_folds, select_block
from gold_forecasting.phase10.training import (
    SILVER_MODEL_FEATURES,
    _fit_silver,
    evaluate_silver_fold,
    route_silver_predictions,
    silver_usable_mask,
)

PRICE_NAMES = tuple(sorted(ALLOWED_FEATURE_NAMES))


def synthetic_table() -> pd.DataFrame:
    """No local market data: deliberately sparse blocks exercise all frozen years."""
    rng = np.random.default_rng(1010)
    times = pd.DatetimeIndex(
        [
            stamp
            for year in range(2020, 2025)
            for month in (1, 4, 7, 11)
            for stamp in pd.date_range(f"{year}-{month:02d}-02T00:00:00Z", periods=36, freq="3min")
        ]
    )
    n = len(times)
    classes = np.arange(n, dtype=np.int64) % 3
    returns = (classes - 1) * 20.0
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
            "silver_is_missing": False,
            "silver_is_stale": False,
        }
    )
    table["sample_id"] = [
        hashlib.sha256(f"XAU_USD|synthetic|{stamp.isoformat()}|15".encode()).hexdigest()
        for stamp in times
    ]
    for name in PRICE_NAMES:
        table[name] = rng.normal(0, 1, n)
    for name in SILVER_MODEL_FEATURES:
        table[name] = rng.normal(0, 1, n)
    table["silver_age_seconds"] = 60.0
    table.loc[table.index % 11 == 0, "silver_is_missing"] = True
    table.loc[table.index % 13 == 0, "silver_is_stale"] = True
    table.loc[table.index % 7 == 0, "silver_momentum_5_bps"] = np.nan
    return table


def frozen_predictions(rows: pd.DataFrame, fold: str) -> pd.DataFrame:
    # Fixed fixture outputs intentionally distinct from any fitted challenger.
    probabilities = np.tile([0.234567890123, 0.531234567891, 0.234197541986], (len(rows), 1))
    expected = np.full(len(rows), -0.017345678901)
    gold = rows.drop(columns=[*SILVER_MODEL_FEATURES, "silver_is_missing", "silver_is_stale"])
    return prediction_records(gold.assign(fold=fold), probabilities, expected).reset_index(
        drop=True
    )


def evaluate_fixture(table: pd.DataFrame, directory: Path, year: int = 2022) -> dict:
    fold = next(f for f in make_walk_forward_folds() if f.name == f"test_{year}")
    outer = frozen_predictions(select_block(table, fold.test, purge=False), fold.name)
    inner = tuple(
        frozen_predictions(select_block(table, part.validation, purge=False), fold.name)
        for part in fold.inner_folds
    )
    return evaluate_silver_fold(table, PRICE_NAMES, fold, outer, inner, DecisionPolicy(), directory)


@pytest.mark.parametrize("year", [2022, 2023, 2024])
def test_actual_nested_logistic_preserves_gold_folds_and_fallback(
    tmp_path: Path, year: int
) -> None:
    table = synthetic_table()
    result = evaluate_fixture(table, tmp_path, year)
    selected = json.loads((tmp_path / "selection.json").read_text())
    assert len(selected["candidates"]) == 3
    assert {(c["spec"]["value"], c["class_weight"]) for c in selected["candidates"]} == {
        (value, "balanced") for value in (0.1, 1.0, 10.0)
    }
    assert selected["calibration_status"] == "reserved_not_fitted"
    reference = pd.read_parquet(tmp_path / "reference/outer_predictions.parquet")
    challenger = pd.read_parquet(tmp_path / "silver/outer_predictions.parquet")
    fallback = challenger["used_price_only_fallback"]
    assert fallback.any() and (~fallback).any()
    np.testing.assert_array_equal(reference.sample_id, challenger.sample_id)
    np.testing.assert_array_equal(
        reference.loc[fallback, PROBABILITY_COLUMNS],
        challenger.loc[fallback, PROBABILITY_COLUMNS],
    )
    assert (
        result["models"]["reference"]["sample_digest"]
        == result["models"]["silver"]["sample_digest"]
    )
    fold = next(f for f in make_walk_forward_folds() if f.name == f"test_{year}")
    train = select_block(table, fold.train, gap_minutes=181)
    fitted = train.loc[silver_usable_mask(train)]
    model = joblib.load(tmp_path / "final_model.joblib")
    assert model.preprocessor.train_sample_digest == sample_id_digest(fitted.sample_id)
    assert model.preprocessor.train_row_count == len(fitted) < len(train)
    assert set(model.preprocessor.feature_names) == set(PRICE_NAMES) | set(SILVER_MODEL_FEATURES)


def test_outer_and_reserved_calibration_mutation_cannot_change_model_or_policy(
    tmp_path: Path,
) -> None:
    table = synthetic_table()
    evaluate_fixture(table, tmp_path / "original")
    changed = table.copy()
    future = changed.prediction_time_utc.ge("2021-10-01T00:00:00Z")
    changed.loc[future, [*PRICE_NAMES, *SILVER_MODEL_FEATURES]] *= 10000
    changed.loc[future, "target_class_id"] = 2 - changed.loc[future, "target_class_id"]
    changed.loc[future, "target_class"] = np.asarray(CLASS_ORDER)[
        changed.loc[future, "target_class_id"]
    ]
    changed.loc[future, "arithmetic_return_bps"] *= -1
    changed.loc[future, "exit_bid_open"] = 2000 * (
        1 + changed.loc[future, "arithmetic_return_bps"] / 10000
    )
    evaluate_fixture(changed, tmp_path / "changed")
    for file in ("selection.json", "training_audit.json"):
        assert json.loads((tmp_path / "original" / file).read_text()) == json.loads(
            (tmp_path / "changed" / file).read_text()
        )
    a, b = (joblib.load(tmp_path / name / "final_model.joblib") for name in ("original", "changed"))
    np.testing.assert_array_equal(a.estimator.coef_, b.estimator.coef_)


def test_absent_source_uses_exact_frozen_predictions_without_any_fit(
    tmp_path: Path, monkeypatch
) -> None:
    table = synthetic_table().assign(silver_is_missing=True)
    monkeypatch.setattr(
        "gold_forecasting.phase10.training.fit_model",
        lambda *args, **kwargs: pytest.fail("absent context must never fit a model"),
    )
    evaluate_fixture(table, tmp_path)
    ref = pd.read_parquet(tmp_path / "reference/outer_predictions.parquet")
    got = pd.read_parquet(tmp_path / "silver/outer_predictions.parquet")
    assert got.used_price_only_fallback.all()
    np.testing.assert_array_equal(ref[PROBABILITY_COLUMNS], got[PROBABILITY_COLUMNS])
    np.testing.assert_array_equal(ref.expected_return_bps, got.expected_return_bps)
    assert not (tmp_path / "final_model.joblib").exists()


def test_missing_values_are_never_passed_to_silver_predictor() -> None:
    table = synthetic_table().iloc[:36].copy()
    reference = frozen_predictions(table, "test_2022")

    class Spy:
        def predict(self, frame):
            assert np.isfinite(frame[list(SILVER_MODEL_FEATURES)].to_numpy()).all()
            assert not frame.silver_is_missing.any() and not frame.silver_is_stale.any()
            return np.tile([0.2, 0.2, 0.6], (len(frame), 1)), np.full(len(frame), 3.0)

    got = route_silver_predictions(table, reference, Spy())
    assert got.used_price_only_fallback.sum() == (~silver_usable_mask(table)).sum()
    stale = got.silver_is_stale | got.silver_is_missing
    np.testing.assert_array_equal(
        got.loc[stale, PROBABILITY_COLUMNS], reference.loc[stale, PROBABILITY_COLUMNS]
    )


@pytest.mark.parametrize("mutation", ["order", "subset", "target", "time", "holdout", "flag"])
def test_prediction_alignment_and_holdout_guards(mutation: str) -> None:
    rows = synthetic_table().iloc[:36].copy()
    ref = frozen_predictions(rows, "test_2022")
    if mutation == "order":
        ref = ref.iloc[::-1]
    elif mutation == "subset":
        ref = ref.iloc[:-1]
    elif mutation == "target":
        ref.loc[0, "arithmetic_return_bps"] += 1
    elif mutation == "time":
        ref.loc[0, "prediction_time_utc"] += pd.Timedelta(minutes=1)
    elif mutation == "holdout":
        rows.loc[0, "prediction_time_utc"] = pd.Timestamp("2025-01-01T00:00:00Z")
    else:
        rows["silver_usable"] = True
    with pytest.raises(ValueError):
        route_silver_predictions(rows, ref, None)


def test_fold_alignment_rejected_before_fit(tmp_path: Path) -> None:
    table = synthetic_table()
    fold = replace(make_walk_forward_folds()[0], name="unfrozen")
    with pytest.raises(ValueError, match=r"frozen.*schedule"):
        evaluate_silver_fold(table, PRICE_NAMES, fold, table, (), DecisionPolicy(), tmp_path)


def test_train_missing_source_or_missing_class_does_not_impute_a_source() -> None:
    table = synthetic_table()
    usable = silver_usable_mask(table)
    table.loc[usable & table.target_class_id.eq(2), "silver_is_missing"] = True
    model, audit = _fit_silver(
        table,
        (*PRICE_NAMES, *SILVER_MODEL_FEATURES),
        ModelSpec("logistic", 0.1),
        BenchmarkConfig(),
    )
    assert model is None and audit["model_fitted"] is False


def test_shared_trainer_retains_frozen_phase7_balanced_weighting() -> None:
    table = synthetic_table()
    model = fit_model(table, PRICE_NAMES, ModelSpec("logistic", 0.1), BenchmarkConfig())
    assert model.estimator.class_weight == "balanced"
