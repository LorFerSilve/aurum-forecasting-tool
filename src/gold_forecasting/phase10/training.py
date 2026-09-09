"""Frozen classical methodology with exact, row-preserving silver fallback."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from gold_forecasting.artifacts import write_json_atomic, write_parquet_atomic
from gold_forecasting.backtesting.v1 import DecisionPolicy
from gold_forecasting.benchmark.config import BenchmarkConfig
from gold_forecasting.benchmark.models import (
    ALLOWED_FEATURE_NAMES,
    FittedModel,
    ModelSpec,
    candidate_specs,
    fit_model,
    prediction_records,
)
from gold_forecasting.benchmark.pipeline import (
    GAP_MINUTES,
    PROBABILITY_COLUMNS,
    _evaluate_predictions,
    _metrics,
    _save_model,
    select_policy,
)
from gold_forecasting.classification import validate_probability_matrix
from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.evaluation.walk_forward import (
    WalkForwardFold,
    make_walk_forward_folds,
    select_block,
    validate_development_frame,
)
from gold_forecasting.phase10.features import SILVER_FEATURE_NAMES

SILVER_MODEL_FEATURES = tuple(
    name for name in SILVER_FEATURE_NAMES if name not in {"silver_is_missing", "silver_is_stale"}
)
_PREDICTION_FIELDS = [*PROBABILITY_COLUMNS, "expected_return_bps", "predicted_class"]


def silver_usable_mask(table: pd.DataFrame) -> pd.Series:
    """Routing uses actual finite context, never imputed flags or levels."""
    for name in ("silver_is_missing", "silver_is_stale"):
        if not is_bool_dtype(table[name].dtype) or table[name].isna().any():
            raise ValueError("silver routing flags must be nonmissing booleans")
    finite = np.isfinite(table[list(SILVER_MODEL_FEATURES)].to_numpy(dtype=np.float64)).all(axis=1)
    usable = pd.Series(
        ~table["silver_is_missing"] & ~table["silver_is_stale"] & finite,
        index=table.index,
    )
    if "silver_usable" in table and not table["silver_usable"].equals(usable):
        raise ValueError("silver_usable disagrees with actual source feature availability")
    return usable


def context_coverage(table: pd.DataFrame, fallback: pd.Series | None = None) -> dict[str, Any]:
    usable = silver_usable_mask(table)
    ages = table["silver_age_seconds"].dropna().astype(float)
    missing, stale = table["silver_is_missing"], table["silver_is_stale"]
    return {
        "rows": len(table),
        "usable_rows": int(usable.sum()),
        "usable_fraction": float(usable.mean()) if len(table) else 0.0,
        "missing_rows": int(missing.sum()),
        "stale_rows": int(stale.sum()),
        "insufficient_history_rows": int((~missing & ~stale & ~usable).sum()),
        "fallback_rows": int((~usable if fallback is None else fallback).sum()),
        "fallback_fraction": float((~usable if fallback is None else fallback).mean()),
        "age_seconds": {
            "maximum": float(ages.max()) if len(ages) else None,
            "p05": float(ages.quantile(0.05)) if len(ages) else None,
            "p50": float(ages.quantile(0.50)) if len(ages) else None,
            "p95": float(ages.quantile(0.95)) if len(ages) else None,
        },
    }


def _aligned_reference(rows: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    validate_development_frame(rows)
    validate_development_frame(reference)
    if rows.empty or rows["sample_id"].duplicated().any():
        raise ValueError("prediction rows require a nonempty unique common universe")
    columns = [
        "sample_id",
        "prediction_time_utc",
        "entry_time_utc",
        "label_end_time_utc",
        "instrument",
        "source",
        "horizon_minutes",
        "target_class",
        "target_class_id",
        "arithmetic_return_bps",
        "entry_bid_open",
        "exit_bid_open",
    ]
    for name in columns:
        if not rows[name].reset_index(drop=True).equals(reference[name].reset_index(drop=True)):
            raise ValueError(f"frozen reference common-universe parity differs: {name}")
    validate_probability_matrix(reference[PROBABILITY_COLUMNS].to_numpy(), expected_rows=len(rows))
    if not np.isfinite(reference["expected_return_bps"].to_numpy()).all():
        raise ValueError("frozen expected returns must be finite")
    return reference.reset_index(drop=True).copy()


def route_silver_predictions(
    rows: pd.DataFrame,
    reference: pd.DataFrame,
    model: FittedModel | None,
) -> pd.DataFrame:
    """Copy frozen predictions first and replace only supported context rows.

    Expected returns and predicted classes also fall back exactly. The frozen
    probabilities never pass through clipping, renormalization or a refit.
    """
    result = _aligned_reference(rows, reference)
    rows = rows.reset_index(drop=True)
    usable = silver_usable_mask(rows)
    for name in (*SILVER_MODEL_FEATURES, "silver_is_missing", "silver_is_stale"):
        result[name] = rows[name]
    result["silver_usable"] = usable
    fallback = ~usable if model is not None else pd.Series(True, index=rows.index)
    result["used_price_only_fallback"] = fallback
    if model is not None and usable.any():
        subset = rows.loc[usable]
        probabilities, expected = model.predict(subset)
        predicted = prediction_records(subset, probabilities, expected)
        for name in _PREDICTION_FIELDS:
            result.loc[usable, name] = predicted[name].to_numpy()
    if not result.loc[fallback, _PREDICTION_FIELDS].equals(
        reference.reset_index(drop=True).loc[fallback, _PREDICTION_FIELDS]
    ):
        raise ValueError("exact frozen price-only fallback parity failed")
    return result


def _fit_silver(
    train: pd.DataFrame,
    names: tuple[str, ...],
    spec: ModelSpec,
    config: BenchmarkConfig,
) -> tuple[FittedModel | None, dict[str, Any]]:
    usable = silver_usable_mask(train)
    supported = train.loc[usable]
    audit: dict[str, Any] = {
        "gold_train_rows": len(train),
        "gold_train_digest": sample_id_digest(train["sample_id"]),
        "usable_train_rows": len(supported),
        "usable_train_digest": sample_id_digest(supported["sample_id"]),
        "fit_split": "train",
        "feature_names": list(names),
        "class_weight": "balanced",
    }
    if set(supported["target_class_id"].unique()) != {0, 1, 2}:
        audit.update(model_fitted=False, reason="insufficient_usable_train_class_support")
        return None, audit
    model = fit_model(
        supported,
        names,
        spec,
        config,
        allowed_feature_names=frozenset((*ALLOWED_FEATURE_NAMES, *SILVER_MODEL_FEATURES)),
    )
    preprocessor = model.preprocessor
    audit.update(
        model_fitted=True,
        train_sample_digest=preprocessor.train_sample_digest,
        train_row_count=preprocessor.train_row_count,
        imputer_statistics=preprocessor.pipeline.named_steps["imputer"].statistics_.tolist(),
        scaler_mean=preprocessor.pipeline.named_steps["scaler"].mean_.tolist(),
        scaler_scale=preprocessor.pipeline.named_steps["scaler"].scale_.tolist(),
        class_returns=model.class_returns.tolist(),
    )
    return model, audit


def evaluate_silver_fold(
    table: pd.DataFrame,
    price_feature_names: tuple[str, ...],
    fold: WalkForwardFold,
    reference_records: pd.DataFrame,
    reference_inner: tuple[pd.DataFrame, ...],
    reference_policy: DecisionPolicy,
    directory: Path,
) -> dict[str, Any]:
    """Select the six protocol candidates and policy on complete inner universes."""
    validate_development_frame(table)
    schedule = {item.name: item for item in make_walk_forward_folds()}
    if fold.name not in schedule or fold != schedule[fold.name]:
        raise ValueError("silver fold differs from the frozen 2022/2023/2024 schedule")
    if set(price_feature_names) != ALLOWED_FEATURE_NAMES or len(price_feature_names) != 18:
        raise ValueError("silver challenger requires the exact frozen MVP price feature basis")
    if len(reference_inner) != len(fold.inner_folds):
        raise ValueError("frozen inner reference fold count differs")
    names = (*price_feature_names, *SILVER_MODEL_FEATURES)
    config = BenchmarkConfig(horizons=(15,))
    train = select_block(table, fold.train, gap_minutes=GAP_MINUTES)
    calibration = select_block(table, fold.calibration, purge=False)
    test = select_block(table, fold.test, purge=False)
    if train.empty or calibration.empty or test.empty:
        raise ValueError("silver evaluation requires nonempty gold train/calibration/test blocks")
    _aligned_reference(test, reference_records)
    split = {
        "fold": fold.name,
        "gap_minutes": GAP_MINUTES,
        "train_rows": len(train),
        "test_rows": len(test),
        "calibration_rows_reserved": len(calibration),
        "train_digest": sample_id_digest(train["sample_id"]),
        "calibration_digest": sample_id_digest(calibration["sample_id"]),
        "test_digest": sample_id_digest(test["sample_id"]),
    }
    inner_blocks = []
    for inner, reference in zip(fold.inner_folds, reference_inner, strict=True):
        inner_train = select_block(table, inner.train, gap_minutes=GAP_MINUTES)
        validation = select_block(table, inner.validation, purge=False)
        _aligned_reference(validation, reference)
        if inner_train.empty:
            raise ValueError("silver inner training gold coverage is empty")
        inner_blocks.append((inner, inner_train, validation, reference))
    audits: list[dict[str, Any]] = []
    predictions: dict[str, list[pd.DataFrame]] = {}
    for spec in candidate_specs("logistic"):
        candidate_name = spec.name
        scores: list[dict[str, Any]] = []
        records = []
        for inner, inner_train, validation, reference in inner_blocks:
            model, evidence = _fit_silver(inner_train, names, spec, config)
            predicted = route_silver_predictions(validation, reference, model)
            scores.append(
                {
                    "inner_fold": inner.name,
                    "metrics": _metrics(predicted),
                    "training": evidence,
                    "validation_rows": len(validation),
                    "validation_sample_digest": sample_id_digest(validation["sample_id"]),
                    "context": context_coverage(
                        validation, predicted["used_price_only_fallback"]
                    ),
                }
            )
            records.append(predicted)
        audits.append(
            {
                "name": candidate_name,
                "spec": asdict(spec),
                "class_weight": "balanced",
                "folds": scores,
                "mean_macro_f1": float(np.mean([s["metrics"]["macro_f1"] for s in scores])),
                "mean_log_loss": float(np.mean([s["metrics"]["log_loss"] for s in scores])),
            }
        )
        predictions[candidate_name] = records
    selected = sorted(audits, key=lambda a: (-a["mean_macro_f1"], a["mean_log_loss"]))[0]
    selected_records = predictions[selected["name"]]
    policy, policy_audit = select_policy(selected_records, config.minimum_policy_trades)
    selection = {
        "candidates": audits,
        "selected_name": selected["name"],
        "selected_spec": selected["spec"],
        "class_weight": "balanced",
        "selected_policy": asdict(policy),
        "policy_candidates": policy_audit,
        "calibration_status": "reserved_not_fitted",
    }
    for index, (predicted, reference) in enumerate(
        zip(selected_records, reference_inner, strict=True)
    ):
        write_parquet_atomic(directory / f"selected_inner_{index}.parquet", predicted)
        write_parquet_atomic(directory / f"reference_inner_{index}.parquet", reference)
    model, training_audit = _fit_silver(
        train,
        names,
        ModelSpec(**selected["spec"]),
        config,
    )
    predicted = route_silver_predictions(test, reference_records, model)
    if model is not None:
        checkpoint = directory / "final_model.joblib"
        _save_model(model, checkpoint)
        reloaded = route_silver_predictions(test, reference_records, joblib.load(checkpoint))
        if not predicted.equals(reloaded):
            raise ValueError("saved silver model prediction parity failed")
    training_audit["checkpoint_prediction_parity"] = True
    training_audit["context"] = context_coverage(train)
    write_json_atomic(directory / "training_audit.json", training_audit)
    write_json_atomic(directory / "split_audit.json", split)
    write_json_atomic(directory / "selection.json", selection)
    results = {
        "reference": _evaluate_predictions(
            reference_records,
            reference_policy,
            directory / "reference",
        ),
        "silver": _evaluate_predictions(predicted, policy, directory / "silver"),
    }
    return {
        "split": split,
        "models": results,
        "context": context_coverage(test, predicted["used_price_only_fallback"]),
    }
