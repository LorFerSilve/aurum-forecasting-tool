"""Synthetic ingestion-to-ablation proof, explicitly separate from market evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gold_forecasting.artifacts import (
    file_digest,
    sha256_file,
    write_json_atomic,
    write_parquet_atomic,
)
from gold_forecasting.classification import CLASS_ORDER
from gold_forecasting.datasets.preprocessing import fit_train_preprocessor, sample_id_digest
from gold_forecasting.evaluation.metrics import compute_classification_metrics
from gold_forecasting.models.logistic import predict_logistic, select_logistic_candidate
from gold_forecasting.phase10.bundle import load_context_bundle
from gold_forecasting.phase10.contracts import TIME_COLUMNS, ContextSource
from gold_forecasting.phase10.events import EVENT_COLUMNS, join_events
from gold_forecasting.phase10.features import (
    PRICE_FEATURE_NAMES,
    SILVER_FEATURE_NAMES,
    build_silver_features,
)
from gold_forecasting.phase10.point_in_time import context_coverage, join_context
from gold_forecasting.registry import get_git_code_version

PROTOCOL = "phase10-context-smoke-v1"
ARTIFACTS = (
    "source.json", "observations.csv", "observations.manifest.json",
    "events.parquet", "features.parquet", "predictions.parquet",
    "missing_source_predictions.parquet", "summary.json",
)


def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, ContextSource]:
    """Known simulated releases, outages and five-step labels; no market files."""
    source = ContextSource(
        source_id="silver", description="Synthetic silver integration fixture",
        source_url="synthetic://phase10", market_hours="continuous synthetic 3min grid",
        source_timezone="UTC", publication_delay_seconds=60, stale_after_seconds=600,
        availability_basis="synthetic", revision_policy="append_only",
        availability_evidence="Generated release timestamps; not market evidence", enabled=True,
    )
    rng = np.random.default_rng(20260909)
    times = pd.date_range("2024-01-08", periods=900, freq="3min", tz="UTC")
    gold_steps = rng.normal(0, 2, len(times))
    gold_close = 2000 * np.exp(np.cumsum(gold_steps) / 10_000)
    silver_close = 24 * np.exp(np.cumsum(0.5 * gold_steps + rng.normal(0, 3, len(times))) / 10_000)
    predictions = pd.DataFrame({
        "sample_id": [f"synthetic-{i:04d}" for i in range(len(times))],
        "prediction_time_utc": times,
        "gold_close": gold_close,
    })
    future_return = 10_000 * (predictions["gold_close"].shift(-5) / gold_close - 1)
    predictions["target"] = np.select(
        [future_return.lt(-3), future_return.gt(3)], ["down", "up"], default="neutral"
    )
    predictions["target_end_utc"] = predictions["prediction_time_utc"] + pd.Timedelta(minutes=15)
    predictions = predictions.iloc[:-5].copy()
    # Freshness must degrade inside train and test, and later recover without fill.
    keep = np.ones(len(times), dtype=bool)
    keep[220:240] = False
    keep[790:825] = False
    observed = times - pd.Timedelta(seconds=60)
    observations = pd.DataFrame({
        "source_id": "silver",
        "observation_id": [f"silver-{i:04d}" for i in range(len(times))],
        "observed_at_utc": observed,
        "available_at_utc": times,
        "ingested_at_utc": times + pd.Timedelta(seconds=5),
        "value": silver_close,
        "revision_id": "synthetic-v1",
        "source_uri": "synthetic://phase10/silver",
        "raw_sha256": "0" * 64,
    }).loc[keep].reset_index(drop=True)
    return predictions, observations, source


def _smoke_ablation(features: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Same chronological rows, train-only scaling and exact fallback predictions.

    These two small models are plumbing fixtures, not the frozen Phase-7 model.
    Both boundaries purge the synthetic label's full future interval.
    """
    table = features.dropna(subset=list(PRICE_FEATURE_NAMES)).copy()
    validation_start = table["prediction_time_utc"].iloc[int(len(table) * 0.6)]
    test_start = table["prediction_time_utc"].iloc[int(len(table) * 0.8)]
    train_mask = table["target_end_utc"].lt(validation_start)
    validation_mask = (
        table["prediction_time_utc"].ge(validation_start) & table["target_end_utc"].lt(test_start)
    )
    test_mask = table["prediction_time_utc"].ge(test_start)
    table["split"] = np.select(
        [train_mask, validation_mask, test_mask], ["train", "validation", "test"], default="purged"
    )
    train, validation, test = (
        table.loc[table["split"].eq(split)] for split in ("train", "validation", "test")
    )
    prediction_frames: list[pd.DataFrame] = []
    summaries: dict[str, Any] = {}
    price_probabilities: np.ndarray[Any, Any] | None = None
    for name, names in (
        ("price_only_fixture", PRICE_FEATURE_NAMES),
        ("silver_fixture", (*PRICE_FEATURE_NAMES, *SILVER_FEATURE_NAMES)),
    ):
        context_supported = train[list(SILVER_FEATURE_NAMES)].notna().all(axis=1).any()
        fit_context = name == "silver_fixture" and context_supported
        if name == "price_only_fixture" or fit_context:
            preprocessor = fit_train_preprocessor(table, tuple(names))
            model = select_logistic_candidate(
                preprocessor.transform(train), train["target"],
                preprocessor.transform(validation), validation["target"],
                c_values=(1.0,), class_weight_options=(None,), seed=20260909,
            )
            probabilities = predict_logistic(
                model.estimator, preprocessor.transform(test)
            ).probabilities.copy()
        else:
            assert price_probabilities is not None
            probabilities = price_probabilities.copy()
        unavailable = (
            test[list(SILVER_FEATURE_NAMES)].isna().any(axis=1).to_numpy(dtype=bool)
        )
        # Availability flags cannot serve as a substitute for the actual source.
        unavailable |= test["silver_is_missing"].to_numpy() | test["silver_is_stale"].to_numpy()
        if name == "price_only_fixture":
            price_probabilities = probabilities.copy()
            fallback_count = 0
        else:
            assert price_probabilities is not None
            if not fit_context:
                unavailable[:] = True
            probabilities[unavailable] = price_probabilities[unavailable]
            fallback_count = int(unavailable.sum())
        report = compute_classification_metrics(test["target"], probabilities).to_dict()
        summaries[name] = {
            **report, "feature_names": list(names), "fallback_rows": fallback_count,
            "context_model_fitted": bool(fit_context),
            "train_sample_digest": sample_id_digest(train["sample_id"]),
            "test_sample_digest": sample_id_digest(test["sample_id"]),
        }
        predicted = test[["sample_id", "prediction_time_utc", "target"]].copy()
        predicted["variant"] = name
        predicted["used_price_only_fallback"] = unavailable if fallback_count else False
        for i, label in enumerate(CLASS_ORDER):
            predicted["p_" + label] = probabilities[:, i]
        prediction_frames.append(predicted)
    return {
        "split_rows": table["split"].value_counts().to_dict(),
        "validation_start_utc": validation_start.isoformat(),
        "test_start_utc": test_start.isoformat(),
        "purge_minutes": 15,
        "variants": summaries,
    }, pd.concat(prediction_frames, ignore_index=True)


def _synthetic_events(times: pd.Series) -> pd.DataFrame:
    """Scheduled events with a known reschedule and a known cancellation."""
    records = []
    for event_id, event_type, scheduled, released, revision, status in (
        ("cpi", "us_cpi", 300, 5, "v1", "scheduled"),
        ("jobs", "us_jobs", 500, 5, "v1", "scheduled"),
        ("jobs", "us_jobs", 510, 480, "v2", "scheduled"),
        ("fomc", "fomc", 700, 5, "v1", "scheduled"),
        ("fomc", "fomc", 700, 690, "v2", "cancelled"),
    ):
        records.append({
            "event_id": event_id, "event_type": event_type,
            "scheduled_at_utc": times.iloc[scheduled],
            "available_at_utc": times.iloc[released],
            "revision_id": revision, "status": status,
            "source_uri": "synthetic://phase10/calendar", "raw_sha256": "0" * 64,
        })
    return pd.DataFrame(records, columns=EVENT_COLUMNS)


def run_context_dry_run(output: str | Path) -> dict[str, Any]:
    """Create a new immutable smoke-run directory and verify its completion."""
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    predictions, observations, source = synthetic_inputs()
    write_json_atomic(destination / "source.json", source.model_dump(mode="json"))
    serialized = observations.copy()
    for column in TIME_COLUMNS:
        serialized[column] = observations[column].map(lambda value: value.isoformat())
    serialized.to_csv(destination / "observations.csv", index=False)
    write_json_atomic(destination / "observations.manifest.json", {
        "schema_version": 1, "source_id": source.source_id,
        "availability_basis": source.availability_basis,
        "observation_start_utc": observations["observed_at_utc"].min().isoformat(),
        "observation_end_utc": (
            observations["observed_at_utc"].max() + pd.Timedelta(seconds=1)
        ).isoformat(),
        "row_count": len(observations), "file": "observations.csv",
        "sha256": sha256_file(destination / "observations.csv"),
    })
    loaded = load_context_bundle(destination / "observations.manifest.json", source)
    joined = join_context(predictions, loaded, source)
    events = _synthetic_events(predictions["prediction_time_utc"])
    write_parquet_atomic(destination / "events.parquet", events)
    features = join_events(build_silver_features(joined), events)
    write_parquet_atomic(destination / "features.parquet", features)
    ablation, predicted = _smoke_ablation(features)
    write_parquet_atomic(destination / "predictions.parquet", predicted)
    # Exercise a complete absent-source path using exactly the same gold rows.
    missing = build_silver_features(join_context(predictions, None, source))
    if not features[list(PRICE_FEATURE_NAMES)].equals(missing[list(PRICE_FEATURE_NAMES)]):
        raise ValueError("optional context changed the price-only feature path")
    missing_ablation, missing_predicted = _smoke_ablation(missing)
    write_parquet_atomic(destination / "missing_source_predictions.parquet", missing_predicted)
    summary = {
        "protocol": PROTOCOL, "status": "completed", "synthetic_only": True,
        "formal_benchmark": False, "holdout_opened": False,
        "champion_changed": False, "trading_enabled": False,
        "code_version": get_git_code_version(),
        "coverage": context_coverage(joined, "silver"),
        "missing_source_coverage": context_coverage(missing, "silver"),
        "ablation": ablation,
        "missing_source_ablation": missing_ablation,
        "event_calendar": {
            "missing_rows": int(features["event_is_missing"].sum()),
            "phase_counts": features["event_phase"].dropna().value_counts().to_dict(),
            "complete_calendar_claimed": False,
            "used_as_model_features": False,
        },
    }
    write_json_atomic(destination / "summary.json", summary)
    write_json_atomic(destination / "completion.json", {
        "protocol": PROTOCOL,
        "files": [file_digest(destination / name, relative_to=destination).model_dump()
                  for name in ARTIFACTS],
    })
    return verify_context_dry_run(destination)


def verify_context_dry_run(output: str | Path) -> dict[str, Any]:
    """Verify the fixed artifact inventory and explicit synthetic-only scope."""
    destination = Path(output).resolve(strict=True)
    completion = json.loads((destination / "completion.json").read_text(encoding="utf-8"))
    if completion.get("protocol") != PROTOCOL:
        raise ValueError("unsupported context smoke protocol")
    records = completion.get("files")
    if not isinstance(records, list) or [r.get("path") for r in records] != list(ARTIFACTS):
        raise ValueError("context smoke artifact inventory differs")
    for expected in records:
        path = destination / expected["path"]
        if path.is_symlink() or path.resolve() != path:
            raise ValueError("context artifact is redirected")
        if file_digest(path, relative_to=destination).model_dump() != expected:
            raise ValueError(f"context artifact integrity differs: {path.name}")
    summary: dict[str, Any] = json.loads(
        (destination / "summary.json").read_text(encoding="utf-8")
    )
    scope = {
        "protocol": PROTOCOL, "status": "completed", "synthetic_only": True,
        "formal_benchmark": False, "holdout_opened": False,
        "champion_changed": False, "trading_enabled": False,
    }
    if any(summary.get(key) != value for key, value in scope.items()):
        raise ValueError("context smoke scope differs from the synthetic-only contract")
    return {**summary, "verified_files": len(records), "output_directory": str(destination)}
