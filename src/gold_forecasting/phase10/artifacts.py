"""Fail-closed, replayable evidence for the frozen exploratory silver ablation.

Validation reads only JSON, text and Parquet. Serialized estimators are inventoried
but never deserialized. A copied, cryptographically pinned Phase-7 inventory also
authenticates the reference predictions: merely rehashing a changed fallback
baseline must not turn it into the frozen champion.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from gold_forecasting.artifacts import (
    content_version,
    file_digest,
    sha256_file,
    write_json_atomic,
)
from gold_forecasting.backtesting.v1 import DecisionPolicy, run_backtest_v1
from gold_forecasting.benchmark.pipeline import (
    BASE_COSTS,
    PROBABILITY_COLUMNS,
    STRESS_COSTS,
    _metrics,
    aggregate_model_metrics,
    select_policy,
)
from gold_forecasting.classification import CLASS_ORDER, validate_probability_matrix
from gold_forecasting.config import _load_yaml_mapping
from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.evaluation.walk_forward import (
    WalkForwardFold,
    make_walk_forward_folds,
    select_block,
    validate_development_frame,
)
from gold_forecasting.phase7.pipeline import _economic_gate, _predictive_gate
from gold_forecasting.phase10.config import Phase10Config
from gold_forecasting.phase10.contracts import ContextSource
from gold_forecasting.phase10.features import SILVER_FEATURE_NAMES, build_silver_features
from gold_forecasting.phase10.real_preflight import validate_modeled_silver_source
from gold_forecasting.phase10.reference import PHASE7_REFERENCE_COMPLETION, PHASE7_REFERENCE_RUN

PROTOCOL = "phase10-silver-modeled-v1"
_YEARS = [2022, 2023, 2024]
_MODELED_FEATURES = tuple(
    name for name in SILVER_FEATURE_NAMES if name not in {"silver_is_missing", "silver_is_stale"}
)
_PREDICTION_VALUES = [*PROBABILITY_COLUMNS, "expected_return_bps", "predicted_class"]
_ROOT_FILES = {
    "config.yaml",
    "resolved_config.json",
    "source_config.json",
    "preflight.json",
    "schedule.json",
    "dependencies.json",
    "requirements.lock",
    "protocol.md",
    "features.parquet",
    "summary.json",
    "summary.md",
    "reference_completion.json",
}
_FOLD_FILES = {
    "split_audit.json",
    "selection.json",
    "training_audit.json",
    "reference_split.json",
    "reference_selection.json",
    "reference_evaluation.json",
    "selected_inner_0.parquet",
    "selected_inner_1.parquet",
    "reference_inner_0.parquet",
    "reference_inner_1.parquet",
}
_EVALUATION_FILES = {
    "outer_predictions.parquet",
    "evaluation.json",
    "base_decisions.parquet",
    "base_trades.parquet",
    "stress_trades.parquet",
}


class Phase10ArtifactError(ValueError):
    """Persisted evidence violates the frozen exploratory research contract."""


def _json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in items:
            if name in result:
                raise Phase10ArtifactError(f"duplicate JSON key in {path.name}: {name}")
            result[name] = value
        return result

    def invalid_constant(value: str) -> None:
        raise Phase10ArtifactError(f"nonfinite JSON value in {path.name}: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase10ArtifactError(f"invalid JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise Phase10ArtifactError(f"JSON artifact must be an object: {path.name}")
    return value


def _equal(actual: Any, expected: Any, context: str) -> None:
    """Numerical replay tolerates only machine-level arithmetic differences."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise Phase10ArtifactError(f"{context}: object keys differ")
        for key, value in expected.items():
            _equal(actual[key], value, f"{context}/{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise Phase10ArtifactError(f"{context}: sequence differs")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _equal(left, right, f"{context}/{index}")
    elif isinstance(expected, float):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not np.isfinite(actual)
            or not np.isclose(actual, expected, rtol=1e-12, atol=1e-12)
        ):
            raise Phase10ArtifactError(f"{context}: numerical replay differs")
    elif type(actual) is not type(expected) or actual != expected:
        raise Phase10ArtifactError(f"{context}: value differs")


def _same_frame(actual: pd.DataFrame, expected: pd.DataFrame, context: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_exact=True,
            check_dtype=False,
        )
    except AssertionError as exc:
        raise Phase10ArtifactError(f"{context}: row/value parity differs") from exc


def context_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize routing, including feature warm-up and observed age distribution."""
    for column in (
        "silver_usable",
        "silver_is_missing",
        "silver_is_stale",
        "used_price_only_fallback",
    ):
        if (
            column not in frame
            or not is_bool_dtype(frame[column].dtype)
            or frame[column].isna().any()
        ):
            raise Phase10ArtifactError(f"{column} must contain nonmissing booleans")
    usable = frame["silver_usable"]
    missing = frame["silver_is_missing"]
    stale = frame["silver_is_stale"]
    fallback = frame["used_price_only_fallback"]
    if (usable & (missing | stale)).any():
        raise Phase10ArtifactError("missing/stale source marked usable")
    if ((~usable) & (~fallback)).any():
        raise Phase10ArtifactError("unusable silver context bypassed the frozen fallback")
    ages = frame["silver_age_seconds"].dropna().astype(float)
    if not np.isfinite(ages).all() or (ages < 0).any():
        raise Phase10ArtifactError("invalid silver ages")
    return {
        "rows": len(frame),
        "usable_rows": int(usable.sum()),
        "missing_rows": int(missing.sum()),
        "stale_rows": int(stale.sum()),
        "insufficient_history_rows": int((~usable & ~missing & ~stale).sum()),
        "fallback_rows": int(fallback.sum()),
        "fallback_fraction": float(fallback.mean()) if len(frame) else 0.0,
        "usable_fraction": float(usable.mean()) if len(frame) else 0.0,
        "age_seconds": {
            "maximum": float(ages.max()) if len(ages) else None,
            "p05": float(ages.quantile(0.05)) if len(ages) else None,
            "p50": float(ages.quantile(0.5)) if len(ages) else None,
            "p95": float(ages.quantile(0.95)) if len(ages) else None,
        },
    }


def summarize_phase10(fold_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Use the shared Phase-7 metric aggregation and exact admission gates."""
    if len(fold_summaries) != 3 or any(
        set(fold["models"]) != {"reference", "silver"} for fold in fold_summaries
    ):
        raise Phase10ArtifactError("comparison requires three aligned reference/silver folds")
    models = aggregate_model_metrics(fold_summaries)
    predictive = _predictive_gate(models["silver"], models["reference"])
    economic = _economic_gate(models["silver"], models["reference"])
    return {
        "models": models,
        "predictive_gate_passed": predictive,
        "economic_gate_passed": economic,
        "decision": "seek_strict_source" if predictive else "stop",
        "evidence_status": (
            "exploratory_signal_worth_strict_source"
            if predictive
            else "exploratory_no_predictive_admission"
        ),
        "champion_promotion": False,
        "trading_activation": False,
    }


def _safe_relative(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or "\x00" in value
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or PurePosixPath(value).as_posix() != value
    ):
        raise Phase10ArtifactError("unsafe artifact path")
    return value


def _inventory(root: Path) -> dict[str, dict[str, Any]]:
    """Walk without following links; reject even unlisted links and extra files."""
    if root.is_symlink():
        raise Phase10ArtifactError("symlink run directory is forbidden")
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise Phase10ArtifactError("symlink/junction artifacts are forbidden")
        if path.is_file() and path.name not in {"completion.json", "run.json"}:
            relative = _safe_relative(path.relative_to(root).as_posix())
            result[relative] = file_digest(path, relative_to=root).model_dump(mode="json")
        elif not path.is_dir() and path.name not in {"completion.json", "run.json"}:
            raise Phase10ArtifactError("artifact is not a regular file")
    return result


def _parse_inventory(completion: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if set(completion) != {"files", "version"} or not isinstance(completion["files"], list):
        raise Phase10ArtifactError("invalid completion manifest")
    if completion["version"] != content_version({"files": completion["files"]}):
        raise Phase10ArtifactError("completion content version mismatch")
    result: dict[str, dict[str, Any]] = {}
    folded: set[str] = set()
    for entry in completion["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size_bytes"}:
            raise Phase10ArtifactError("invalid inventory entry")
        path = _safe_relative(entry["path"])
        if path.casefold() in folded:
            raise Phase10ArtifactError("duplicate artifact path")
        if (
            not isinstance(entry["sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]) is None
            or type(entry["size_bytes"]) is not int
            or entry["size_bytes"] < 0
        ):
            raise Phase10ArtifactError("invalid artifact digest")
        folded.add(path.casefold())
        result[path] = entry
    return result


def _expected_files(root: Path, paths: set[str]) -> None:
    required = set(_ROOT_FILES)
    resolved = Phase10Config.model_validate(_json(root / "resolved_config.json"))
    required.update(
        {
            "configs/project_root.yaml",
            "configs/project_instrument.yaml",
            "configs/project_features.yaml",
            "configs/project_labels.yaml",
            "configs/project_costs.yaml",
            "configs/project_splits.yaml",
            "configs/project_model.yaml",
            "configs/project_backtest.yaml",
            f"configs/{Path(resolved.features_config).name}",
            f"configs/{Path(resolved.source_config).name}",
            f"configs/{Path(resolved.phase7_champion_config).name}",
        }
    )
    optional: set[str] = set()
    for year in _YEARS:
        prefix = f"folds/test_{year}/"
        required.update(prefix + name for name in _FOLD_FILES)
        optional.add(prefix + "final_model.joblib")
        for variant in ("reference", "silver"):
            required.update(prefix + variant + "/" + name for name in _EVALUATION_FILES)
    source = {
        name for name in paths if name.startswith("source_snapshot/") and name.endswith(".py")
    }
    if not source:
        raise Phase10ArtifactError("source snapshot is missing")
    if required - paths or paths - required - optional - source:
        raise Phase10ArtifactError(
            "artifact inventory differs from required contract: "
            f"missing={sorted(required - paths)}, "
            f"extra={sorted(paths - required - optional - source)}"
        )


def _authenticated_reference(root: Path, actual: dict[str, dict[str, Any]]) -> None:
    completion = _json(root / "reference_completion.json")
    canonical = _parse_inventory(completion)
    if completion["version"] != PHASE7_REFERENCE_COMPLETION:
        raise Phase10ArtifactError("Phase-7 reference completion differs from cryptographic pin")
    for year in _YEARS:
        prefix = f"folds/test_{year}/"
        phase7 = f"horizon_15/mvp/test_{year}/"
        mapping = {
            "reference/outer_predictions.parquet": "logistic/outer_predictions.parquet",
            "reference_inner_0.parquet": "logistic/selected_inner_0.parquet",
            "reference_inner_1.parquet": "logistic/selected_inner_1.parquet",
            "reference_split.json": "split_audit.json",
            "reference_selection.json": "logistic/selection.json",
            "reference_evaluation.json": "logistic/evaluation.json",
        }
        for local, reference in mapping.items():
            saved, pinned = actual.get(prefix + local), canonical.get(phase7 + reference)
            if (
                saved is None
                or pinned is None
                or saved["sha256"] != pinned["sha256"]
                or saved["size_bytes"] != pinned["size_bytes"]
            ):
                raise Phase10ArtifactError(f"frozen Phase-7 reference digest mismatch: {local}")


def _identity(
    root: Path,
    *,
    expected_run_status: str = "succeeded",
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary, run = _json(root / "summary.json"), _json(root / "run.json")
    contract = {
        "protocol": PROTOCOL,
        "run_mode": "exploratory_modeled_latency",
        "holdout_opened": False,
        "test_years": _YEARS,
        "gap_minutes": 181,
        "horizon_minutes": 15,
        "champion_promotion": False,
        "trading_activation": False,
        "baseline": {
            "run": PHASE7_REFERENCE_RUN,
            "completion": PHASE7_REFERENCE_COMPLETION,
            "variant": "mvp",
            "family": "logistic",
        },
    }
    for key, value in contract.items():
        _equal(summary.get(key), value, f"summary/{key}")
    if expected_run_status not in {"running", "succeeded"}:
        raise Phase10ArtifactError("unsupported expected run status")
    if run.get("status") != expected_run_status or run.get("run_id") != root.name:
        raise Phase10ArtifactError(
            f"run must have its own identity and {expected_run_status} status"
        )

    resolved = Phase10Config.model_validate(_json(root / "resolved_config.json"))
    snapshot = Phase10Config.model_validate(_load_yaml_mapping(root / "config.yaml"))
    _equal(
        snapshot.model_dump(mode="json"),
        resolved.model_dump(mode="json"),
        "config snapshot/resolved config",
    )
    config_digest = sha256_file(root / "config.yaml")
    if run.get("config_sha256") != config_digest:
        raise Phase10ArtifactError("registry config hash differs from sealed config snapshot")

    source_path = root / "configs" / Path(resolved.source_config).name
    source = ContextSource.model_validate(_load_yaml_mapping(source_path))
    validate_modeled_silver_source(source)
    source_payload = source.model_dump(mode="json")
    _equal(_json(root / "source_config.json"), source_payload, "source config JSON/YAML")

    preflight = _json(root / "preflight.json")
    if (
        preflight.get("status") != "passed"
        or preflight.get("exploratory_ablation_ready") is not True
        or preflight.get("formal_benchmark_ready") is not False
        or preflight.get("strict_pit_source_ready") is not False
        or preflight.get("formal_run_opened") is not False
        or preflight.get("holdout_opened") is not False
        or preflight.get("champion_changed") is not False
        or preflight.get("trading_activated") is not False
    ):
        raise Phase10ArtifactError("run lacks passed exploratory-only preflight")
    if preflight.get("config_sha256") != config_digest:
        raise Phase10ArtifactError("preflight config hash differs from sealed config snapshot")
    if preflight.get("source_config_sha256") != sha256_file(source_path):
        raise Phase10ArtifactError("preflight source hash differs from sealed source config")
    _equal(preflight.get("source"), source_payload, "preflight/source config")
    for name in ("code_version", "data_version"):
        if not isinstance(summary.get(name), str) or not summary[name]:
            raise Phase10ArtifactError(f"missing run {name}")
        for label, value in (("run", run), ("preflight", preflight)):
            _equal(value.get(name), summary[name], f"{label}/{name}")
    if summary["code_version"] in {"unavailable", "uncommitted"} or summary[
        "code_version"
    ].endswith("+dirty"):
        raise Phase10ArtifactError("exploratory market run must use committed clean code")
    schedule = _json(root / "schedule.json")
    _equal(
        schedule,
        {
            "gap_minutes": 181,
            "folds": [fold.as_record() for fold in make_walk_forward_folds()],
        },
        "frozen schedule",
    )
    if set(summary.get("folds", {})) != {f"test_{year}" for year in _YEARS}:
        raise Phase10ArtifactError("run requires every frozen outer fold")
    return summary, preflight


def _predictions(frame: pd.DataFrame, block: pd.DataFrame, context: str) -> None:
    validate_development_frame(frame)
    if frame.empty or frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise Phase10ArtifactError(f"{context}: missing/duplicate/empty sample universe")
    if frame["horizon_minutes"].ne(15).any() or frame["instrument"].ne("XAU_USD").any():
        raise Phase10ArtifactError(f"{context}: changed gold instrument/horizon")
    audit = [
        name
        for name in frame.columns
        if name in block.columns and name not in _PREDICTION_VALUES and name != "fold"
    ]
    _same_frame(frame.loc[:, audit], block.loc[:, audit], f"{context}/common universe")
    probabilities = validate_probability_matrix(frame[PROBABILITY_COLUMNS].to_numpy())
    if not np.isfinite(frame["expected_return_bps"].to_numpy()).all():
        raise Phase10ArtifactError(f"{context}: invalid expected returns")
    predicted = np.array(CLASS_ORDER)[probabilities.argmax(axis=1)]
    if not np.array_equal(frame["predicted_class"].to_numpy(), predicted):
        raise Phase10ArtifactError(f"{context}: prediction class differs from probabilities")


def _fallback(silver: pd.DataFrame, reference: pd.DataFrame, block: pd.DataFrame) -> None:
    for name in ("silver_usable", "used_price_only_fallback"):
        if name not in silver or not is_bool_dtype(silver[name]) or silver[name].isna().any():
            raise Phase10ArtifactError(f"fallback needs boolean {name}")
    usable = block["silver_usable"].to_numpy(dtype=bool)
    recorded_usable = silver["silver_usable"].to_numpy(dtype=bool)
    fallback = silver["used_price_only_fallback"].to_numpy(dtype=bool)
    if not np.array_equal(recorded_usable, usable):
        raise Phase10ArtifactError("silver usable routing differs from feature history")
    if ((~usable) & (~fallback)).any():
        raise Phase10ArtifactError("unusable silver context bypassed the frozen fallback")
    _same_frame(
        silver.loc[fallback, _PREDICTION_VALUES],
        reference.loc[fallback, _PREDICTION_VALUES],
        "exact frozen price-only fallback",
    )


def _evaluation(root: Path, records: pd.DataFrame) -> dict[str, Any]:
    saved = _json(root / "evaluation.json")
    policy = DecisionPolicy(**saved["policy"])
    base = run_backtest_v1(records, costs=BASE_COSTS, policy=policy)
    stress = run_backtest_v1(
        records,
        costs=STRESS_COSTS,
        policy=policy,
        decision_costs=BASE_COSTS,
    )
    calculated = {
        "classification": _metrics(records),
        "base_backtest": base.metrics,
        "stress_backtest": stress.metrics,
        "policy": asdict(policy),
        "sample_digest": sample_id_digest(records["sample_id"]),
    }
    _equal(saved, calculated, f"{root.name}/evaluation replay")
    for filename, frame in (
        ("base_decisions.parquet", base.decisions),
        ("base_trades.parquet", base.trades),
        ("stress_trades.parquet", stress.trades),
    ):
        _same_frame(pd.read_parquet(root / filename), frame, f"{root.name}/{filename}")
    return calculated


def _features(root: Path) -> pd.DataFrame:
    table = pd.read_parquet(root / "features.parquet")
    validate_development_frame(table)
    if table["sample_id"].duplicated().any() or table["sample_id"].isna().any():
        raise Phase10ArtifactError("features contain duplicate/missing gold sample IDs")
    rebuilt = build_silver_features(table)
    _same_frame(
        table.loc[:, list(SILVER_FEATURE_NAMES)],
        rebuilt.loc[:, list(SILVER_FEATURE_NAMES)],
        "causal silver features",
    )
    usable = (
        ~table["silver_is_missing"]
        & ~table["silver_is_stale"]
        & np.isfinite(table.loc[:, list(_MODELED_FEATURES)].to_numpy()).all(axis=1)
    )
    if (
        "silver_usable" not in table
        or not is_bool_dtype(table["silver_usable"])
        or table["silver_usable"].isna().any()
        or not np.array_equal(table["silver_usable"].to_numpy(), usable.to_numpy())
    ):
        raise Phase10ArtifactError("silver usability must require complete real feature history")
    return table


def _fold(root: Path, table: pd.DataFrame, fold: WalkForwardFold) -> dict[str, Any]:
    directory = root / "folds" / fold.name
    train = select_block(table, fold.train, gap_minutes=181)
    test = select_block(table, fold.test, purge=False)
    calibration = select_block(table, fold.calibration, purge=False)
    split = {
        "fold": fold.name,
        "train_rows": len(train),
        "test_rows": len(test),
        "calibration_rows_reserved": len(calibration),
        "gap_minutes": 181,
        "train_digest": sample_id_digest(train["sample_id"]),
        "test_digest": sample_id_digest(test["sample_id"]),
        "calibration_digest": sample_id_digest(calibration["sample_id"]),
    }
    _equal(_json(directory / "split_audit.json"), split, "split/fold alignment")
    _equal(_json(directory / "reference_split.json"), split, "frozen gold sample universe")
    records: dict[str, pd.DataFrame] = {}
    models: dict[str, Any] = {}
    for variant in ("reference", "silver"):
        records[variant] = pd.read_parquet(directory / variant / "outer_predictions.parquet")
        _predictions(records[variant], test, f"{fold.name}/{variant}")
        models[variant] = _evaluation(directory / variant, records[variant])
    _equal(
        models["reference"]["policy"],
        _json(directory / "reference_evaluation.json")["policy"],
        "frozen baseline policy",
    )
    _fallback(records["silver"], records["reference"], test)
    inner_records = []
    for index, inner in enumerate(fold.inner_folds):
        block = select_block(table, inner.validation, purge=False)
        reference = pd.read_parquet(directory / f"reference_inner_{index}.parquet")
        silver = pd.read_parquet(directory / f"selected_inner_{index}.parquet")
        _predictions(reference, block, f"{fold.name}/{inner.name}/reference")
        _predictions(silver, block, f"{fold.name}/{inner.name}/silver")
        _fallback(silver, reference, block)
        inner_records.append(silver)
    selection = _json(directory / "selection.json")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise Phase10ArtifactError("silver selection requires exactly three frozen candidates")
    expected_specs = [
        {"family": "logistic", "value": value}
        for value in (0.1, 1.0, 10.0)
    ]
    if [candidate.get("spec") for candidate in candidates] != expected_specs:
        raise Phase10ArtifactError("silver logistic C-grid differs from frozen Phase-7")
    if any(candidate.get("class_weight") != "balanced" for candidate in candidates):
        raise Phase10ArtifactError("silver class weighting differs from frozen Phase-7")
    for candidate in candidates:
        scores = candidate.get("folds")
        if not isinstance(scores, list) or len(scores) != len(fold.inner_folds):
            raise Phase10ArtifactError("silver candidate inner-fold evidence is incomplete")
        mean_f1 = float(np.mean([score["metrics"]["macro_f1"] for score in scores]))
        mean_loss = float(np.mean([score["metrics"]["log_loss"] for score in scores]))
        _equal(candidate.get("mean_macro_f1"), mean_f1, "candidate mean macro-F1")
        _equal(candidate.get("mean_log_loss"), mean_loss, "candidate mean log loss")
    selected = min(
        candidates,
        key=lambda candidate: (
            -candidate["mean_macro_f1"],
            candidate["mean_log_loss"],
        ),
    )
    _equal(selection.get("selected_name"), selected.get("name"), "selected candidate name")
    _equal(selection.get("selected_spec"), selected.get("spec"), "selected candidate spec")
    _equal(selection.get("class_weight"), "balanced", "selected class weighting")
    _equal(
        selection.get("calibration_status"),
        "reserved_not_fitted",
        "calibration isolation",
    )
    selected_scores = selected["folds"]
    for index, (record, score) in enumerate(
        zip(inner_records, selected_scores, strict=True)
    ):
        _equal(
            score.get("metrics"),
            _metrics(record),
            f"selected inner metrics/{index}",
        )
        _equal(
            score.get("validation_rows"),
            len(record),
            f"selected inner rows/{index}",
        )
        _equal(
            score.get("validation_sample_digest"),
            sample_id_digest(record["sample_id"]),
            f"selected inner digest/{index}",
        )
        _equal(
            score.get("context"),
            context_summary(record),
            f"selected inner context/{index}",
        )

    training = _json(directory / "training_audit.json")
    _equal(training.get("selected_spec"), selected.get("spec"), "final selected model spec")
    if training.get("class_weight") != "balanced":
        raise Phase10ArtifactError("final silver model class weighting differs")
    checkpoint = directory / "final_model.joblib"
    fitted = training.get("model_fitted")
    if not isinstance(fitted, bool) or checkpoint.is_file() != fitted:
        raise Phase10ArtifactError("final model checkpoint presence differs from fit audit")
    if training.get("checkpoint_prediction_parity") is not True:
        raise Phase10ArtifactError("final model checkpoint prediction parity is unproven")

    policy, attempts = select_policy(inner_records, minimum_trades=20)
    _equal(selection["selected_policy"], asdict(policy), "inner-only policy selection")
    _equal(selection["policy_candidates"], attempts, "policy candidates replay")
    _equal(models["silver"]["policy"], asdict(policy), "frozen selected outer policy")
    return {"split": split, "models": models, "context": context_summary(records["silver"])}


def _validate(
    root: Path,
    inventory: dict[str, dict[str, Any]],
    *,
    expected_run_status: str = "succeeded",
) -> dict[str, Any]:
    _expected_files(root, set(inventory))
    _authenticated_reference(root, inventory)
    summary, _ = _identity(root, expected_run_status=expected_run_status)
    table = _features(root)
    folds = {fold.name: _fold(root, table, fold) for fold in make_walk_forward_folds()}
    _equal(summary["folds"], folds, "fold summary replay")
    _equal(summary["comparison"], summarize_phase10(list(folds.values())), "run gates replay")
    return summary


def complete_phase10_run(directory: str | Path) -> dict[str, Any]:
    """Seal only a semantically valid, finished exploratory run."""
    root = Path(directory).absolute()
    if (root / "completion.json").exists():
        raise Phase10ArtifactError("completion already exists; runs are immutable")
    inventory = _inventory(root)
    _validate(root, inventory, expected_run_status="running")
    files = list(inventory.values())
    payload = {"files": files, "version": content_version({"files": files})}
    write_json_atomic(root / "completion.json", payload)
    return payload


def verify_phase10_run(directory: str | Path) -> dict[str, Any]:
    """Verify inventory, reference parity, fallback, metrics and economic gates."""
    root = Path(directory).absolute()
    completion_path = root / "completion.json"
    if completion_path.is_symlink():
        raise Phase10ArtifactError("symlink completion is forbidden")
    completion = _json(completion_path)
    expected = _parse_inventory(completion)
    actual = _inventory(root)
    if expected != actual:
        raise Phase10ArtifactError("artifact inventory or digest mismatch")
    summary = _validate(root, actual)
    return {
        "protocol": PROTOCOL,
        "run_id": root.name,
        "verified_files": len(actual),
        "completion_version": completion["version"],
        "holdout_opened": False,
        "champion_promotion": False,
        "trading_activation": False,
        "decision": summary["comparison"]["decision"],
    }
