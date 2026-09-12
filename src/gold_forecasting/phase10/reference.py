"""Pinned Phase-7 15m reference and exact-universe parity for silver ablation.

Only authenticated data, JSON and Parquet are opened. No model is unpickled,
refitted, or selected. The Phase-9 manifest reader is reused without requiring
Phase-8 artifacts or adopting Phase-9's subset-universe convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from gold_forecasting.backtesting.v1 import DecisionPolicy
from gold_forecasting.benchmark.pipeline import PROBABILITY_COLUMNS
from gold_forecasting.classification import CLASS_ORDER
from gold_forecasting.config import _load_yaml_mapping
from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.evaluation.walk_forward import (
    WalkForwardFold,
    make_walk_forward_folds,
    select_block,
    validate_development_frame,
)
from gold_forecasting.phase8.reference import Phase7ChampionConfig
from gold_forecasting.phase9.reference import (
    Phase9ReferenceError,
    VerifiedReferenceRun,
    _load_verified_run,
    _policy_from_payload,
)

PHASE7_REFERENCE_RUN = "20260907T014255645674Z-b2afe281"
PHASE7_REFERENCE_CODE = "3f0a703568224fe9169b1e9f8d61dad131f0005b"
PHASE7_REFERENCE_COMPLETION = (
    "sha256:beac58092d06350bb067fbd2df144bb9cb2507f45053c19487474c87d0ceca0f"
)
PHASE10_TEST_YEARS = (2022, 2023, 2024)
PHASE10_GAP_MINUTES = 181
_PREDICTION_COLUMNS = (*PROBABILITY_COLUMNS, "predicted_class", "expected_return_bps")


class Phase10ReferenceError(ValueError):
    """The frozen identity, gold universe, or saved predictions are not intact."""


@dataclass(frozen=True, slots=True)
class Phase10ReferenceFold:
    """Original outer and selected inner predictions, never recomputed."""

    records: pd.DataFrame
    inner_records: tuple[pd.DataFrame, ...]
    policy: DecisionPolicy
    split: dict[str, Any]
    selection: dict[str, Any]
    parity: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Phase10Reference:
    run: VerifiedReferenceRun
    table: pd.DataFrame
    feature_names: tuple[str, ...]
    folds: dict[str, Phase10ReferenceFold]

    def fold(self, name: str) -> Phase10ReferenceFold:
        try:
            return self.folds[name]
        except KeyError as exc:
            raise Phase10ReferenceError(f"unexpected frozen Phase-7 fold: {name}") from exc


def _validate_gold_rows(frame: pd.DataFrame, *, context: str) -> None:
    validate_development_frame(frame)
    required = {"sample_id", "instrument", "horizon_minutes", "target_class_id"}
    if not required.issubset(frame.columns):
        raise Phase10ReferenceError(f"{context} misses required gold identity columns")
    if (
        frame.empty
        or frame["sample_id"].isna().any()
        or frame["sample_id"].duplicated().any()
        or not frame["sample_id"].map(lambda value: isinstance(value, str) and bool(value)).all()
    ):
        raise Phase10ReferenceError(f"{context} requires unique nonempty string sample IDs")
    if not frame["instrument"].eq("XAU_USD").all():
        raise Phase10ReferenceError(f"{context} contains a non-gold instrument")
    if not frame["horizon_minutes"].eq(15).all():
        raise Phase10ReferenceError(f"{context} contains the wrong horizon")
    if not frame["prediction_time_utc"].is_monotonic_increasing:
        raise Phase10ReferenceError(f"{context} is not in frozen chronological sample order")


def assert_exact_gold_parity(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    columns: tuple[str, ...] | None = None,
    context: str = "gold universe",
) -> None:
    """Require identical ordered IDs and values; never drop or reorder samples."""
    _validate_gold_rows(reference, context=f"{context} reference")
    _validate_gold_rows(candidate, context=context)
    if not np.array_equal(reference["sample_id"].to_numpy(), candidate["sample_id"].to_numpy()):
        raise Phase10ReferenceError(
            f"{context} ordered sample ID parity failed; subsets are forbidden"
        )
    compared = (
        columns
        if columns is not None
        else tuple(str(column) for column in reference.columns if column not in _PREDICTION_COLUMNS)
    )
    missing = set(compared).difference(candidate.columns)
    if missing:
        raise Phase10ReferenceError(f"{context} misses frozen gold columns: {sorted(missing)}")
    for column in compared:
        left = reference[column].reset_index(drop=True)
        right = candidate[column].reset_index(drop=True)
        # Dtype storage details may differ, but every numeric bit/value must agree.
        try:
            pd.testing.assert_series_equal(left, right, check_dtype=False, check_exact=True)
        except AssertionError as exc:
            raise Phase10ReferenceError(
                f"{context} frozen gold value parity failed: {column}"
            ) from exc


def _block_evidence(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        raise Phase10ReferenceError("frozen gold split has no rows")
    return {"rows": len(frame), "sample_digest": sample_id_digest(frame["sample_id"])}


def _fold_evidence(table: pd.DataFrame, fold: WalkForwardFold) -> dict[str, Any]:
    return {
        "fold": fold.name,
        "gap_minutes": PHASE10_GAP_MINUTES,
        "train": _block_evidence(select_block(table, fold.train, gap_minutes=PHASE10_GAP_MINUTES)),
        "calibration": _block_evidence(select_block(table, fold.calibration, purge=False)),
        "test": _block_evidence(select_block(table, fold.test, purge=False)),
        "inner_folds": {
            inner.name: {
                "train": _block_evidence(
                    select_block(table, inner.train, gap_minutes=PHASE10_GAP_MINUTES)
                ),
                "validation": _block_evidence(select_block(table, inner.validation, purge=False)),
            }
            for inner in fold.inner_folds
        },
    }


def _validate_predictions(records: pd.DataFrame, *, context: str) -> None:
    missing = set(_PREDICTION_COLUMNS).difference(records.columns)
    if missing:
        raise Phase10ReferenceError(f"{context} misses prediction columns: {sorted(missing)}")
    probabilities = records[PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        or not np.isfinite(records["expected_return_bps"].to_numpy(dtype=np.float64)).all()
    ):
        raise Phase10ReferenceError(f"{context} contains invalid frozen probabilities/returns")
    labels = np.asarray(CLASS_ORDER)[probabilities.argmax(axis=1)]
    if not np.array_equal(labels, records["predicted_class"].to_numpy()):
        raise Phase10ReferenceError(f"{context} predicted class differs from frozen probabilities")


def _load_fold(
    run: VerifiedReferenceRun, table: pd.DataFrame, fold: WalkForwardFold
) -> Phase10ReferenceFold:
    base = f"horizon_15/mvp/{fold.name}"
    split = run.read_json(f"{base}/split_audit.json")
    summary = run.read_json(f"{base}/fold_summary.json")
    evidence = _fold_evidence(table, fold)
    expected_split = {
        "fold": fold.name,
        "gap_minutes": PHASE10_GAP_MINUTES,
        "train_rows": evidence["train"]["rows"],
        "train_digest": evidence["train"]["sample_digest"],
        "test_rows": evidence["test"]["rows"],
        "test_digest": evidence["test"]["sample_digest"],
        "calibration_rows_reserved": evidence["calibration"]["rows"],
        "calibration_digest": evidence["calibration"]["sample_digest"],
    }
    if split != expected_split or summary.get("split") != expected_split:
        raise Phase10ReferenceError(f"frozen split audit parity failed: {fold.name}")
    if summary.get("models", {}).get("logistic", {}).get("sample_digest") != split["test_digest"]:
        raise Phase10ReferenceError(f"frozen model test digest parity failed: {fold.name}")
    family = f"{base}/logistic"
    selection = run.read_json(f"{family}/selection.json")
    if selection.get("calibration_status") != "reserved_not_fitted":
        raise Phase10ReferenceError(f"frozen calibration contract differs: {fold.name}")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or [item.get("spec") for item in candidates] != [
        {"family": "logistic", "value": value} for value in (0.1, 1.0, 10.0)
    ]:
        raise Phase10ReferenceError(f"frozen logistic candidate grid differs: {fold.name}")
    for candidate in candidates:
        scores = candidate.get("folds")
        if not isinstance(scores, list) or [row.get("inner_fold") for row in scores] != [
            inner.name for inner in fold.inner_folds
        ]:
            raise Phase10ReferenceError(f"frozen inner fold alignment differs: {fold.name}")
        for score in scores:
            inner_evidence = evidence["inner_folds"][score["inner_fold"]]
            for block, prefix in (("train", "train"), ("validation", "validation")):
                if (
                    score.get(f"{prefix}_rows") != inner_evidence[block]["rows"]
                    or score.get(f"{prefix}_sample_digest")
                    != inner_evidence[block]["sample_digest"]
                ):
                    raise Phase10ReferenceError(
                        f"frozen inner {block} digest parity failed: {fold.name}"
                    )
        for metric in ("macro_f1", "log_loss"):
            mean = float(np.mean([score["metrics"][metric] for score in scores]))
            if not np.isfinite(mean) or candidate.get(f"mean_{metric}") != mean:
                raise Phase10ReferenceError(f"frozen inner selection metric differs: {fold.name}")
    selected = min(candidates, key=lambda item: (-item["mean_macro_f1"], item["mean_log_loss"]))
    if (
        selection.get("selected_spec") != selected["spec"]
        or selection.get("selected_rounds") is not None
    ):
        raise Phase10ReferenceError(f"frozen inner model selection differs: {fold.name}")
    records = run.read_parquet(f"{family}/outer_predictions.parquet")
    expected_test = select_block(table, fold.test, purge=False).assign(fold=fold.name)
    assert_exact_gold_parity(records, expected_test, context=f"{fold.name} outer reference")
    if "fold" not in records or not records["fold"].eq(fold.name).all():
        raise Phase10ReferenceError(f"frozen outer fold identity differs: {fold.name}")
    _validate_predictions(records, context=f"{fold.name} outer")
    inner_records = []
    for index, inner in enumerate(fold.inner_folds):
        saved = run.read_parquet(f"{family}/selected_inner_{index}.parquet")
        expected = select_block(table, inner.validation, purge=False)
        assert_exact_gold_parity(saved, expected, context=f"{fold.name}/{inner.name} reference")
        _validate_predictions(saved, context=f"{fold.name}/{inner.name}")
        inner_records.append(saved)
    return Phase10ReferenceFold(
        records=records,
        inner_records=tuple(inner_records),
        policy=_policy_from_payload(
            selection.get("selected_policy"), context=f"{fold.name} policy"
        ),
        split=split,
        selection=selection,
        parity=evidence,
    )


def load_phase10_reference(
    project_root: str | Path,
    champion_config: str | Path | None = None,
) -> Phase10Reference:
    """Authenticate the fixed Phase-7 run, full MVP universe and every used fold."""
    root = Path(project_root).resolve(strict=True)
    champion_path = (
        Path(champion_config)
        if champion_config is not None
        else root / "configs/phase7_champion.yaml"
    )
    try:
        champions = Phase7ChampionConfig.model_validate(_load_yaml_mapping(champion_path))
        champion = champions.research_champions.get(15)
        if (
            champions.protocol != "phase7-v2"
            or champions.release != "v0.2.0"
            or champions.run_id != PHASE7_REFERENCE_RUN
            or champions.code_version != PHASE7_REFERENCE_CODE
            or champions.holdout_opened
            or champions.trading_champion is not None
            or champion is None
            or champion.variant != "mvp"
            or champion.family != "logistic"
        ):
            raise Phase10ReferenceError(
                "Phase-7 champion differs from the frozen 15m mvp/logistic pin"
            )
        run = _load_verified_run(
            root,
            phase_directory="phase7_runs",
            expected_run_id=PHASE7_REFERENCE_RUN,
            expected_protocol="phase7-v2",
            expected_code_version=PHASE7_REFERENCE_CODE,
            expected_completion_version=PHASE7_REFERENCE_COMPLETION,
        )
        config = run.read_json("resolved_config.json")
        if (
            config.get("protocol_version") != "phase7-v2"
            or config.get("test_years") != list(PHASE10_TEST_YEARS)
            or run.summary.get("test_years") != list(PHASE10_TEST_YEARS)
            or config.get("minimum_policy_trades") != 20
            or config.get("seed") != 20260906
            or 15 not in config.get("horizons", [])
        ):
            raise Phase10ReferenceError("frozen Phase-7 configuration/fold contract differs")
        catalog = run.read_json("ablation_manifest.json")
        names = catalog.get("variants", {}).get("mvp")
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) and name for name in names)
            or len(names) != len(set(names))
        ):
            raise Phase10ReferenceError("frozen MVP feature inventory is invalid")
        path = run.verified_path("horizon_15/model_table.parquet")
        # Exclude other Phase-7 ablations while keeping every row, source and target.
        columns = [name for name in pq.read_schema(path).names if not name.startswith("p7_")]
        table = pd.read_parquet(path, columns=columns)
        _validate_gold_rows(table, context="frozen full gold universe")
        if not set(names).issubset(table.columns):
            raise Phase10ReferenceError("frozen model table misses MVP features")
        folds = {
            fold.name: _load_fold(run, table, fold)
            for fold in make_walk_forward_folds(PHASE10_TEST_YEARS)
        }
    except Phase9ReferenceError as exc:
        raise Phase10ReferenceError(str(exc)) from exc
    return Phase10Reference(run, table, tuple(names), folds)


def verify_phase10_universe(
    reference: Phase10Reference,
    table: pd.DataFrame,
    feature_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Prove the entire gold universe and every train/inner/test block unchanged."""
    if feature_names is not None and feature_names != reference.feature_names:
        raise Phase10ReferenceError("Phase-10 price-only feature order differs from frozen MVP")
    assert_exact_gold_parity(reference.table, table)
    folds: dict[str, Any] = {}
    for fold in make_walk_forward_folds(PHASE10_TEST_YEARS):
        evidence = _fold_evidence(table, fold)
        if evidence != reference.fold(fold.name).parity:
            raise Phase10ReferenceError(
                f"Phase-10 train/inner/test split digest parity failed: {fold.name}"
            )
        folds[fold.name] = evidence
    return {
        "common_universe": _block_evidence(table),
        "exact_ordered_sample_ids": True,
        "exact_gold_values": True,
        "feature_names": list(reference.feature_names),
        "folds": folds,
        "holdout_opened": False,
    }


__all__ = [
    "PHASE7_REFERENCE_CODE",
    "PHASE7_REFERENCE_COMPLETION",
    "PHASE7_REFERENCE_RUN",
    "PHASE10_GAP_MINUTES",
    "PHASE10_TEST_YEARS",
    "Phase10Reference",
    "Phase10ReferenceError",
    "Phase10ReferenceFold",
    "assert_exact_gold_parity",
    "load_phase10_reference",
    "verify_phase10_universe",
]
