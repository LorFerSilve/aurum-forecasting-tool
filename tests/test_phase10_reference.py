"""Cryptographic identity, complete-universe and fold parity mutation guards."""

from __future__ import annotations

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
from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.evaluation.walk_forward import (
    WalkForwardError,
    make_walk_forward_folds,
    select_block,
)
from gold_forecasting.phase10 import reference as module
from gold_forecasting.phase10.reference import (
    Phase10ReferenceError,
    assert_exact_gold_parity,
    load_phase10_reference,
    verify_phase10_universe,
)


def _table() -> pd.DataFrame:
    times = pd.date_range("2020-01-02T00:00:00Z", "2024-12-02T00:00:00Z", freq="30D")
    count = len(times)
    return pd.DataFrame(
        {
            "sample_id": [f"gold-{index}" for index in range(count)],
            "instrument": "XAU_USD",
            "source": "histdata",
            "prediction_time_utc": times,
            "entry_time_utc": times + pd.Timedelta(minutes=1),
            "label_end_time_utc": times + pd.Timedelta(minutes=16),
            "horizon_minutes": 15,
            "target_class_id": np.ones(count, dtype=np.int8),
            "target_class": "neutral",
            "arithmetic_return_bps": 0.0,
            "entry_bid_open": 2000.0,
            "exit_bid_open": 2000.0,
            "source_raw_file_hash": "a" * 64,
            "feature": np.linspace(0.0, 1.0, count),
        }
    )


def _predictions(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns="feature").assign(
        p_down=0.2, p_neutral=0.6, p_up=0.2, predicted_class="neutral", expected_return_bps=0.0
    )


def _repin(root: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    entries = [
        file_digest(path, relative_to=root).model_dump(mode="json")
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"completion.json", "run.json"}
    ]
    version = content_version({"files": entries})
    write_json_atomic(root / "completion.json", {"version": version, "files": entries})
    monkeypatch.setattr(module, "PHASE7_REFERENCE_COMPLETION", version)
    return version


def _reference_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a manifest-authenticated, development-only fixture, never real pins."""
    monkeypatch.setattr(module, "PHASE7_REFERENCE_RUN", "fixture")
    monkeypatch.setattr(module, "PHASE7_REFERENCE_CODE", "fixture-code")
    root = tmp_path / "reports/phase7_runs/fixture"
    write_json_atomic(
        root / "run.json",
        {
            "run_id": "fixture",
            "status": "succeeded",
            "code_version": "fixture-code",
        },
    )
    write_json_atomic(
        root / "summary.json",
        {
            "protocol": "phase7-v2",
            "holdout_opened": False,
            "test_years": [2022, 2023, 2024],
        },
    )
    write_json_atomic(
        root / "resolved_config.json",
        {
            "protocol_version": "phase7-v2",
            "test_years": [2022, 2023, 2024],
            "minimum_policy_trades": 20,
            "seed": 20260906,
            "horizons": [15],
        },
    )
    write_json_atomic(root / "ablation_manifest.json", {"variants": {"mvp": ["feature"]}})
    table = _table()
    write_parquet_atomic(root / "horizon_15/model_table.parquet", table)
    for fold in make_walk_forward_folds():
        base = root / f"horizon_15/mvp/{fold.name}"
        train = select_block(table, fold.train, gap_minutes=181)
        test = select_block(table, fold.test, purge=False)
        calibration = select_block(table, fold.calibration, purge=False)
        split = {
            "fold": fold.name,
            "gap_minutes": 181,
            "train_rows": len(train),
            "train_digest": sample_id_digest(train["sample_id"]),
            "test_rows": len(test),
            "test_digest": sample_id_digest(test["sample_id"]),
            "calibration_rows_reserved": len(calibration),
            "calibration_digest": sample_id_digest(calibration["sample_id"]),
        }
        write_json_atomic(base / "split_audit.json", split)
        write_json_atomic(
            base / "fold_summary.json",
            {
                "split": split,
                "models": {"logistic": {"sample_digest": split["test_digest"]}},
            },
        )
        scores = []
        for index, inner in enumerate(fold.inner_folds):
            inner_train = select_block(table, inner.train, gap_minutes=181)
            validation = select_block(table, inner.validation, purge=False)
            scores.append(
                {
                    "inner_fold": inner.name,
                    "train_rows": len(inner_train),
                    "train_sample_digest": sample_id_digest(inner_train["sample_id"]),
                    "validation_rows": len(validation),
                    "validation_sample_digest": sample_id_digest(validation["sample_id"]),
                    "metrics": {"macro_f1": 0.4, "log_loss": 1.0},
                }
            )
            write_parquet_atomic(
                base / f"logistic/selected_inner_{index}.parquet", _predictions(validation)
            )
        write_json_atomic(
            base / "logistic/selection.json",
            {
                "calibration_status": "reserved_not_fitted",
                "selected_spec": {"family": "logistic", "value": 0.1},
                "selected_rounds": None,
                "selected_policy": {"confidence_threshold": 1.0, "min_expected_net_bps": 1e12},
                "candidates": [
                    {
                        "spec": {"family": "logistic", "value": value},
                        "folds": scores,
                        "mean_macro_f1": 0.4,
                        "mean_log_loss": 1.0,
                    }
                    for value in (0.1, 1.0, 10.0)
                ],
            },
        )
        write_parquet_atomic(
            base / "logistic/outer_predictions.parquet", _predictions(test).assign(fold=fold.name)
        )
    config = tmp_path / "configs/phase7_champion.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "release: v0.2.0\nprotocol: phase7-v2\nrun_id: fixture\ncode_version: fixture-code\n"
        "scope: research_fallback_only\nholdout_opened: false\ntrading_champion: null\n"
        "research_champions:\n  15:\n    variant: mvp\n    family: logistic\n"
        "economic_promotion_candidates: 0\nnote: fixture\n",
        encoding="utf-8",
    )
    _repin(root, monkeypatch)
    return root


def test_authenticated_reference_loads_all_outer_inner_blocks_without_phase8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reference_fixture(tmp_path, monkeypatch)
    reference = load_phase10_reference(tmp_path)
    report = verify_phase10_universe(reference, _table(), ("feature",))
    assert report["common_universe"]["rows"] == len(_table())
    assert report["exact_gold_values"] is True
    assert report["holdout_opened"] is False
    assert set(reference.folds) == {"test_2022", "test_2023", "test_2024"}
    assert reference.fold("test_2022").policy.confidence_threshold == 1.0
    for fold in reference.folds.values():
        assert len(fold.inner_records) == 2
        assert fold.parity["gap_minutes"] == 181
        assert set(fold.parity) == {
            "fold",
            "gap_minutes",
            "train",
            "calibration",
            "test",
            "inner_folds",
        }
    with pytest.raises(Phase10ReferenceError, match=r"unexpected.*fold"):
        reference.fold("test_2025")


@pytest.mark.parametrize(
    "relative",
    [
        "resolved_config.json",
        "ablation_manifest.json",
        "horizon_15/model_table.parquet",
        "horizon_15/mvp/test_2022/split_audit.json",
        "horizon_15/mvp/test_2022/logistic/selection.json",
        "horizon_15/mvp/test_2022/logistic/selected_inner_0.parquet",
        "horizon_15/mvp/test_2022/logistic/outer_predictions.parquet",
    ],
)
def test_required_artifact_tampering_fails_before_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    root = _reference_fixture(tmp_path, monkeypatch)
    path = root / relative
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(Phase10ReferenceError, match="digest mismatch"):
        load_phase10_reference(tmp_path)


def test_completion_is_pinned_not_only_self_consistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _reference_fixture(tmp_path, monkeypatch)
    pin = module.PHASE7_REFERENCE_COMPLETION
    write_json_atomic(root / "unrelated.json", {"mutation": True})
    _repin(root, monkeypatch)
    monkeypatch.setattr(module, "PHASE7_REFERENCE_COMPLETION", pin)
    with pytest.raises(Phase10ReferenceError, match="completion version differs"):
        load_phase10_reference(tmp_path)


@pytest.mark.parametrize(
    "column,value",
    [
        ("target_class_id", 2),
        ("arithmetic_return_bps", 99.0),
        ("feature", 8.0),
        ("source_raw_file_hash", "b" * 64),
        ("entry_bid_open", 12.0),
    ],
)
def test_universe_rejects_changed_targets_features_and_source_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    value: Any,
) -> None:
    _reference_fixture(tmp_path, monkeypatch)
    reference = load_phase10_reference(tmp_path)
    changed = _table()
    changed.loc[0, column] = value
    with pytest.raises(Phase10ReferenceError, match="value parity"):
        verify_phase10_universe(reference, changed)


@pytest.mark.parametrize("mutation", ["subset", "extra", "reorder", "duplicate", "missing_id"])
def test_universe_never_drops_or_repairs_gold_samples(mutation: str) -> None:
    original = _table()
    changed = original.copy()
    if mutation == "subset":
        changed = changed.iloc[1:]
    elif mutation == "extra":
        changed = pd.concat([changed, changed.iloc[-1:].assign(sample_id="extra")])
    elif mutation == "reorder":
        changed = changed.iloc[::-1]
    elif mutation == "duplicate":
        changed.loc[1, "sample_id"] = changed.loc[0, "sample_id"]
    else:
        changed.loc[0, "sample_id"] = None
    with pytest.raises(Phase10ReferenceError):
        assert_exact_gold_parity(original, changed)


def test_universe_rejects_holdout_before_subset_selection() -> None:
    original = _table()
    changed = original.copy()
    changed.loc[0, "prediction_time_utc"] = pd.Timestamp("2025-01-01T00:00:00Z")
    with pytest.raises(WalkForwardError, match="holdout"):
        assert_exact_gold_parity(original, changed)


@pytest.mark.parametrize(
    "mutation",
    [
        "gap",
        "train_digest",
        "inner_train",
        "inner_validation",
        "inner_order",
        "outer_fold",
        "inner_probability",
        "holdout",
        "wrong_horizon",
    ],
)
def test_semantic_guard_rejects_consistently_hashed_wrong_fold_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    import json

    root = _reference_fixture(tmp_path, monkeypatch)
    base = root / "horizon_15/mvp/test_2022"
    if mutation in {"gap", "train_digest"}:
        path = base / "split_audit.json"
        payload = json.loads(path.read_text())
        payload["gap_minutes" if mutation == "gap" else "train_digest"] = (
            16 if mutation == "gap" else "sha256:" + "0" * 64
        )
        write_json_atomic(path, payload)
    elif mutation in {"inner_train", "inner_validation", "inner_order"}:
        path = base / "logistic/selection.json"
        payload = json.loads(path.read_text())
        score = payload["candidates"][0]["folds"][0]
        if mutation == "inner_order":
            score["inner_fold"] = "2022_inner_q3"
        else:
            score[f"{mutation.removeprefix('inner_')}_sample_digest"] = "sha256:" + "0" * 64
        write_json_atomic(path, payload)
    else:
        path = base / (
            "logistic/selected_inner_0.parquet"
            if mutation == "inner_probability"
            else "logistic/outer_predictions.parquet"
        )
        frame = pd.read_parquet(path)
        if mutation == "outer_fold":
            frame["fold"] = "test_2023"
        elif mutation == "inner_probability":
            frame["p_up"] = float("nan")
        elif mutation == "holdout":
            frame.loc[0, "prediction_time_utc"] = pd.Timestamp("2025-01-01T00:00:00Z")
        else:
            frame["horizon_minutes"] = 30
        write_parquet_atomic(path, frame)
    _repin(root, monkeypatch)
    with pytest.raises((Phase10ReferenceError, WalkForwardError)):
        load_phase10_reference(tmp_path)


def test_mvp_feature_order_is_frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _reference_fixture(tmp_path, monkeypatch)
    reference = load_phase10_reference(tmp_path)
    with pytest.raises(Phase10ReferenceError, match="feature order"):
        verify_phase10_universe(reference, _table(), ("new_feature",))


def test_reference_champion_identity_cannot_be_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reference_fixture(tmp_path, monkeypatch)
    path = tmp_path / "configs/phase7_champion.yaml"
    path.write_text(path.read_text().replace("family: logistic", "family: ridge"))
    with pytest.raises(Phase10ReferenceError, match="mvp/logistic pin"):
        load_phase10_reference(tmp_path)
