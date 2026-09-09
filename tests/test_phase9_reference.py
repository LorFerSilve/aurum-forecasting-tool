"""Frozen Phase-7/8 reference loader guards for Phase 9."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gold_forecasting.artifacts import (
    content_version,
    file_digest,
    write_json_atomic,
    write_parquet_atomic,
)
from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.phase9.config import Phase9Config
from gold_forecasting.phase9.reference import (
    Phase9ReferenceError,
    load_phase9_reference_bundle,
)


def _records(*, horizon: int = 15) -> pd.DataFrame:
    times = pd.date_range(
        "2022-02-01T12:00:00Z",
        periods=2,
        freq="15min",
    )
    return pd.DataFrame(
        {
            "sample_id": ["sample-a", "sample-b"],
            "fold": ["test_2022", "test_2022"],
            "horizon_minutes": [horizon, horizon],
            "prediction_time_utc": times,
        }
    )


def _write_completed_run(
    root: Path,
    *,
    run_id: str,
    code_version: str,
    protocol: str,
    summary: dict[str, object],
    files: dict[str, object],
) -> str:
    run_root = root / run_id
    run_root.mkdir(parents=True)
    write_json_atomic(
        run_root / "run.json",
        {
            "run_id": run_id,
            "status": "succeeded",
            "code_version": code_version,
        },
    )
    write_json_atomic(
        run_root / "summary.json",
        {
            "protocol": protocol,
            "holdout_opened": False,
            **summary,
        },
    )
    for relative, payload in files.items():
        destination = run_root / relative
        if isinstance(payload, pd.DataFrame):
            write_parquet_atomic(destination, payload)
        else:
            assert isinstance(payload, dict)
            write_json_atomic(destination, payload)
    entries = [
        file_digest(
            item,
            relative_to=run_root,
        ).model_dump(mode="json")
        for item in sorted(run_root.rglob("*"))
        if item.is_file()
        and item.name not in {"run.json", "completion.json"}
    ]
    version = content_version({"files": entries})
    write_json_atomic(
        run_root / "completion.json",
        {
            "files": entries,
            "version": version,
        },
    )
    return version


def _fixture(
    tmp_path: Path,
    *,
    phase8_horizon: int = 15,
) -> tuple[Phase9Config, Path]:
    reports = tmp_path / "reports"
    phase7_records = _records()
    phase8_records = _records(
        horizon=phase8_horizon,
    )
    digest = sample_id_digest(
        phase7_records["sample_id"]
    )
    phase7_version = _write_completed_run(
        reports / "phase7_runs",
        run_id="p7",
        code_version="p7-code",
        protocol="phase7-v2",
        summary={},
        files={
            "horizon_15/mvp/test_2022/fold_summary.json": {
                "split": {
                    "test_digest": digest,
                },
                "models": {
                    "logistic": {
                        "sample_digest": digest,
                    }
                },
            },
            "horizon_15/mvp/test_2022/logistic/selection.json": {
                "selected_policy": {
                    "confidence_threshold": 0.6,
                    "min_expected_net_bps": 2.0,
                }
            },
            "horizon_15/mvp/test_2022/logistic/outer_predictions.parquet": (
                phase7_records
            ),
        },
    )
    phase8_version = _write_completed_run(
        reports / "phase8_runs",
        run_id="p8",
        code_version="p8-code",
        protocol="phase8-v1",
        summary={
            "phase7_reference_run": "p7",
            "test_years": [2022, 2023, 2024],
        },
        files={
            "source_manifests.json": {
                "1min": {
                    "dataset_version": "sha256:" + "a" * 64,
                },
                "3min": {
                    "dataset_version": "sha256:" + "b" * 64,
                },
                "15min": {
                    "dataset_version": "sha256:" + "c" * 64,
                },
            },
            "horizon_15/test_2022/training_audit.json": {
                "test_digest": digest,
                "selected_policy": {
                    "confidence_threshold": 1.0,
                    "min_expected_net_bps": 1e12,
                },
            },
            "horizon_15/test_2022/ensemble/outer_predictions.parquet": (
                phase8_records
            ),
        },
    )
    champion = tmp_path / "phase7_champion.yaml"
    champion.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "release: v0.2.0",
                "protocol: phase7-v2",
                "run_id: p7",
                "code_version: p7-code",
                "scope: research_fallback_only",
                "holdout_opened: false",
                "trading_champion: null",
                "research_champions:",
                "  15:",
                "    variant: mvp",
                "    family: logistic",
                "economic_promotion_candidates: 0",
                "note: fixture",
                "",
            ]
        ),
        encoding="utf-8",
    )
    base = Phase9Config().model_dump()
    config = Phase9Config.model_construct(
        **{
            **base,
            "phase7_reference_run": "p7",
            "phase7_reference_code": "p7-code",
            "phase7_reference_completion": phase7_version,
            "phase8_reference_run": "p8",
            "phase8_reference_code": "p8-code",
            "phase8_reference_completion": phase8_version,
        }
    )
    return config, champion


def test_reference_bundle_loads_frozen_fold_and_policy(
    tmp_path: Path,
) -> None:
    config, champion = _fixture(tmp_path)

    bundle = load_phase9_reference_bundle(
        tmp_path,
        config,
        champion,
    )
    phase7 = bundle.phase7_fold("test_2022")
    phase8 = bundle.phase8_fold("test_2022")

    assert phase7.records["sample_id"].tolist() == [
        "sample-a",
        "sample-b",
    ]
    assert phase7.policy.confidence_threshold == 0.6
    assert phase8.policy.confidence_threshold == 1.0
    assert phase7.original_sample_digest == phase8.original_sample_digest


def test_reference_loader_detects_modified_prediction_artifact(
    tmp_path: Path,
) -> None:
    config, champion = _fixture(tmp_path)
    bundle = load_phase9_reference_bundle(
        tmp_path,
        config,
        champion,
    )
    target = (
        tmp_path
        / "reports"
        / "phase8_runs"
        / "p8"
        / "horizon_15"
        / "test_2022"
        / "ensemble"
        / "outer_predictions.parquet"
    )
    target.write_bytes(
        target.read_bytes() + b"tamper"
    )

    with pytest.raises(
        Phase9ReferenceError,
        match="digest mismatch",
    ):
        bundle.phase8_fold("test_2022")


def test_reference_loader_rejects_wrong_horizon_even_when_manifest_matches(
    tmp_path: Path,
) -> None:
    config, champion = _fixture(
        tmp_path,
        phase8_horizon=30,
    )
    bundle = load_phase9_reference_bundle(
        tmp_path,
        config,
        champion,
    )

    with pytest.raises(
        Phase9ReferenceError,
        match="wrong horizon",
    ):
        bundle.phase8_fold("test_2022")


def test_reference_loader_rejects_wrong_completion_pin(
    tmp_path: Path,
) -> None:
    config, champion = _fixture(tmp_path)
    changed = config.model_copy(
        update={
            "phase8_reference_completion": (
                "sha256:" + "0" * 64
            )
        }
    )

    with pytest.raises(
        Phase9ReferenceError,
        match="completion version",
    ):
        load_phase9_reference_bundle(
            tmp_path,
            changed,
            champion,
        )
