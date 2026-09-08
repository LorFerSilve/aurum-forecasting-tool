"""Integrity tests for phase-9 artifacts and validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gold_forecasting.phase9.artifacts import (
    Phase9ArtifactError,
    finalize_phase9_manifest,
    verify_phase9,
    write_phase9_run_contract,
)
from gold_forecasting.phase9.config import Phase9Config
from gold_forecasting.phase9.orchestration import build_phase9_schedule


def _minimal_run(
    root: Path,
    *,
    holdout_opened: bool = False,
) -> None:
    config = Phase9Config()
    root.mkdir(parents=True)
    write_phase9_run_contract(
        root,
        config=config,
        folds=build_phase9_schedule(config),
        phase8_reference_run="phase8-canonical-test",
        code_version="abc123",
        runtime={
            "python": "3.11",
            "torch": "2.11.0+cpu",
        },
        requirements_lock="numpy==2.4.6\n",
        neural_lock="torch==2.11.0\n",
    )
    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": root.name,
                "status": "succeeded",
            }
        ),
        encoding="utf-8",
    )
    finalize_phase9_manifest(
        root,
        {
            "protocol": "phase9-v1",
            "holdout_opened": holdout_opened,
            "test_years": [2022, 2023, 2024],
            "phase8_reference_run": (
                "phase8-canonical-test"
            ),
            "folds": {},
        },
    )


def test_verify_phase9_accepts_hashed_closed_holdout_run(
    tmp_path: Path,
) -> None:
    root = tmp_path / "phase9-run"
    _minimal_run(root)

    result = verify_phase9(root)

    assert result["run_id"] == "phase9-run"
    assert result["protocol"] == "phase9-v1"
    assert (
        result["phase8_reference_run"]
        == "phase8-canonical-test"
    )
    assert result["verified_files"] >= 6


def test_finalize_phase9_rejects_open_holdout(
    tmp_path: Path,
) -> None:
    root = tmp_path / "phase9-run"
    root.mkdir()

    with pytest.raises(
        Phase9ArtifactError,
        match="holdout",
    ):
        finalize_phase9_manifest(
            root,
            {
                "protocol": "phase9-v1",
                "holdout_opened": True,
                "test_years": [
                    2022,
                    2023,
                    2024,
                ],
            },
        )


def test_verify_phase9_detects_modified_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "phase9-run"
    _minimal_run(root)
    (root / "runtime.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="digest mismatch",
    ):
        verify_phase9(root)
