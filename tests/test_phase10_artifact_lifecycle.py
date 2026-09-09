"""Phase-10 completion sealing must precede the mutable registry terminal state."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gold_forecasting.evaluation.walk_forward import make_walk_forward_folds
from gold_forecasting.phase10.artifacts import (
    PROTOCOL,
    Phase10ArtifactError,
    _identity,
    _inventory,
)
from gold_forecasting.phase10.reference import (
    PHASE7_REFERENCE_COMPLETION,
    PHASE7_REFERENCE_RUN,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _identity_fixture(root: Path, *, status: str) -> None:
    code_version = "0123456789abcdef"
    data_version = "sha256:test-data"
    _write_json(
        root / "summary.json",
        {
            "protocol": PROTOCOL,
            "run_mode": "exploratory_modeled_latency",
            "holdout_opened": False,
            "test_years": [2022, 2023, 2024],
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
            "code_version": code_version,
            "data_version": data_version,
            "folds": {
                "test_2022": {},
                "test_2023": {},
                "test_2024": {},
            },
        },
    )
    _write_json(
        root / "run.json",
        {
            "run_id": root.name,
            "status": status,
            "code_version": code_version,
            "data_version": data_version,
        },
    )
    _write_json(
        root / "preflight.json",
        {
            "status": "passed",
            "holdout_opened": False,
            "formal_benchmark_ready": False,
            "code_version": code_version,
            "data_version": data_version,
        },
    )
    _write_json(
        root / "schedule.json",
        {
            "gap_minutes": 181,
            "folds": [fold.as_record() for fold in make_walk_forward_folds()],
        },
    )


def test_completion_inventory_excludes_registry_and_completion_manifests(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "run.json").write_text("registry", encoding="utf-8")
    (root / "completion.json").write_text("completion", encoding="utf-8")
    (root / "artifact.txt").write_text("sealed", encoding="utf-8")

    inventory = _inventory(root)

    assert set(inventory) == {"artifact.txt"}


def test_identity_distinguishes_preseal_running_from_final_succeeded(
    tmp_path: Path,
) -> None:
    root = tmp_path / "phase10-run"
    root.mkdir()
    _identity_fixture(root, status="running")

    _identity(root, expected_run_status="running")
    with pytest.raises(Phase10ArtifactError, match="succeeded status"):
        _identity(root)

    payload = json.loads((root / "run.json").read_text(encoding="utf-8"))
    payload["status"] = "succeeded"
    _write_json(root / "run.json", payload)

    _identity(root)
    with pytest.raises(Phase10ArtifactError, match="running status"):
        _identity(root, expected_run_status="running")


def test_identity_rejects_unknown_expected_status(tmp_path: Path) -> None:
    root = tmp_path / "phase10-run"
    root.mkdir()
    _identity_fixture(root, status="running")

    with pytest.raises(Phase10ArtifactError, match="unsupported expected run status"):
        _identity(root, expected_run_status="failed")
