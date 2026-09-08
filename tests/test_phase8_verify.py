"""End-to-end verification guard for completed phase-8 run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gold_forecasting.artifacts import (
    content_version,
    file_digest,
)
from gold_forecasting.phase8.pipeline import verify_phase8


def _minimal_completed_run(
    root: Path,
    *,
    holdout_opened: bool = False,
) -> None:
    root.mkdir(parents=True)
    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": root.name,
                "status": "succeeded",
            }
        ),
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "protocol": "phase8-v1",
                "holdout_opened": holdout_opened,
                "phase7_reference_run": "phase7-run",
            }
        ),
        encoding="utf-8",
    )
    files = [
        file_digest(
            root / "summary.json",
            relative_to=root,
        ).model_dump(mode="json")
    ]
    (root / "completion.json").write_text(
        json.dumps(
            {
                "files": files,
                "version": content_version(
                    {"files": files}
                ),
            }
        ),
        encoding="utf-8",
    )


def test_verify_phase8_accepts_hashed_completed_run(
    tmp_path: Path,
) -> None:
    root = tmp_path / "phase8-run"
    _minimal_completed_run(root)

    result = verify_phase8(root)

    assert result["run_id"] == "phase8-run"
    assert result["verified_files"] == 1
    assert result["protocol"] == "phase8-v1"
    assert result["phase7_reference_run"] == "phase7-run"


def test_verify_phase8_rejects_opened_holdout(
    tmp_path: Path,
) -> None:
    root = tmp_path / "phase8-run"
    _minimal_completed_run(
        root,
        holdout_opened=True,
    )

    with pytest.raises(ValueError, match="holdout"):
        verify_phase8(root)
