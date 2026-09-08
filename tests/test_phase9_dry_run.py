"""Synthetic full-stack dry-run for phase 9."""

from __future__ import annotations

from pathlib import Path

from gold_forecasting.phase9.dry_run import (
    run_phase9_synthetic_dry_run,
)


def test_phase9_synthetic_dry_run_reaches_verified_completion(
    tmp_path: Path,
) -> None:
    output = tmp_path / "phase9-synthetic"

    result = run_phase9_synthetic_dry_run(
        output
    )

    assert result["protocol"] == "phase9-v1"
    assert (
        result["phase8_reference_run"]
        == "synthetic-phase8-reference"
    )
    assert result["verified_files"] > 20
    assert (
        output
        / "test_2022"
        / "direct"
        / "outer_predictions.parquet"
    ).exists()
    assert (
        output
        / "test_2022"
        / "recursive"
        / "outer_predictions.parquet"
    ).exists()
    assert (
        output
        / "completion.json"
    ).exists()
