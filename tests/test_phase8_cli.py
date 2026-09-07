"""Phase-8 CLI delegates to the guarded neural pipeline without hidden side effects."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import gold_forecasting.phase8.pipeline as phase8_pipeline
from gold_forecasting.cli import app


def test_phase8_run_command_uses_config_and_reports_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tmp_path / "phase8.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")
    output = tmp_path / "reports" / "phase8_runs" / "run-id"
    calls: list[Path] = []

    def run_phase8(path: str | Path) -> Path:
        calls.append(Path(path))
        return output

    monkeypatch.setattr(phase8_pipeline, "run_phase8", run_phase8)

    result = CliRunner().invoke(app, ["phase8", "run", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert calls == [config]
    assert f"Completed phase 8: {output}" in result.output


def test_phase8_validate_command_prints_verification_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_directory = tmp_path / "reports" / "phase8_runs" / "run-id"
    run_directory.mkdir(parents=True)
    calls: list[Path] = []

    def verify_phase8(path: str | Path) -> dict[str, object]:
        calls.append(Path(path))
        return {
            "run_id": "run-id",
            "protocol": "phase8-v1",
            "verified_files": 42,
        }

    monkeypatch.setattr(phase8_pipeline, "verify_phase8", verify_phase8)

    result = CliRunner().invoke(app, ["phase8", "validate", str(run_directory)])

    assert result.exit_code == 0, result.output
    assert calls == [run_directory]
    assert '"protocol": "phase8-v1"' in result.output
    assert '"verified_files": 42' in result.output
