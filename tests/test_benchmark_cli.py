"""CLI reproduction preserves completed runs and writes bounded audit reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import gold_forecasting.benchmark.reproduce as reproduction
from gold_forecasting.cli import app


@pytest.fixture
def run_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "reports" / "benchmark_runs" / "frozen-run"
    directory.mkdir(parents=True)
    (directory / "completion.json").write_text('{"frozen": true}', encoding="utf-8")
    return directory


def _successful_report(directory: Path) -> dict[str, Any]:
    return {
        "run_id": directory.name,
        "final_fit_parity_count": 72,
        "inner_policy_parity_count": 72,
        "verified_files": 2320,
        "original_run_modified": False,
        "parities": [{"test_rows": 1000}] * 72,
    }


@pytest.mark.parametrize("custom_path", [False, True])
def test_reproduce_writes_report_outside_original_and_prints_counts(
    run_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    custom_path: bool,
) -> None:
    calls: list[Path] = []

    def reproduce(directory: Path) -> dict[str, Any]:
        calls.append(directory)
        return _successful_report(directory)

    monkeypatch.setattr(reproduction, "reproduce_benchmark", reproduce)
    destination = (
        tmp_path / "audit" / "reproduction.json"
        if custom_path
        else run_directory.parent.parent / "phase6_reproduction_frozen-run.json"
    )
    args = ["benchmark", "reproduce", str(run_directory)]
    if custom_path:
        args += ["--report", str(destination)]
    original = (run_directory / "completion.json").read_bytes()

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 0, result.output
    assert calls == [run_directory.resolve()]
    assert json.loads(destination.read_text(encoding="utf-8")) == _successful_report(run_directory)
    assert "final fits=72, inner policies=72, verified files=2320" in result.output
    assert str(destination.resolve()) in result.output
    assert "test_rows" not in result.output
    assert list(run_directory.iterdir()) == [run_directory / "completion.json"]
    assert (run_directory / "completion.json").read_bytes() == original


@pytest.mark.parametrize("target", ["inside", "traversal_inside", "existing", "default_existing"])
def test_reproduce_rejects_bad_report_before_refitting(
    run_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    calls: list[Path] = []

    def unexpected_reproduction(directory: Path) -> dict[str, Any]:
        calls.append(directory)
        pytest.fail("invalid report destination must fail before reproducing")

    monkeypatch.setattr(reproduction, "reproduce_benchmark", unexpected_reproduction)
    args = ["benchmark", "reproduce", str(run_directory)]
    if target == "inside":
        destination = run_directory / "reproduction.json"
    elif target == "traversal_inside":
        destination = run_directory.parent / "frozen-run" / ".." / "frozen-run" / "report.json"
    elif target == "existing":
        destination = tmp_path / "existing.json"
        destination.write_text("preserve existing report", encoding="utf-8")
    else:
        destination = run_directory.parent.parent / "phase6_reproduction_frozen-run.json"
        destination.write_text("preserve existing report", encoding="utf-8")
    if target != "default_existing":
        args += ["--report", str(destination)]

    result = CliRunner().invoke(app, args)

    assert result.exit_code != 0
    assert not calls
    if "existing" in target:
        assert destination.read_text(encoding="utf-8") == "preserve existing report"
    else:
        assert not destination.exists()
    assert list(run_directory.iterdir()) == [run_directory / "completion.json"]


def test_reproduction_failure_does_not_write_report(
    run_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(directory: Path) -> dict[str, Any]:
        raise reproduction.BenchmarkReproductionError("probability parity failed")

    monkeypatch.setattr(reproduction, "reproduce_benchmark", fail)
    report = tmp_path / "not-created" / "report.json"

    result = CliRunner().invoke(
        app, ["benchmark", "reproduce", str(run_directory), "--report", str(report)]
    )

    assert result.exit_code == 1
    assert "probability parity failed" in result.output
    assert not report.parent.exists()
    assert list(run_directory.iterdir()) == [run_directory / "completion.json"]


def test_report_created_during_refit_is_not_overwritten(
    run_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "concurrent-report.json"

    def reproduce(directory: Path) -> dict[str, Any]:
        report.write_text("other completed reproduction", encoding="utf-8")
        return _successful_report(directory)

    monkeypatch.setattr(reproduction, "reproduce_benchmark", reproduce)

    result = CliRunner().invoke(
        app, ["benchmark", "reproduce", str(run_directory), "--report", str(report)]
    )

    assert result.exit_code == 1
    assert "report appeared during reproduction" in result.output
    assert report.read_text(encoding="utf-8") == "other completed reproduction"
