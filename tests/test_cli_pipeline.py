from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

import gold_forecasting.pipeline as pipeline_module
from gold_forecasting.cli import app
from gold_forecasting.pipeline import MVP_STAGES, run_mvp_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _isolated_config(tmp_path: Path) -> Path:
    config_directory = tmp_path / "configs"
    shutil.copytree(PROJECT_ROOT / "configs", config_directory)
    root_path = config_directory / "mvp.yaml"
    payload = yaml.safe_load(root_path.read_text(encoding="utf-8"))
    payload["run"]["output_root"] = str(tmp_path / "runs")
    root_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return root_path


def test_config_validate_command() -> None:
    result = CliRunner().invoke(
        app,
        ["config", "validate", "--config", str(PROJECT_ROOT / "configs" / "mvp.yaml")],
    )

    assert result.exit_code == 0
    assert "Valid configuration: XAU_USD, protocol v0.0" in result.stdout


def test_dry_run_exercises_all_stage_boundaries_and_records_them(tmp_path: Path) -> None:
    config_path = _isolated_config(tmp_path)

    record = run_mvp_pipeline(config_path, dry_run=True)
    manifest = json.loads(record.manifest_path.read_text(encoding="utf-8"))

    assert record.status == "succeeded"
    assert manifest["metadata"]["instrument"] == "XAU_USD"
    assert [stage["name"] for stage in manifest["metadata"]["stages"]] == list(MVP_STAGES)
    assert all(
        stage["status"] == "validated_placeholder" for stage in manifest["metadata"]["stages"]
    )
    assert set(manifest["metadata"]["component_config_sha256"]) == {
        "instrument",
        "features",
        "labels",
        "costs",
        "splits",
        "model",
        "backtest",
    }
    assert len(list((record.output_directory / "config_components").glob("*.yaml"))) == 7


def test_real_pipeline_records_completed_stage_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _isolated_config(tmp_path)
    dataset_version = "sha256:" + "a" * 64
    prepared = SimpleNamespace(
        dataset=SimpleNamespace(dataset_version=dataset_version),
        data_action="reused",
    )
    analysis = SimpleNamespace(
        training=SimpleNamespace(bundle=SimpleNamespace(model_version="v0.1.0+abc123")),
        test_rows=123,
        backtest=SimpleNamespace(metrics=SimpleNamespace(executed_trade_count=7)),
        example_prediction=SimpleNamespace(predicted_class="up"),
    )
    monkeypatch.setattr(pipeline_module, "_prepare_mvp_artifacts", lambda config: prepared)
    monkeypatch.setattr(
        pipeline_module,
        "run_mvp_analysis",
        lambda config, output: analysis,
    )

    record = run_mvp_pipeline(config_path)
    manifest = json.loads(record.manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "succeeded"
    assert manifest["finished_at"].endswith("Z")
    assert manifest["data_version"] == dataset_version
    assert manifest["metadata"]["mode"] == "research_mvp_v0.1"
    assert [stage["name"] for stage in manifest["metadata"]["stages"]] == list(MVP_STAGES)
