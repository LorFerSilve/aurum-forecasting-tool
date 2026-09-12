"""Context integration evidence, exact fallback, reproducibility and CLI gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from gold_forecasting.cli import app
from gold_forecasting.phase10.dry_run import run_context_dry_run, verify_context_dry_run
from gold_forecasting.phase10.preflight import inspect_context_source


@pytest.fixture(scope="module")
def completed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("context") / "run"
    run_context_dry_run(path)
    return path


def test_dry_run_proves_same_samples_fallback_and_scope(completed: Path) -> None:
    result = verify_context_dry_run(completed)
    assert result["verified_files"] == 8
    assert result["synthetic_only"] is True
    for field in ("formal_benchmark", "champion_changed", "holdout_opened", "trading_enabled"):
        assert result[field] is False
    assert result["coverage"]["stale_rows"] > 0
    assert result["event_calendar"]["missing_rows"] == 5
    assert set(result["event_calendar"]["phase_counts"]) == {"before", "during", "after", "normal"}
    assert result["event_calendar"]["used_as_model_features"] is False
    assert result["missing_source_coverage"]["usable_rows"] == 0
    variants = result["ablation"]["variants"]
    price, silver = variants["price_only_fixture"], variants["silver_fixture"]
    assert price["test_sample_digest"] == silver["test_sample_digest"]
    assert price["train_sample_digest"] == silver["train_sample_digest"]
    assert silver["fallback_rows"] > 0
    predicted = pd.read_parquet(completed / "predictions.parquet")
    left = predicted.loc[predicted["variant"].eq("price_only_fixture")].set_index("sample_id")
    right = predicted.loc[predicted["variant"].eq("silver_fixture")].set_index("sample_id")
    np.testing.assert_array_equal(left.index, right.index)
    fallback = right["used_price_only_fallback"]
    columns = ["p_down", "p_neutral", "p_up"]
    np.testing.assert_array_equal(left.loc[fallback, columns], right.loc[fallback, columns])
    absent = pd.read_parquet(completed / "missing_source_predictions.parquet")
    absent_price = absent.loc[absent["variant"].eq("price_only_fixture")].set_index("sample_id")
    absent_silver = absent.loc[absent["variant"].eq("silver_fixture")].set_index("sample_id")
    assert absent_silver["used_price_only_fallback"].all()
    np.testing.assert_array_equal(absent_price[columns], absent_silver[columns])
    np.testing.assert_array_equal(left[columns], absent_price[columns])
    assert result["missing_source_ablation"]["variants"]["silver_fixture"][
        "context_model_fitted"
    ] is False


def test_dry_run_is_reproducible_and_refuses_overwrite(completed: Path, tmp_path: Path) -> None:
    second = tmp_path / "second"
    run_context_dry_run(second)
    for name in ("features.parquet", "predictions.parquet"):
        pd.testing.assert_frame_equal(
            pd.read_parquet(completed / name), pd.read_parquet(second / name)
        )
    with pytest.raises(FileExistsError):
        run_context_dry_run(second)


def test_validation_rejects_corruption(completed: Path, tmp_path: Path) -> None:
    import shutil

    copied = tmp_path / "copied"
    shutil.copytree(completed, copied)
    with (copied / "observations.csv").open("a", encoding="utf-8") as stream:
        stream.write("corrupt")
    with pytest.raises(ValueError, match="integrity"):
        verify_context_dry_run(copied)


def test_cli_validates_completed_run(completed: Path) -> None:
    result = CliRunner().invoke(app, ["phase10", "validate", str(completed)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["verified_files"] == 8


def test_default_source_is_blocked_without_loading_any_bundle(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "configs" / "phase10_silver.yaml"
    result = inspect_context_source(source, tmp_path / "absent.manifest.json")
    assert result["status"] == "blocked"
    assert result["bundle_loaded"] is False
    assert "historical_release_evidence_unproven" in result["blockers"]
    cli = CliRunner().invoke(app, ["phase10", "preflight", "--source", str(source)])
    assert cli.exit_code == 1
    assert json.loads(cli.output)["formal_benchmark_ready"] is False
