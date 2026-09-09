"""Phase-10 completion sealing must precede the mutable registry terminal state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gold_forecasting.evaluation.walk_forward import make_walk_forward_folds
from gold_forecasting.phase10.config import Phase10Config
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
    config = Phase10Config()
    source = {
        "source_id": "silver",
        "description": "HistData XAGUSD M1 exploratory source with modeled historical availability",
        "source_url": "https://www.histdata.com/f-a-q/data-files-detailed-specification/",
        "market_hours": "OTC vendor quote stream; gaps and market closures stay missing/stale",
        "source_timezone": "Fixed UTC-05:00 without daylight saving time",
        "publication_delay_seconds": 60,
        "stale_after_seconds": 600,
        "availability_basis": "modeled_latency",
        "revision_policy": "append_only",
        "availability_evidence": (
            "Exploratory only. HistData does not provide historical per-row release timestamps."
        ),
        "missing_policy": "price_only_fallback",
        "enabled": True,
    }
    _write_json(root / "resolved_config.json", config.model_dump(mode="json"))
    (root / "config.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "protocol_version: phase10-silver-modeled-v1",
                "data_config: phase5.yaml",
                "features_config: features_phase7.yaml",
                "source_config: phase10_silver_exploratory.yaml",
                "phase7_champion_config: phase7_champion.yaml",
                "bundle_path: data/context/phase10/silver/silver.bundle-set.json",
                "archive_directory: data/raw/phase10/silver",
                "output_directory: reports/phase10_runs",
                "test_years: [2022, 2023, 2024]",
                "horizon_minutes: 15",
                "gap_minutes: 181",
                "seed: 20260906",
                "minimum_policy_trades: 20",
                "",
            ]
        ),
        encoding="utf-8",
    )
    configs = root / "configs"
    configs.mkdir()
    source_path = configs / "phase10_silver_exploratory.yaml"
    source_path.write_text(
        "\n".join(
            [
                "source_id: silver",
                "description: HistData XAGUSD M1 exploratory source with modeled historical availability",
                "source_url: https://www.histdata.com/f-a-q/data-files-detailed-specification/",
                "market_hours: OTC vendor quote stream; gaps and market closures stay missing/stale",
                "source_timezone: Fixed UTC-05:00 without daylight saving time",
                "publication_delay_seconds: 60",
                "stale_after_seconds: 600",
                "availability_basis: modeled_latency",
                "revision_policy: append_only",
                "availability_evidence: Exploratory only. HistData does not provide historical per-row release timestamps.",
                "missing_policy: price_only_fallback",
                "enabled: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_json(root / "source_config.json", source)
    config_sha = hashlib.sha256((root / "config.yaml").read_bytes()).hexdigest()
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
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
            "config_sha256": config_sha,
        },
    )
    _write_json(
        root / "preflight.json",
        {
            "status": "passed",
            "exploratory_ablation_ready": True,
            "formal_benchmark_ready": False,
            "strict_pit_source_ready": False,
            "formal_run_opened": False,
            "holdout_opened": False,
            "champion_changed": False,
            "trading_activated": False,
            "code_version": code_version,
            "data_version": data_version,
            "config_sha256": config_sha,
            "source_config_sha256": source_sha,
            "source": source,
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
