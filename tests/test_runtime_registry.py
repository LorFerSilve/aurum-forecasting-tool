from __future__ import annotations

import hashlib
import io
import json
import os
import random
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gold_forecasting.registry import RunRegistry
from gold_forecasting.runtime import configure_logging, format_utc, seed_everything, utc_now


def test_configure_logging_emits_an_rfc3339_utc_timestamp() -> None:
    stream = io.StringIO()
    logger = configure_logging(stream=stream)

    logger.info("runtime ready")

    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z INFO root: runtime ready\n",
        stream.getvalue(),
    )


def test_utc_clock_and_formatter_are_timezone_explicit() -> None:
    timestamp = utc_now()

    assert timestamp.utcoffset() is not None
    assert format_utc(timestamp).endswith("Z")


def test_seed_everything_repeats_python_and_numpy_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    seed_everything(20260904)
    first_python = [random.random() for _ in range(4)]
    first_numpy = np.random.random(4)
    seed_everything(20260904)

    assert [random.random() for _ in range(4)] == first_python
    np.testing.assert_array_equal(np.random.random(4), first_numpy)
    assert os.environ["PYTHONHASHSEED"] == "20260904"


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_everything_rejects_values_numpy_cannot_seed(seed: int) -> None:
    with pytest.raises(ValueError):
        seed_everything(seed)


@pytest.mark.parametrize("seed", [True, 1.5, "1"])
def test_seed_everything_rejects_non_integer_values(seed: object) -> None:
    with pytest.raises(TypeError):
        seed_everything(seed)  # type: ignore[arg-type]


def test_registry_snapshots_config_and_records_required_versions(tmp_path: Path) -> None:
    config_path = tmp_path / "source" / "mvp.yaml"
    config_path.parent.mkdir()
    config_bytes = b"run:\n  seed: 42\n"
    config_path.write_bytes(config_bytes)
    registry = RunRegistry(
        tmp_path / "runs",
        config_path,
        "dataset-sha256:abc123",
        code_root=tmp_path,
    )

    record = registry.start_run(dry_run=True)
    manifest = json.loads(record.manifest_path.read_text(encoding="utf-8"))

    assert record.status == "running"
    assert record.dry_run is True
    assert record.output_directory.parent == (tmp_path / "runs").resolve()
    assert record.config_snapshot_path.read_bytes() == config_bytes
    assert manifest["run_id"] == record.run_id
    assert manifest["status"] == "running"
    assert manifest["dry_run"] is True
    assert manifest["config_path"] == str(config_path.resolve())
    assert manifest["config_snapshot_path"] == str(record.config_snapshot_path)
    assert manifest["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    assert manifest["data_version"] == "dataset-sha256:abc123"
    assert manifest["code_version"] in {"uncommitted", "unavailable"}
    assert manifest["output_directory"] == str(record.output_directory)
    assert manifest["finished_at"] is None
    assert manifest["metadata"] == {}
    assert record.started_at.endswith("Z")


def test_finish_run_atomically_sets_terminal_status(tmp_path: Path) -> None:
    config_path = tmp_path / "mvp.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    registry = RunRegistry(tmp_path / "runs", config_path, "unbuilt", code_root=tmp_path)
    running = registry.start_run()

    finished = registry.finish_run(
        running,
        status="succeeded",
        metadata={"rows": 0, "stage": "dry-run"},
    )
    manifest = json.loads(running.manifest_path.read_text(encoding="utf-8"))

    assert running.status == "running"
    assert finished.status == "succeeded"
    assert finished.finished_at is not None and finished.finished_at.endswith("Z")
    assert manifest["status"] == "succeeded"
    assert manifest["finished_at"] == finished.finished_at
    assert manifest["metadata"] == {"rows": 0, "stage": "dry-run"}
    assert list(running.output_directory.glob("*.tmp")) == []
    with pytest.raises(RuntimeError, match="already"):
        registry.finish_run(running)


def test_finish_run_does_not_replace_manifest_when_serialization_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "mvp.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    registry = RunRegistry(tmp_path / "runs", config_path, "unbuilt", code_root=tmp_path)
    running = registry.start_run()
    original_manifest = running.manifest_path.read_bytes()

    with pytest.raises(TypeError):
        registry.finish_run(running, metadata={"not_json": object()})

    assert running.manifest_path.read_bytes() == original_manifest


def test_registry_rejects_a_record_from_another_output_root(tmp_path: Path) -> None:
    config_path = tmp_path / "mvp.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    registry = RunRegistry(tmp_path / "runs", config_path, "unbuilt", code_root=tmp_path)
    running = registry.start_run()
    foreign = replace(
        running,
        output_directory=tmp_path / "elsewhere" / running.run_id,
        manifest_path=tmp_path / "elsewhere" / running.run_id / "run.json",
    )

    with pytest.raises(ValueError, match="does not belong"):
        registry.finish_run(foreign)


def test_registry_requires_an_existing_config(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs", tmp_path / "missing.yaml", "unbuilt")

    with pytest.raises(FileNotFoundError, match="config file does not exist"):
        registry.start_run()
