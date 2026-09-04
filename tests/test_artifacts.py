from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gold_forecasting.artifacts import (
    ArtifactError,
    build_dataset_manifest,
    content_version,
    file_digest,
    sha256_file,
    write_bytes_atomic,
    write_json_atomic,
    write_manifest_atomic,
)


def test_file_digest_streams_known_content(tmp_path: Path) -> None:
    artifact = tmp_path / "data" / "sample.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"gold\n")

    digest = file_digest(artifact, relative_to=tmp_path)

    assert digest.path == "data/sample.bin"
    assert digest.size_bytes == 5
    assert digest.sha256 == sha256_file(artifact)
    assert digest.sha256 == "3bb0ac0514ee5ab7e91040c7aba0e969bfb308a4035f3c883f3877e2e83f9dec"


def test_content_version_is_order_independent_and_rejects_nan() -> None:
    assert content_version({"a": 1, "b": 2}) == content_version({"b": 2, "a": 1})
    with pytest.raises(ArtifactError, match="canonical JSON"):
        content_version({"invalid": float("nan")})


def test_manifest_version_ignores_creation_time(tmp_path: Path) -> None:
    artifact = tmp_path / "sample.parquet"
    artifact.write_bytes(b"same content")
    output = file_digest(artifact)
    common = {
        "layer": "curated",
        "source": "histdata",
        "instrument": "XAU_USD",
        "timeframe": "1min",
        "row_count": 2,
        "inputs": (),
        "outputs": (output,),
        "parameters": {"timezone": "UTC"},
        "period_start_utc": datetime(2024, 1, 1, tzinfo=UTC),
        "period_end_utc": datetime(2024, 1, 2, tzinfo=UTC),
    }

    first = build_dataset_manifest(
        **common,
        created_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    second = build_dataset_manifest(
        **common,
        created_at_utc=datetime(2026, 2, 1, tzinfo=UTC),
    )

    assert first.dataset_version == second.dataset_version
    assert first.created_at_utc != second.created_at_utc


def test_atomic_manifest_is_valid_json_and_leaves_no_temp_file(tmp_path: Path) -> None:
    artifact = tmp_path / "sample.parquet"
    artifact.write_bytes(b"content")
    manifest = build_dataset_manifest(
        layer="raw",
        source="histdata",
        instrument="XAU_USD",
        timeframe="1min",
        row_count=0,
        inputs=(),
        outputs=(file_digest(artifact),),
        parameters={},
        period_start_utc=None,
        period_end_utc=None,
    )
    destination = tmp_path / "manifest.json"

    write_manifest_atomic(destination, manifest)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["dataset_version"] == manifest.dataset_version
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_json_rejects_non_serializable_payload_before_writing(tmp_path: Path) -> None:
    destination = tmp_path / "invalid.json"
    with pytest.raises(TypeError):
        write_json_atomic(destination, {"invalid": object()})
    assert not destination.exists()


def test_atomic_bytes_replace_existing_content(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"old")

    write_bytes_atomic(destination, b"new")

    assert destination.read_bytes() == b"new"
    assert list(tmp_path.glob("*.tmp")) == []
