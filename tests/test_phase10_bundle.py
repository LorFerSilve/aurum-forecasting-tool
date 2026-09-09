"""Offline bundle admission keeps scope and provenance ahead of CSV parsing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from gold_forecasting.phase10 import bundle
from gold_forecasting.phase10.bundle import load_context_bundle
from gold_forecasting.phase10.contracts import OBSERVATION_COLUMNS, ContextSource


def _source() -> ContextSource:
    return ContextSource(
        source_id="silver",
        description="Silver test source with publication evidence",
        source_url="https://example.test/silver",
        market_hours="Explicit source records",
        source_timezone="UTC",
        publication_delay_seconds=60,
        stale_after_seconds=600,
        availability_basis="provider_timestamp",
        revision_policy="vintages",
        availability_evidence="Synthetic fixture representing provider release timestamps",
    )


def _rows() -> list[dict[str, object]]:
    row: dict[str, object] = {
        "source_id": "silver",
        "observation_id": "silver-20200101-1200",
        "observed_at_utc": "2020-01-01T12:00:00Z",
        "available_at_utc": "2020-01-01T12:01:00+00:00",
        "ingested_at_utc": "2026-09-09T00:00:00Z",
        "value": 18.0,
        "revision_id": "original",
        "source_uri": "https://example.test/silver/original",
        "raw_sha256": "a" * 64,
    }
    return [
        row,
        {
            **row,
            "available_at_utc": "2020-01-02T12:01:00Z",
            "revision_id": "corrected",
            "value": 19.0,
            "source_uri": "https://example.test/silver/corrected",
            "raw_sha256": "b" * 64,
        },
    ]


def _write_bundle(
    tmp_path: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    updates: dict[str, Any] | None = None,
    csv_payload: bytes | None = None,
) -> Path:
    observations = _rows() if rows is None else rows
    payload = (
        pd.DataFrame(observations, columns=OBSERVATION_COLUMNS).to_csv(index=False).encode()
        if csv_payload is None
        else csv_payload
    )
    (tmp_path / "observations.csv").write_bytes(payload)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_id": "silver",
        "availability_basis": "provider_timestamp",
        "observation_start_utc": "2020-01-01T00:00:00Z",
        "observation_end_utc": "2020-01-03T00:00:00+00:00",
        "row_count": len(observations),
        "file": "observations.csv",
        "sha256": hashlib.sha256(payload).hexdigest(),
        **(updates or {}),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_valid_bundle_preserves_revisions_and_upstream_hashes(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path)
    result = load_context_bundle(path, _source(), strict_pit=True)
    assert tuple(result.columns) == OBSERVATION_COLUMNS
    assert result["revision_id"].tolist() == ["original", "corrected"]
    assert result["value"].tolist() == [18.0, 19.0]
    assert result["raw_sha256"].tolist() == ["a" * 64, "b" * 64]
    assert str(result["observed_at_utc"].dtype) == "datetime64[ns, UTC]"


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"source_id": "other"}, "source_id"),
        ({"availability_basis": "modeled_latency"}, "availability_basis"),
        ({"observation_start_utc": "2019-12-31T23:59:59Z"}, "development"),
        ({"observation_end_utc": "2025-01-01T00:00:01Z"}, "development"),
        ({"observation_start_utc": "2020-01-03T00:00:00Z"}, "development"),
        ({"observation_end_utc": "2020-01-01T00:00:00Z"}, "development"),
        ({"observation_start_utc": "2020-01-01T00:00:00"}, "explicit ISO"),
        ({"observation_start_utc": "2020-01-01T01:00:00+01:00"}, "explicit ISO"),
        ({"observation_start_utc": "2020-02-30T00:00:00Z"}, "invalid UTC"),
        ({"schema_version": True}, "integer 1"),
        ({"schema_version": 2}, "schema_version"),
        ({"row_count": True}, "row_count"),
        ({"row_count": -1}, "row_count"),
        ({"row_count": "2"}, "row_count"),
        ({"extra": "forbidden"}, "Extra inputs"),
        ({"sha256": "not-a-hash"}, "sha256"),
    ],
)
def test_invalid_metadata_is_rejected_before_csv_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, Any],
    match: str,
) -> None:
    path = _write_bundle(tmp_path, updates=updates)
    original = bundle._read_bounded

    def metadata_only(file_path: Path, limit: int, name: str) -> bytes:
        assert name == "context manifest", "CSV was accessed before metadata admission"
        return original(file_path, limit, name)

    monkeypatch.setattr(bundle, "_read_bounded", metadata_only)
    with pytest.raises(ValueError, match=match):
        load_context_bundle(path, _source())


@pytest.mark.parametrize(
    "filename",
    ["../outside.csv", "..\\outside.csv", "/outside.csv", "C:\\outside.csv",
     "C:outside.csv", "sub/file.csv", "observations.csv:alternate", "data.txt", " data.csv"],
)
def test_unsafe_or_non_csv_paths_are_rejected(tmp_path: Path, filename: str) -> None:
    path = _write_bundle(tmp_path, updates={"file": filename})
    with pytest.raises(ValueError, match=r"single sibling \.csv"):
        load_context_bundle(path, _source())


def test_missing_csv_and_directory_raise_clear_errors(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path)
    csv_path = tmp_path / "observations.csv"
    csv_path.unlink()
    with pytest.raises(ValueError, match="context CSV is missing"):
        load_context_bundle(path, _source())
    csv_path.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        load_context_bundle(path, _source())


def test_csv_symlink_is_rejected(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path)
    csv_path = tmp_path / "observations.csv"
    target = tmp_path / "target.csv"
    csv_path.rename(target)
    try:
        csv_path.symlink_to(target)
    except OSError:
        pytest.skip("creating symlinks is not permitted on this host")
    with pytest.raises(ValueError, match="must not be a symlink"):
        load_context_bundle(path, _source())


def test_bad_checksum_precedes_csv_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_bundle(tmp_path, updates={"sha256": "0" * 64}, csv_payload=b"invalid CSV")

    def cannot_parse(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("CSV parsed before checksum verification")

    monkeypatch.setattr(pd, "read_csv", cannot_parse)
    with pytest.raises(ValueError, match="SHA-256"):
        load_context_bundle(path, _source())


def test_parsing_uses_the_verified_buffer_after_disk_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_bundle(tmp_path)
    original = bundle._read_bounded

    def mutate_after_read(file_path: Path, limit: int, name: str) -> bytes:
        payload = original(file_path, limit, name)
        if name == "context CSV":
            file_path.write_bytes(b"changed after the verified snapshot was read")
        return payload

    monkeypatch.setattr(bundle, "_read_bounded", mutate_after_read)
    assert load_context_bundle(path, _source())["value"].tolist() == [18.0, 19.0]


@pytest.mark.parametrize("name", ["observed_at_utc", "available_at_utc", "ingested_at_utc"])
@pytest.mark.parametrize("timestamp", ["2020-01-01T12:00:00", "2020-01-01T13:00:00+01:00"])
def test_csv_timestamps_require_explicit_utc(
    tmp_path: Path, name: str, timestamp: str,
) -> None:
    rows = _rows()
    rows[0][name] = timestamp
    path = _write_bundle(tmp_path, rows=rows)
    with pytest.raises(ValueError, match="explicit ISO timestamps in UTC"):
        load_context_bundle(path, _source())


def test_declared_span_is_half_open(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path, updates={"observation_end_utc": "2020-01-01T12:00:00Z"})
    with pytest.raises(ValueError, match="half-open bundle span"):
        load_context_bundle(path, _source())


def test_row_count_is_verified(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path, updates={"row_count": 1})
    with pytest.raises(ValueError, match="row_count differs"):
        load_context_bundle(path, _source())


def test_empty_bundle_retains_valid_schema(tmp_path: Path) -> None:
    result = load_context_bundle(_write_bundle(tmp_path, rows=[]), _source())
    assert result.empty
    assert str(result["observed_at_utc"].dtype) == "datetime64[ns, UTC]"


@pytest.mark.parametrize("column", ["value", "raw_sha256", "source_id"])
def test_invalid_row_contract_is_not_hidden(tmp_path: Path, column: str) -> None:
    rows = _rows()
    rows[0][column] = "invalid"
    path = _write_bundle(tmp_path, rows=rows)
    with pytest.raises(ValueError, match=column):
        load_context_bundle(path, _source())


def test_missing_value_remains_missing(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["value"] = ""
    result = load_context_bundle(_write_bundle(tmp_path, rows=rows), _source())
    assert pd.isna(result.loc[0, "value"])


def test_modeled_latency_is_rejected_by_strict_gate_before_csv_access(
    tmp_path: Path,
) -> None:
    source = _source().model_copy(update={"availability_basis": "modeled_latency"})
    path = _write_bundle(tmp_path, updates={"availability_basis": "modeled_latency"})
    assert len(load_context_bundle(path, source)) == 2
    (tmp_path / "observations.csv").unlink()
    with pytest.raises(ValueError, match="provider release evidence"):
        load_context_bundle(path, source, strict_pit=True)


@pytest.mark.parametrize("header", ["unexpected", "source_id,source_id"])
def test_csv_schema_must_match_exactly(tmp_path: Path, header: str) -> None:
    path = _write_bundle(tmp_path, csv_payload=header.encode(), updates={"row_count": 0})
    with pytest.raises(ValueError, match="exactly OBSERVATION_COLUMNS"):
        load_context_bundle(path, _source())


def test_csv_size_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_bundle(tmp_path)
    monkeypatch.setattr(bundle, "_MAX_CSV_BYTES", 16)
    with pytest.raises(ValueError, match="size limit"):
        load_context_bundle(path, _source())
