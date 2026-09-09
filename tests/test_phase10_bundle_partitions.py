"""Partitioned and Parquet Phase-10 observation bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from gold_forecasting.phase10.bundle import (
    load_context_bundle,
    load_context_bundle_set,
    load_context_data,
)
from gold_forecasting.phase10.contracts import OBSERVATION_COLUMNS, ContextSource


def _source() -> ContextSource:
    return ContextSource(
        source_id="silver",
        description="Partitioned silver fixture",
        source_url="https://example.test/silver",
        market_hours="fixture",
        source_timezone="UTC",
        publication_delay_seconds=60,
        stale_after_seconds=600,
        availability_basis="modeled_latency",
        revision_policy="append_only",
        availability_evidence="Modeled fixture latency only",
        enabled=True,
    )


def _row(timestamp: str, value: float, raw_hash: str) -> dict[str, object]:
    observed = pd.Timestamp(timestamp, tz="UTC")
    return {
        "source_id": "silver",
        "observation_id": "silver-" + observed.strftime("%Y%m%dT%H%M%SZ"),
        "observed_at_utc": observed,
        "available_at_utc": observed + pd.Timedelta(minutes=1),
        "ingested_at_utc": pd.Timestamp("2026-09-09T00:00:00Z"),
        "value": value,
        "revision_id": "sha256:" + raw_hash,
        "source_uri": "https://example.test/silver",
        "raw_sha256": raw_hash,
    }


def _write_parquet_bundle(
    root: Path,
    *,
    name: str,
    rows: list[dict[str, object]],
) -> str:
    frame = pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)
    data_name = name + ".parquet"
    data_path = root / data_name
    frame.to_parquet(data_path, index=False)
    payload = data_path.read_bytes()
    observed = frame["observed_at_utc"]
    manifest_name = name + ".manifest.json"
    (root / manifest_name).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_id": "silver",
                "availability_basis": "modeled_latency",
                "observation_start_utc": observed.min().isoformat(),
                "observation_end_utc": (
                    observed.max() + pd.Timedelta(nanoseconds=1)
                ).isoformat(),
                "row_count": len(frame),
                "format": "parquet",
                "file": data_name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return manifest_name


def test_parquet_bundle_roundtrip(tmp_path: Path) -> None:
    manifest = _write_parquet_bundle(
        tmp_path,
        name="silver-2020",
        rows=[_row("2020-01-02T12:00:00", 18.0, "a" * 64)],
    )

    result = load_context_bundle(tmp_path / manifest, _source())

    assert tuple(result.columns) == OBSERVATION_COLUMNS
    assert result["value"].tolist() == [18.0]
    assert str(result["observed_at_utc"].dtype) == "datetime64[ns, UTC]"


def test_bundle_set_concatenates_and_revalidates_partitions(tmp_path: Path) -> None:
    first = _write_parquet_bundle(
        tmp_path,
        name="silver-2020",
        rows=[_row("2020-01-02T12:00:00", 18.0, "a" * 64)],
    )
    second = _write_parquet_bundle(
        tmp_path,
        name="silver-2021",
        rows=[_row("2021-01-02T12:00:00", 25.0, "b" * 64)],
    )
    set_path = tmp_path / "silver.bundle-set.json"
    set_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "bundle_set",
                "source_id": "silver",
                "availability_basis": "modeled_latency",
                "bundles": [first, second],
            }
        ),
        encoding="utf-8",
    )

    result = load_context_bundle_set(set_path, _source())

    assert result["value"].tolist() == [18.0, 25.0]
    assert load_context_data(set_path, _source()).equals(result)


def test_bundle_set_rejects_duplicate_observation_across_partitions(tmp_path: Path) -> None:
    row = _row("2020-01-02T12:00:00", 18.0, "a" * 64)
    first = _write_parquet_bundle(tmp_path, name="part-a", rows=[row])
    second = _write_parquet_bundle(tmp_path, name="part-b", rows=[row])
    set_path = tmp_path / "silver.bundle-set.json"
    set_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "bundle_set",
                "source_id": "silver",
                "availability_basis": "modeled_latency",
                "bundles": [first, second],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate observation version identity"):
        load_context_bundle_set(set_path, _source())


@pytest.mark.parametrize(
    "bundle_name",
    ["../part.manifest.json", "sub/part.manifest.json", "part.json", " part.manifest.json"],
)
def test_bundle_set_rejects_unsafe_partition_paths(
    tmp_path: Path,
    bundle_name: str,
) -> None:
    path = tmp_path / "silver.bundle-set.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "bundle_set",
                "source_id": "silver",
                "availability_basis": "modeled_latency",
                "bundles": [bundle_name],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="single sibling"):
        load_context_bundle_set(path, _source())


def test_bundle_format_must_match_file_suffix(tmp_path: Path) -> None:
    path = tmp_path / "bad.manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_id": "silver",
                "availability_basis": "modeled_latency",
                "observation_start_utc": "2020-01-01T00:00:00Z",
                "observation_end_utc": "2020-01-02T00:00:00Z",
                "row_count": 0,
                "format": "parquet",
                "file": "observations.csv",
                "sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"requires a \\.parquet file"):
        load_context_bundle(path, _source())
