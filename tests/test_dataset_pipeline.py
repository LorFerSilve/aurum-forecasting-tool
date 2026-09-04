from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

from gold_forecasting.artifacts import (
    build_dataset_manifest,
    file_digest,
    write_manifest_atomic,
    write_parquet_atomic,
)
from gold_forecasting.datasets import (
    DatasetBuildError,
    build_mvp_dataset,
    load_mvp_model_table,
    validate_mvp_dataset,
)
from gold_forecasting.resampling import resample_candles

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fixture_config(tmp_path: Path) -> Path:
    shutil.copytree(PROJECT_ROOT / "configs", tmp_path / "configs")
    path = tmp_path / "configs" / "splits_mvp.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["splits"] = {
        "train": {
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-01-01T02:00:00Z",
        },
        "validation": {
            "start": "2024-01-01T02:00:00Z",
            "end": "2024-01-01T04:00:00Z",
        },
        "test": {
            "start": "2024-01-01T04:00:00Z",
            "end": "2025-01-01T00:00:00Z",
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return tmp_path / "configs" / "mvp.yaml"


def _one_minute_fixture() -> pd.DataFrame:
    periods = 360
    opens = pd.date_range("2024-01-01T00:00:00Z", periods=periods, freq="1min")
    base = 2_000.0 + np.arange(periods) * 0.01
    close = base + np.sin(np.arange(periods) / 4.0) * 0.08
    return pd.DataFrame(
        {
            "timestamp_open_utc": opens,
            "timestamp_close_utc": opens + pd.Timedelta(minutes=1),
            "bid_open": base,
            "bid_high": np.maximum(base, close) + 0.1,
            "bid_low": np.minimum(base, close) - 0.1,
            "bid_close": close,
            "instrument": "XAU_USD",
            "timeframe": "1min",
            "is_complete": True,
            "source": "histdata",
            "raw_file_hash": "a" * 64,
            "ingested_at_utc": datetime(2024, 1, 2, tzinfo=UTC),
            "dataset_version": "sha256:" + "b" * 64,
        }
    )


def _write_curated_layer(
    root: Path,
    timeframe: str,
    frame: pd.DataFrame,
    *,
    inputs=(),
):
    output = (
        root
        / "data"
        / "curated"
        / "histdata"
        / "XAU_USD"
        / timeframe
        / "source_year=2024"
        / "candles.parquet"
    )
    write_parquet_atomic(output, frame)
    manifest = build_dataset_manifest(
        layer="curated",
        source="histdata",
        instrument="XAU_USD",
        timeframe=timeframe,
        row_count=len(frame),
        inputs=inputs,
        outputs=(file_digest(output, relative_to=root),),
        parameters={"years": [2024], "build_scope": "complete"},
        period_start_utc=frame["timestamp_open_utc"].min().to_pydatetime(),
        period_end_utc=frame["timestamp_close_utc"].max().to_pydatetime(),
    )
    write_manifest_atomic(output.parents[1] / "manifest.json", manifest)
    return manifest


def _write_raw_layer(root: Path, frame: pd.DataFrame):
    directory = root / "data" / "raw" / "histdata" / "XAUUSD" / "1min"
    archive = directory / "HISTDATA_COM_ASCII_XAUUSD_M1_2024.zip"
    metadata = archive.with_suffix(".metadata.json")
    directory.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"fixture archive")
    metadata.write_text("{}\n", encoding="utf-8")
    manifest = build_dataset_manifest(
        layer="raw",
        source="histdata",
        instrument="XAU_USD",
        timeframe="1min",
        row_count=len(frame),
        inputs=(),
        outputs=tuple(
            file_digest(path, relative_to=root) for path in (archive, metadata)
        ),
        parameters={
            "years": [2024],
            "build_scope": "complete",
            "adapter": "histdata_ascii_m1",
            "source_timezone_offset": "-05:00",
            "source_observes_dst": False,
            "timestamp_semantics": "candle_open",
            "price_side": "bid",
            "volume_reliable": False,
        },
        period_start_utc=frame["timestamp_open_utc"].min().to_pydatetime(),
        period_end_utc=frame["timestamp_close_utc"].max().to_pydatetime(),
    )
    write_manifest_atomic(directory / "manifest.json", manifest)
    return manifest


def _write_curated_fixture(root: Path) -> None:
    one_minute = _one_minute_fixture()
    raw_manifest = _write_raw_layer(root, one_minute)
    manifest_1min = _write_curated_layer(
        root,
        "1min",
        one_minute,
        inputs=raw_manifest.outputs,
    )
    for timeframe in ("3min", "15min"):
        frame = resample_candles(one_minute, timeframe)
        _write_curated_layer(root, timeframe, frame, inputs=manifest_1min.outputs)


def test_dataset_build_twice_is_stable_and_validates_every_audit_row(tmp_path: Path) -> None:
    config_path = _fixture_config(tmp_path)
    _write_curated_fixture(tmp_path)

    first = build_mvp_dataset(config_path)
    first_table, _ = load_mvp_model_table(config_path)
    second = build_mvp_dataset(config_path)
    second_table, _ = load_mvp_model_table(config_path)

    assert first.dataset_version == second.dataset_version
    assert first.row_counts == second.row_counts
    assert set(first.row_counts) == {"train", "validation", "test"}
    assert all(count > 0 for count in first.row_counts.values())
    assert validate_mvp_dataset(config_path) == first.row_counts
    assert_frame_equal(second_table, first_table)
    sample_index = pd.read_parquet(second.sample_index_path)
    assert sample_index["sample_id"].tolist() == second_table["sample_id"].tolist()
    for split_name in ("train", "validation"):
        boundary = {
            "train": pd.Timestamp("2024-01-01T02:00:00Z"),
            "validation": pd.Timestamp("2024-01-01T04:00:00Z"),
        }[split_name]
        rows = second_table.loc[second_table["split"].eq(split_name)]
        assert rows["label_end_time_utc"].lt(boundary).all()


def test_dataset_validation_rejects_stale_feature_config(tmp_path: Path) -> None:
    config_path = _fixture_config(tmp_path)
    _write_curated_fixture(tmp_path)
    build_mvp_dataset(config_path)
    features_path = tmp_path / "configs" / "features_mvp.yaml"
    payload = yaml.safe_load(features_path.read_text(encoding="utf-8"))
    payload["return_lags"].append(4)
    features_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(DatasetBuildError, match="feature catalog"):
        validate_mvp_dataset(config_path)
