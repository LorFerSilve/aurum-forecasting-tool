"""Metadata-only scope protection must run before any observation I/O."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from gold_forecasting.artifacts import (
    DatasetManifest,
    FileDigest,
    build_dataset_manifest,
    write_json_atomic,
    write_manifest_atomic,
)
from gold_forecasting.benchmark.data_guard import (
    BenchmarkDataGuardError,
    preflight_development_inputs,
)
from gold_forecasting.config import ProjectConfig, load_project_config
from gold_forecasting.data_pipeline import _manifest_path, _timeframes, _year_output_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YEARS = tuple(range(2020, 2025))


def _digest(path: Path, root: Path) -> FileDigest:
    return FileDigest(path=path.relative_to(root).as_posix(), sha256="a" * 64, size_bytes=0)


def _fixture_project(tmp_path: Path, name: str = "phase5.yaml") -> ProjectConfig:
    source = load_project_config(PROJECT_ROOT / "configs" / name)
    project = source.model_copy(update={"config_path": tmp_path / "configs" / name})
    raw_path = _manifest_path(tmp_path, project, "raw", "1min")
    archives = tuple(raw_path.parent / f"HISTDATA_COM_ASCII_XAUUSD_M1_{year}.zip" for year in YEARS)
    sidecars = tuple(path.with_suffix(".metadata.json") for path in archives)
    raw_outputs = tuple(_digest(path, tmp_path) for path in (*archives, *sidecars))
    common: dict[str, Any] = {
        "source": "histdata",
        "instrument": "XAU_USD",
        "row_count": 5,
        "period_start_utc": datetime(2020, 1, 2, tzinfo=UTC),
        "period_end_utc": datetime(2024, 12, 31, tzinfo=UTC),
        "parameters": {"years": list(YEARS), "build_scope": "complete"},
        "created_at_utc": datetime(2026, 9, 6, tzinfo=UTC),
    }
    raw = build_dataset_manifest(
        layer="raw", timeframe="1min", inputs=(), outputs=raw_outputs, **common
    )
    write_manifest_atomic(raw_path, raw)
    minute_outputs = tuple(
        _digest(_year_output_path(tmp_path, project, "1min", year), tmp_path) for year in YEARS
    )
    for timeframe in _timeframes(project):
        manifest = build_dataset_manifest(
            layer="curated",
            timeframe=timeframe,
            inputs=raw_outputs if timeframe == "1min" else minute_outputs,
            outputs=tuple(
                _digest(_year_output_path(tmp_path, project, timeframe, year), tmp_path)
                for year in YEARS
            ),
            **common,
        )
        write_manifest_atomic(_manifest_path(tmp_path, project, "curated", timeframe), manifest)
    for year, archive, sidecar in zip(YEARS, archives, sidecars, strict=True):
        write_json_atomic(
            sidecar,
            {
                "source": "histdata",
                "source_symbol": "XAUUSD",
                "timeframe": "1min",
                "year": year,
                "archive": archive.name,
                "first_ingested_at_utc": "2026-09-06T00:00:00Z",
            },
        )
    return project


def _rewrite(project: ProjectConfig, layer: str, timeframe: str, **updates: Any) -> None:
    path = _manifest_path(project.config_path.parent.parent, project, layer, timeframe)
    payload = dict(DatasetManifest.model_validate_json(path.read_text(encoding="utf-8")))
    payload.update(updates)
    payload.pop("dataset_version")
    payload.pop("schema_version")
    write_manifest_atomic(path, build_dataset_manifest(**payload))


@pytest.fixture(autouse=True)
def no_observation_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("preflight must not read or hash observation files")

    monkeypatch.setattr("pandas.read_parquet", forbidden)
    monkeypatch.setattr("gold_forecasting.artifacts.sha256_file", forbidden)
    monkeypatch.setattr("gold_forecasting.data_pipeline.sha256_file", forbidden)


@pytest.mark.parametrize("name", ["phase5.yaml", "mvp.yaml"])
def test_valid_metadata_passes_without_any_observation_files(tmp_path: Path, name: str) -> None:
    project = _fixture_project(tmp_path, name)
    assert not list(tmp_path.rglob("*.parquet"))
    assert not list(tmp_path.rglob("*.zip"))
    preflight_development_inputs(project)


def test_empty_optional_timeframes_are_allowed(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path)
    for timeframe in ("1d", "1mo"):
        _rewrite(
            project, "curated", timeframe, row_count=0, period_start_utc=None, period_end_utc=None
        )
    preflight_development_inputs(project)


@pytest.mark.parametrize(
    "layer,timeframe", [("raw", "1min"), ("curated", "1min"), ("curated", "1mo")]
)
@pytest.mark.parametrize(
    "updates",
    [
        {"period_start_utc": datetime(2019, 12, 31, tzinfo=UTC)},
        {"period_end_utc": datetime(2025, 1, 1, 0, 1, tzinfo=UTC)},
        {"period_start_utc": None},
        {"period_end_utc": None},
        {"parameters": {"years": list(range(2020, 2026)), "build_scope": "complete"}},
        {"parameters": {"years": [2024], "build_scope": "partial"}},
    ],
)
def test_unsafe_manifest_bounds_or_years_rejected(
    tmp_path: Path, layer: str, timeframe: str, updates: dict[str, Any]
) -> None:
    project = _fixture_project(tmp_path)
    _rewrite(project, layer, timeframe, **updates)
    with pytest.raises(BenchmarkDataGuardError):
        preflight_development_inputs(project)


@pytest.mark.parametrize("which", ["guard", "override", "start", "end"])
def test_changed_configuration_is_rejected_before_reading_manifests(
    tmp_path: Path, which: str
) -> None:
    project = load_project_config(PROJECT_ROOT / "configs" / "phase5.yaml")
    project = project.model_copy(update={"config_path": tmp_path / "configs" / "phase5.yaml"})
    splits = project.splits
    if which in {"guard", "override"}:
        updates = (
            {"reject_at_or_after": datetime(2026, 1, 1, tzinfo=UTC)}
            if which == "guard"
            else {"allow_holdout_override": True}
        )
        splits = splits.model_copy(
            update={"development_guard": splits.development_guard.model_copy(update=updates)}
        )
    else:
        component = "train" if which == "start" else "test"
        block = getattr(splits.splits, component).model_copy(
            update={which: datetime(2019 if which == "start" else 2026, 1, 1, tzinfo=UTC)}
        )
        splits = splits.model_copy(
            update={"splits": splits.splits.model_copy(update={component: block})}
        )
    project = project.model_copy(update={"splits": splits})
    with pytest.raises(BenchmarkDataGuardError, match=r"boundary|exactly"):
        preflight_development_inputs(project)


@pytest.mark.parametrize(
    "layer,timeframe,field,path",
    [
        (
            "raw",
            "1min",
            "outputs",
            "data/raw/histdata/XAUUSD/1min/HISTDATA_COM_ASCII_XAUUSD_M1_2025.zip",
        ),
        (
            "raw",
            "1min",
            "inputs",
            "data/raw/histdata/XAUUSD/1min/HISTDATA_COM_ASCII_XAUUSD_M1_2025.zip",
        ),
        (
            "curated",
            "1min",
            "outputs",
            "data/curated/histdata/XAU_USD/phase5/1min/utc_year=2025/candles.parquet",
        ),
        (
            "curated",
            "3h",
            "inputs",
            "data/curated/histdata/XAU_USD/phase5/1min/utc_year=2025/candles.parquet",
        ),
        (
            "curated",
            "1min",
            "inputs",
            "data/raw/histdata/XAUUSD/1min/HISTDATA_COM_ASCII_XAUUSD_M1_2025.zip",
        ),
        ("curated", "1min", "inputs", "../outside/holdout.parquet"),
        ("curated", "1min", "inputs", "C:/outside/holdout.parquet"),
    ],
)
def test_all_input_output_references_are_scoped_before_hashing(
    tmp_path: Path, layer: str, timeframe: str, field: str, path: str
) -> None:
    project = _fixture_project(tmp_path)
    _rewrite(
        project,
        layer,
        timeframe,
        **{field: (FileDigest(path=path, sha256="b" * 64, size_bytes=0),)},
    )
    with pytest.raises(BenchmarkDataGuardError):
        preflight_development_inputs(project)


def test_missing_last_timeframe_manifest_is_rejected_without_loading_earlier_data(
    tmp_path: Path,
) -> None:
    project = _fixture_project(tmp_path)
    _manifest_path(tmp_path, project, "curated", "1mo").unlink()
    with pytest.raises(BenchmarkDataGuardError, match="invalid development manifest"):
        preflight_development_inputs(project)


def test_referenced_sidecar_year_is_checked_but_2026_acquisition_date_is_allowed(
    tmp_path: Path,
) -> None:
    project = _fixture_project(tmp_path)
    preflight_development_inputs(project)
    sidecar = (
        _manifest_path(tmp_path, project, "raw", "1min").parent
        / "HISTDATA_COM_ASCII_XAUUSD_M1_2024.metadata.json"
    )
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["year"] = 2025
    write_json_atomic(sidecar, metadata)
    with pytest.raises(BenchmarkDataGuardError, match="source/year"):
        preflight_development_inputs(project)


def test_redirected_allowed_resource_path_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _fixture_project(tmp_path)
    expected = _year_output_path(tmp_path, project, "1min", 2024)
    original = Path.resolve

    def redirected(path: Path, strict: bool = False) -> Path:
        if path == expected:
            return tmp_path / "hidden_holdout.parquet"
        return original(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", redirected)
    with pytest.raises(BenchmarkDataGuardError, match="redirected"):
        preflight_development_inputs(project)
