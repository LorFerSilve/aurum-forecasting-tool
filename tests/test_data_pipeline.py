from __future__ import annotations

import io
import shutil
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import yaml

from gold_forecasting.artifacts import (
    DatasetManifest,
    FileDigest,
    build_dataset_manifest,
    write_manifest_atomic,
)
from gold_forecasting.data_pipeline import (
    DataBuildError,
    build_mvp_data,
    validate_existing_mvp_data,
)
from gold_forecasting.ingestion import HistDataM1Provider

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _isolated_config(tmp_path: Path) -> Path:
    shutil.copytree(PROJECT_ROOT / "configs", tmp_path / "configs")
    return tmp_path / "configs" / "mvp.yaml"


def _archive_bytes() -> bytes:
    start = datetime(2024, 1, 15, 10, 0)
    rows: list[str] = []
    for offset in range(20):
        timestamp = (start + timedelta(minutes=offset)).strftime("%Y%m%d %H%M%S")
        price = 2025.0 + offset / 10
        rows.append(
            f"{timestamp};{price:.2f};{price + 0.2:.2f};{price - 0.2:.2f};{price + 0.05:.2f};0"
        )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("DAT_ASCII_XAUUSD_M1_2024.csv", "\n".join(rows) + "\n")
    return buffer.getvalue()


def _provider() -> tuple[HistDataM1Provider, list[httpx.Request]]:
    requests: list[httpx.Request] = []
    payload = _archive_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            html = """
            <form id="file_down" method="POST" action="/get.php">
              <input type="hidden" name="tk" value="0123456789abcdef">
              <input type="hidden" name="date" value="2024">
              <input type="hidden" name="datemonth" value="2024">
              <input type="hidden" name="platform" value="ASCII">
              <input type="hidden" name="timeframe" value="M1">
              <input type="hidden" name="fxpair" value="XAUUSD">
            </form>
            """
            return httpx.Response(200, text=html)
        return httpx.Response(200, content=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = HistDataM1Provider(
        client=client,
        clock=lambda: datetime(2026, 9, 4, tzinfo=UTC),
    )
    return provider, requests


def _rewrite_raw_manifest(
    path: Path,
    *,
    parameters: dict[str, object] | None = None,
    outputs: tuple[FileDigest, ...] | None = None,
) -> None:
    manifest = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
    replacement = build_dataset_manifest(
        layer=manifest.layer,
        source=manifest.source,
        instrument=manifest.instrument,
        timeframe=manifest.timeframe,
        row_count=manifest.row_count,
        inputs=manifest.inputs,
        outputs=manifest.outputs if outputs is None else outputs,
        parameters=manifest.parameters if parameters is None else parameters,
        period_start_utc=manifest.period_start_utc,
        period_end_utc=manifest.period_end_utc,
        created_at_utc=manifest.created_at_utc,
    )
    write_manifest_atomic(path, replacement)


def test_phase2_build_is_reproducible_and_validates_outputs(tmp_path: Path) -> None:
    config_path = _isolated_config(tmp_path)
    provider, requests = _provider()

    first = build_mvp_data(config_path, provider=provider, years=(2024,))
    second = build_mvp_data(config_path, provider=provider, years=(2024,))

    assert len(requests) == 2  # one GET and one POST; the next build reuses raw bytes
    assert first.dataset_version == second.dataset_version
    assert first.row_counts == {"1min": 20, "3min": 6, "15min": 1}
    assert first.raw_manifest_path.is_file()
    assert all(path.is_file() for path in first.curated_manifest_paths.values())
    assert first.coverage_report_path.is_file()
    assert first.gap_report_path.is_file()
    raw_manifest = DatasetManifest.model_validate_json(
        first.raw_manifest_path.read_text(encoding="utf-8")
    )
    assert {
        name: raw_manifest.parameters[name]
        for name in (
            "adapter",
            "source_timezone_offset",
            "source_observes_dst",
            "timestamp_semantics",
            "price_side",
            "volume_reliable",
        )
    } == {
        "adapter": "histdata_ascii_m1",
        "source_timezone_offset": "-05:00",
        "source_observes_dst": False,
        "timestamp_semantics": "candle_open",
        "price_side": "bid",
        "volume_reliable": False,
    }
    assert validate_existing_mvp_data(config_path, allow_partial=True) == first.row_counts
    with pytest.raises(DataBuildError, match="partial"):
        validate_existing_mvp_data(config_path)


def test_existing_data_reuse_enforces_active_provider_contract(tmp_path: Path) -> None:
    config_path = _isolated_config(tmp_path)
    provider, _ = _provider()
    build_mvp_data(config_path, provider=provider, years=(2024,))
    instrument_path = config_path.parent / "instrument.yaml"
    instrument = yaml.safe_load(instrument_path.read_text(encoding="utf-8"))
    instrument["provider"]["source_timezone_offset"] = "+00:00"
    instrument_path.write_text(
        yaml.safe_dump(instrument, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(DataBuildError, match=r"active provider.*source_timezone_offset"):
        validate_existing_mvp_data(config_path, allow_partial=True)


def test_existing_data_reuse_rejects_raw_provider_provenance_drift(
    tmp_path: Path,
) -> None:
    config_path = _isolated_config(tmp_path)
    provider, _ = _provider()
    result = build_mvp_data(config_path, provider=provider, years=(2024,))
    manifest = DatasetManifest.model_validate_json(
        result.raw_manifest_path.read_text(encoding="utf-8")
    )
    parameters = dict(manifest.parameters)
    parameters["source_observes_dst"] = True
    _rewrite_raw_manifest(result.raw_manifest_path, parameters=parameters)

    with pytest.raises(
        DataBuildError,
        match=r"provider provenance mismatch: source_observes_dst=True",
    ):
        validate_existing_mvp_data(config_path, allow_partial=True)


def test_existing_data_reuse_requires_exact_raw_to_curated_lineage(
    tmp_path: Path,
) -> None:
    config_path = _isolated_config(tmp_path)
    provider, _ = _provider()
    result = build_mvp_data(config_path, provider=provider, years=(2024,))
    manifest = DatasetManifest.model_validate_json(
        result.raw_manifest_path.read_text(encoding="utf-8")
    )
    _rewrite_raw_manifest(result.raw_manifest_path, outputs=tuple(reversed(manifest.outputs)))

    with pytest.raises(DataBuildError, match="curated 1min lineage"):
        validate_existing_mvp_data(config_path, allow_partial=True)


def test_phase2_build_refuses_holdout_year_before_network_access(tmp_path: Path) -> None:
    config_path = _isolated_config(tmp_path)
    provider, requests = _provider()

    with pytest.raises(DataBuildError, match="holdout"):
        build_mvp_data(config_path, provider=provider, years=(2025,))

    assert requests == []
