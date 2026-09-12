"""Real-format local fixtures authenticate EURUSD identity and fail closed."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from gold_forecasting.cli import app
from gold_forecasting.phase10 import histdata_context
from gold_forecasting.phase10.bundle import load_context_data
from gold_forecasting.phase10.contracts import ContextSource, require_strict_pit
from gold_forecasting.phase10.histdata_context import (
    ContextHistDataError,
    import_histdata_context_archives,
    parse_histdata_context_archive,
)


def source(**updates: object) -> ContextSource:
    return ContextSource.model_validate({
        "source_id": "dollar",
        "description": "Inverse EURUSD directional dollar proxy; raw bid close",
        "source_url": "https://www.histdata.com/f-a-q/data-files-detailed-specification/",
        "market_hours": "OTC vendor stream",
        "source_timezone": "Fixed UTC-05:00 without daylight saving time",
        "publication_delay_seconds": 60,
        "stale_after_seconds": 600,
        "availability_basis": "modeled_latency",
        "revision_policy": "append_only",
        "availability_evidence": "Modeled close plus 60 seconds; no historical release proof",
        "enabled": True,
        **updates,
    })


def archive(path: Path, *, symbol: str = "EURUSD", rows: list[str] | None = None) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr(
            f"DAT_ASCII_{symbol}_M1_2020.csv",
            "\n".join(rows or [
                "20200102 120000;1.10;1.12;1.09;1.11;0",
                "20200702 120000;1.11;1.13;1.10;1.12;0",
            ]),
        )
    return path


def parse(path: Path, contract: ContextSource | None = None) -> tuple[pd.DataFrame, dict]:
    return parse_histdata_context_archive(
        path, year=2020, source=contract or source(), symbol="EURUSD",
        ingested_at_utc=pd.Timestamp("2026-09-10T00:00:00Z"),
    )


def test_eurusd_keeps_raw_bid_and_fixed_timezone_with_authenticated_identity(
    tmp_path: Path,
) -> None:
    path = archive(tmp_path / "eurusd.zip")
    table, metadata = parse(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert table.value.tolist() == [1.11, 1.12]  # Never manufactures inverse bid/ask quotes.
    assert table.observed_at_utc.tolist() == [
        pd.Timestamp("2020-01-02T17:01Z"), pd.Timestamp("2020-07-02T17:01Z"),
    ]
    assert table.available_at_utc.sub(table.observed_at_utc).eq(pd.Timedelta(seconds=60)).all()
    assert table.raw_sha256.eq(digest).all()
    assert table.revision_id.eq(f"sha256:{digest}").all()
    assert table.observation_id.tolist() == [
        "eurusd-m1-20200102T170100Z", "eurusd-m1-20200702T170100Z",
    ]
    assert metadata["sha256"] == digest
    assert metadata["source_symbol"] == "EURUSD"
    assert metadata["source_observes_dst"] is False
    assert table.source_uri.str.endswith("/eurusd/2020").all()


@pytest.mark.parametrize("symbol", ["XAUUSD", "XAGUSD", "UDXUSD"])
def test_wrong_instrument_zip_cannot_be_renamed_to_dollar(tmp_path: Path, symbol: str) -> None:
    with pytest.raises(ContextHistDataError, match="EURUSD"):
        parse(archive(tmp_path / "eurusd.zip", symbol=symbol))


@pytest.mark.parametrize("updates", [
    {"source_id": "silver"}, {"availability_basis": "provider_timestamp"},
    {"revision_policy": "vintages"}, {"source_url": "https://example.com/"},
])
def test_false_provenance_contract_rejected_before_archive_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, updates: dict,
) -> None:
    def forbid(*args: object, **kwargs: object) -> bytes:
        pytest.fail("contract guard must precede archive access")
    monkeypatch.setattr(histdata_context, "_read_archive", forbid)
    with pytest.raises(ContextHistDataError):
        parse(tmp_path / "not-opened.zip", source(**updates))


def test_holdout_year_rejected_before_archive_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid(*args: object, **kwargs: object) -> bytes:
        pytest.fail("holdout archive must never be opened")
    monkeypatch.setattr(histdata_context, "_read_archive", forbid)
    with pytest.raises(ContextHistDataError, match="2020-2024"):
        parse_histdata_context_archive(
            tmp_path / "2025.zip", year=2025, source=source(), symbol="EURUSD",
        )


@pytest.mark.parametrize("row", [
    "20200102 120001;1.1;1.2;1.0;1.1;0",  # Not a minute open.
    "20210102 120000;1.1;1.2;1.0;1.1;0",  # Wrong source year.
    "20200102 120000;1.1;1.0;1.0;1.1;0",  # Invalid high.
    "20200102 120000;0;1.2;1.0;1.1;0",    # Zero price.
    "20200102 120000;1.1;1.2;1.0;nan;0",  # Nonfinite price.
])
def test_invalid_market_rows_fail_closed(tmp_path: Path, row: str) -> None:
    with pytest.raises(ContextHistDataError):
        parse(archive(tmp_path / "eurusd.zip", rows=[row]))


def test_duplicate_identity_and_conflicting_value_rules(tmp_path: Path) -> None:
    row = "20200102 120000;1.10;1.12;1.09;1.11;0"
    path = archive(tmp_path / "eurusd.zip", rows=[row, row])
    table, metadata = parse(path)
    assert len(table) == 1
    assert metadata["exact_duplicates_removed"] == 1
    archive(path, rows=[row, row.replace(";1.11;", ";1.115;")])
    with pytest.raises(ContextHistDataError, match="conflicting candles"):
        parse(path)


def test_ignored_zip_members_are_bounded_before_crc_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = archive(tmp_path / "eurusd.zip")
    with zipfile.ZipFile(path, "a") as zipped:
        zipped.writestr("ignored.txt", "a" * 1000)
    monkeypatch.setattr(histdata_context, "_MAX_MEMBER_BYTES", 500)
    with pytest.raises(ContextHistDataError, match="budget"):
        parse(path)


def test_dollar_import_bundle_hashes_and_cli(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    archive(raw / "HISTDATA_COM_ASCII_EURUSD_M1_2020.zip")
    output = tmp_path / "context"
    result = CliRunner().invoke(app, [
        "phase10", "dollar-import", "--archive-directory", str(raw),
        "--output", str(output), "--years", "2020",
    ])
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["source_symbol"] == "EURUSD"
    assert summary["strict_pit"] is False
    assert load_context_data(output / "dollar.bundle-set.json", source()).value.tolist() == [
        1.11, 1.12,
    ]
    with pytest.raises(ValueError, match="provider release evidence"):
        require_strict_pit(source())
    part = output / "dollar-2020.parquet"
    part.write_bytes(part.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="SHA-256"):
        load_context_data(output / "dollar.bundle-set.json", source())
    with pytest.raises(FileExistsError):
        import_histdata_context_archives(raw, output, source(), symbol="EURUSD", years=(2020,))


def test_blocked_cli_preflight_uses_dollar_protocol_and_opens_no_run(tmp_path: Path) -> None:
    from gold_forecasting.phase10.config import load_phase10_config

    config = load_phase10_config("configs/phase10_dollar_ablation.yaml")
    configs = tmp_path / "configs"
    configs.mkdir()
    path = configs / "dollar.yaml"
    path.write_text(json.dumps(config.model_dump(mode="json")))
    result = CliRunner().invoke(app, ["phase10", "preflight", "--config", str(path)])
    assert result.exit_code == 1, result.output
    report = json.loads(result.output)
    assert report["protocol"] == "phase10-dollar-eurusd-modeled-v1"
    assert report["holdout_opened"] is False
    assert not (tmp_path / config.output_directory).exists()
