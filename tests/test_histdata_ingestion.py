"""Tests for the public HistData XAUUSD Generic ASCII M1 adapter."""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pandas as pd
import pytest

from gold_forecasting.ingestion import (
    HISTDATA_USER_AGENT,
    HistDataArchiveError,
    HistDataDownloadError,
    HistDataM1Provider,
    HistDataParseError,
    parse_histdata_m1,
)


def _archive_bytes(rows: str, *, period: str = "2024") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"DAT_ASCII_XAUUSD_M1_{period}.csv", rows)
        archive.writestr(f"DAT_ASCII_XAUUSD_M1_{period}.txt", "source status report")
    return buffer.getvalue()


def _download_form(*, token: str = "0123456789abcdef0123456789abcdef") -> str:
    return f"""
    <html><body>
      <form id="file_status" method="post" action="/getStatus.php">
        <input type="hidden" name="tk" value="wrong-token">
      </form>
      <form id="file_down" method="POST" action="/get.php">
        <input type="hidden" name="tk" value="{token}">
        <input type="hidden" name="date" value="2024">
        <input type="hidden" name="datemonth" value="2024">
        <input type="hidden" name="platform" value="ASCII">
        <input type="hidden" name="timeframe" value="M1">
        <input type="hidden" name="fxpair" value="XAUUSD">
      </form>
    </body></html>
    """


def test_download_uses_public_get_post_flow_and_reuses_immutable_zip(tmp_path: Path) -> None:
    payload = _archive_bytes("20240115 100000;2025.0;2026.0;2024.0;2025.5;0\n")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["User-Agent"] == HISTDATA_USER_AGENT
        if request.method == "GET":
            assert request.url.path == "/download-free-forex-historical-data/"
            assert request.url.query.decode() == "/ascii/1-minute-bar-quotes/xauusd/2024"
            return httpx.Response(200, text=_download_form())

        assert request.method == "POST"
        assert request.url.path == "/get.php"
        assert request.headers["Referer"].endswith(
            "?/ascii/1-minute-bar-quotes/xauusd/2024"
        )
        fields = parse_qs(request.content.decode("ascii"))
        assert fields == {
            "tk": ["0123456789abcdef0123456789abcdef"],
            "date": ["2024"],
            "datemonth": ["2024"],
            "platform": ["ASCII"],
            "timeframe": ["M1"],
            "fxpair": ["XAUUSD"],
        }
        return httpx.Response(
            200,
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    observed_at = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
    provider = HistDataM1Provider(client=client, clock=lambda: observed_at)

    first = provider.download_year(2024, tmp_path)
    second = provider.download_year(2024, tmp_path)

    assert len(requests) == 2
    assert first.path == (
        tmp_path
        / "histdata"
        / "XAUUSD"
        / "1min"
        / "HISTDATA_COM_ASCII_XAUUSD_M1_2024.zip"
    ).resolve()
    assert first.path.read_bytes() == payload
    assert first.sha256 == hashlib.sha256(payload).hexdigest()
    assert first.reused is False
    assert second.reused is True
    assert second.sha256 == first.sha256
    assert second.metadata()["volume_reliable"] is False


def test_download_rejects_html_200_without_creating_raw_file(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=_download_form())
        return httpx.Response(200, text="token expired")

    provider = HistDataM1Provider(client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(HistDataArchiveError, match="not a real ZIP"):
        provider.download_year(2024, tmp_path)

    assert list(tmp_path.rglob("*.zip")) == []


def test_download_requires_the_specific_public_download_form(tmp_path: Path) -> None:
    html = _download_form().replace('id="file_down"', 'id="not_the_download_form"')
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=html))
    )

    with pytest.raises(HistDataDownloadError, match="download token/form"):
        HistDataM1Provider(client=client).download_year(2024, tmp_path)


def test_parse_uses_fixed_utc_minus_five_and_exposes_source_metadata() -> None:
    payload = _archive_bytes(
        "20240115 100000;2025.0;2026.0;2024.0;2025.5;0\n"
        "20240715 100000;2400.0;2401.0;2399.0;2400.5;0\n"
        "20240715 100000;2400.0;2401.0;2399.0;2400.5;0\n"
    )
    ingested_at = datetime(2025, 1, 1, tzinfo=UTC)

    candles = parse_histdata_m1(payload, ingested_at_utc=ingested_at)

    assert candles["timestamp_open_utc"].tolist() == [
        pd.Timestamp("2024-01-15T15:00:00Z"),
        pd.Timestamp("2024-07-15T15:00:00Z"),
    ]
    assert candles["timestamp_close_utc"].tolist() == [
        pd.Timestamp("2024-01-15T15:01:00Z"),
        pd.Timestamp("2024-07-15T15:01:00Z"),
    ]
    assert candles.loc[0, ["bid_open", "bid_high", "bid_low", "bid_close"]].tolist() == [
        2025.0,
        2026.0,
        2024.0,
        2025.5,
    ]
    assert candles["is_complete"].eq(True).all()
    assert candles["raw_file_hash"].eq(hashlib.sha256(payload).hexdigest()).all()
    assert candles["dataset_version"].str.startswith("sha256:").all()
    assert candles.attrs["source_timezone_offset"] == "-05:00"
    assert candles.attrs["source_observes_dst"] is False
    assert candles.attrs["volume_reliable"] is False
    assert candles.attrs["exact_duplicates_removed"] == 1


def test_parse_returns_only_candles_closed_at_the_cutoff() -> None:
    payload = _archive_bytes(
        "20240115 100000;2025;2026;2024;2025.5;0\n"
        "20240115 100100;2025.5;2027;2025;2026;0\n"
    )

    candles = parse_histdata_m1(
        payload,
        ingested_at_utc=datetime(2024, 1, 15, 15, 1, 30, tzinfo=UTC),
    )

    assert candles["timestamp_open_utc"].tolist() == [pd.Timestamp("2024-01-15T15:00:00Z")]
    assert candles["is_complete"].tolist() == [True]


def test_parse_rejects_conflicting_candles_for_one_timestamp() -> None:
    payload = _archive_bytes(
        "20240115 100000;2025;2026;2024;2025.5;0\n"
        "20240115 100000;2025;2027;2024;2026.0;0\n"
    )

    with pytest.raises(HistDataParseError, match="conflicting candles"):
        parse_histdata_m1(payload, ingested_at_utc=datetime(2025, 1, 1, tzinfo=UTC))


def test_parse_rejects_zip_without_expected_histdata_member() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("unrelated.csv", "20240115 100000;1;1;1;1;0\n")

    with pytest.raises(HistDataArchiveError, match="DAT_ASCII_XAUUSD"):
        parse_histdata_m1(buffer.getvalue())


def test_parse_rejects_wrong_headerless_field_count() -> None:
    payload = _archive_bytes("20240115 100000;2025;2026;2024;2025.5\n")

    with pytest.raises(HistDataParseError, match="expected 6 headerless fields"):
        parse_histdata_m1(payload)


def test_archive_record_detects_later_raw_file_mutation(tmp_path: Path) -> None:
    original = _archive_bytes("20240115 100000;2025;2026;2024;2025.5;0\n")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=_download_form())
        return httpx.Response(200, content=original)

    provider = HistDataM1Provider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    archive = provider.download_year(2024, tmp_path)
    archive.path.write_bytes(_archive_bytes("20240115 100000;1;1;1;1;0\n"))

    with pytest.raises(HistDataArchiveError, match="no longer matches"):
        provider.parse_archive(archive)
