"""Safe acquisition guards for the official Phase-10 BLS CPI source."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pandas as pd
import pytest

from gold_forecasting.phase10.contracts import ContextSource, load_source
from gold_forecasting.phase10.cpi_acquisition import (
    MAX_BLS_CPI_SOURCE_BYTES,
    CpiAcquisitionError,
    download_bls_cpi,
)
from gold_forecasting.phase10.cpi_bls import (
    BLS_CPI_FILENAME,
    BLS_CPI_SERIES,
    BLS_CPI_SOURCE_URL,
    parse_bls_cpi_file,
)


@pytest.fixture
def cpi_source() -> ContextSource:
    return load_source("configs/phase10_cpi_exploratory.yaml")


def _payload() -> bytes:
    rows = ["series_id\tyear\tperiod\tvalue\tfootnote_codes"]
    index = 0
    for year in range(2020, 2025):
        for month in range(1, 13):
            rows.append(
                f"{BLS_CPI_SERIES}\t{year}\tM{month:02d}\t{250 + 0.4 * index:.3f}\t"
            )
            index += 1
    return ("\n".join(rows) + "\n").encode()


def _transport(
    payload: bytes,
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == BLS_CPI_SOURCE_URL
        return httpx.Response(
            status_code,
            headers=headers,
            content=payload,
            request=request,
        )

    return httpx.MockTransport(handler)


def test_cpi_acquisition_downloads_exact_source_and_remains_importable(
    tmp_path: Path, cpi_source: ContextSource
) -> None:
    payload = _payload()
    destination = tmp_path / "raw" / BLS_CPI_FILENAME
    result = download_bls_cpi(destination, transport=_transport(payload))

    assert destination.read_bytes() == payload
    assert result["source_url"] == BLS_CPI_SOURCE_URL
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["size_bytes"] == len(payload)

    observations, metadata = parse_bls_cpi_file(
        destination,
        source=cpi_source,
        ingested_at_utc=pd.Timestamp("2026-09-13T00:00:00Z"),
    )
    assert len(observations) == 59
    assert metadata["sha256"] == result["sha256"]


def test_cpi_acquisition_refuses_implicit_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / BLS_CPI_FILENAME
    destination.write_bytes(b"existing")
    with pytest.raises(CpiAcquisitionError, match="already exists"):
        download_bls_cpi(destination, transport=_transport(_payload()))
    assert destination.read_bytes() == b"existing"

    result = download_bls_cpi(
        destination,
        overwrite=True,
        transport=_transport(_payload()),
    )
    assert result["overwrite"] is True
    assert destination.read_bytes() == _payload()


def test_cpi_acquisition_rejects_redirects_and_oversized_responses(tmp_path: Path) -> None:
    redirected = tmp_path / "redirect" / BLS_CPI_FILENAME
    with pytest.raises(CpiAcquisitionError, match="HTTP 302"):
        download_bls_cpi(
            redirected,
            transport=_transport(
                b"",
                status_code=302,
                headers={"location": "https://example.invalid/cpi"},
            ),
        )
    assert not redirected.exists()

    oversized = tmp_path / "oversized" / BLS_CPI_FILENAME
    with pytest.raises(CpiAcquisitionError, match="Content-Length"):
        download_bls_cpi(
            oversized,
            transport=_transport(
                _payload(),
                headers={"content-length": str(MAX_BLS_CPI_SOURCE_BYTES + 1)},
            ),
        )
    assert not oversized.exists()


def test_cpi_acquisition_rejects_noncanonical_transport_payload(tmp_path: Path) -> None:
    destination = tmp_path / BLS_CPI_FILENAME
    with pytest.raises(CpiAcquisitionError, match="canonical BLS header"):
        download_bls_cpi(
            destination,
            transport=_transport(b"wrong\theader\nCUUR0000SA0\n"),
        )
    assert not destination.exists()
    assert list(tmp_path.glob(f".{BLS_CPI_FILENAME}.*.tmp")) == []


def test_cpi_acquisition_requires_exact_filename_and_positive_timeout(tmp_path: Path) -> None:
    with pytest.raises(CpiAcquisitionError, match="named exactly"):
        download_bls_cpi(tmp_path / "renamed.tsv", transport=_transport(_payload()))
    with pytest.raises(CpiAcquisitionError, match="finite and positive"):
        download_bls_cpi(
            tmp_path / BLS_CPI_FILENAME,
            timeout_seconds=0.0,
            transport=_transport(_payload()),
        )
