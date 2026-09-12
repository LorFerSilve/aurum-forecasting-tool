"""Safe acquisition of the official BLS CPI flat file used by Phase 10.

This module deliberately handles transport and local persistence only. Full BLS
schema, series, development-scope and release-calendar validation remains the
responsibility of :mod:`gold_forecasting.phase10.cpi_bls` during import.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import httpx

from gold_forecasting.phase10.cpi_bls import (
    BLS_CPI_FILENAME,
    BLS_CPI_SERIES,
    BLS_CPI_SOURCE_URL,
)

MAX_BLS_CPI_SOURCE_BYTES: Final = 16 * 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS: Final = 30.0
_USER_AGENT: Final = "aurum-forecasting-tool/phase10-cpi-research"
_EXPECTED_HEADER: Final = (
    "series_id",
    "year",
    "period",
    "value",
    "footnote_codes",
)


class CpiAcquisitionError(ValueError):
    """Raised when the official CPI source cannot be acquired safely."""


def _prepare_destination(destination: str | Path, *, overwrite: bool) -> Path:
    path = Path(destination).expanduser()
    if path.name != BLS_CPI_FILENAME:
        raise CpiAcquisitionError(
            f"BLS CPI destination must be named exactly {BLS_CPI_FILENAME}"
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CpiAcquisitionError(f"cannot create CPI source directory: {path.parent}") from exc
    if path.is_symlink():
        raise CpiAcquisitionError("BLS CPI destination must not be a symlink")
    if path.exists():
        if not path.is_file():
            raise CpiAcquisitionError("BLS CPI destination must be a regular file")
        if not overwrite:
            raise CpiAcquisitionError(
                "BLS CPI destination already exists; pass overwrite=True explicitly to replace it"
            )
    return path


def _validate_content_length(headers: httpx.Headers) -> None:
    raw = headers.get("content-length")
    if raw is None:
        return
    try:
        size = int(raw)
    except ValueError as exc:
        raise CpiAcquisitionError("BLS response has an invalid Content-Length") from exc
    if size <= 0 or size > MAX_BLS_CPI_SOURCE_BYTES:
        raise CpiAcquisitionError("BLS response has an invalid or excessive Content-Length")


def _validate_transport_payload(payload: bytes) -> None:
    if not payload:
        raise CpiAcquisitionError("BLS CPI download is empty")
    try:
        first_line = payload.split(b"\n", maxsplit=1)[0].decode("utf-8-sig").rstrip("\r")
    except UnicodeError as exc:
        raise CpiAcquisitionError("BLS CPI download is not valid UTF-8 text") from exc
    columns = tuple(part.strip() for part in first_line.split("\t"))
    if columns != _EXPECTED_HEADER:
        raise CpiAcquisitionError("BLS CPI download does not have the canonical BLS header")
    if BLS_CPI_SERIES.encode("ascii") not in payload:
        raise CpiAcquisitionError(f"BLS CPI download does not contain series {BLS_CPI_SERIES}")


def _persist_atomic(destination: Path, payload: bytes, *, overwrite: bool) -> None:
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{BLS_CPI_FILENAME}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        if overwrite:
            if destination.is_symlink():
                raise CpiAcquisitionError("BLS CPI destination became a symlink during download")
            os.replace(temporary, destination)
            temporary = None
            return

        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise CpiAcquisitionError(
                "BLS CPI destination appeared during download; refusing to overwrite it"
            ) from exc
        temporary.unlink()
        temporary = None
    except OSError as exc:
        raise CpiAcquisitionError(f"cannot persist BLS CPI source: {destination}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def download_bls_cpi(
    destination: str | Path = Path("data/raw/phase10/cpi") / BLS_CPI_FILENAME,
    *,
    overwrite: bool = False,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, object]:
    """Download the fixed official BLS CPI source to a local authenticated input path.

    Redirects are intentionally disabled so a successful acquisition must come from
    the exact frozen BLS URL. The payload is bounded in memory, receives lightweight
    transport-shape validation and is then persisted atomically. The importer performs
    the authoritative schema/content validation afterwards.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise CpiAcquisitionError("timeout_seconds must be finite and positive")
    path = _prepare_destination(destination, overwrite=overwrite)
    payload = bytearray()
    try:
        with (
            httpx.Client(
                follow_redirects=False,
                headers={"User-Agent": _USER_AGENT},
                timeout=timeout_seconds,
                transport=transport,
            ) as client,
            client.stream("GET", BLS_CPI_SOURCE_URL) as response,
        ):
            if response.status_code != httpx.codes.OK:
                raise CpiAcquisitionError(
                    f"BLS CPI download returned HTTP {response.status_code}; "
                    "redirects are rejected"
                )
            if str(response.url) != BLS_CPI_SOURCE_URL:
                raise CpiAcquisitionError(
                    "BLS CPI response URL differs from the frozen source URL"
                )
            _validate_content_length(response.headers)
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                if len(payload) + len(chunk) > MAX_BLS_CPI_SOURCE_BYTES:
                    raise CpiAcquisitionError(
                        "BLS CPI download exceeds the configured size limit"
                    )
                payload.extend(chunk)
    except CpiAcquisitionError:
        raise
    except httpx.HTTPError as exc:
        raise CpiAcquisitionError("BLS CPI download failed") from exc

    data = bytes(payload)
    _validate_transport_payload(data)
    digest = hashlib.sha256(data).hexdigest()
    _persist_atomic(path, data, overwrite=overwrite)
    return {
        "source_url": BLS_CPI_SOURCE_URL,
        "destination": str(path.resolve()),
        "filename": BLS_CPI_FILENAME,
        "size_bytes": len(data),
        "sha256": digest,
        "acquired_at_utc": datetime.now(UTC).isoformat(),
        "overwrite": overwrite,
    }


__all__ = [
    "MAX_BLS_CPI_SOURCE_BYTES",
    "CpiAcquisitionError",
    "download_bls_cpi",
]
