"""Backward-compatible XAGUSD entrypoints for the shared HistData context adapter.

Silver observation identity, metadata, partition filenames and values are unchanged.
The frozen XAUUSD ingestion adapter is independent of this Phase-10 implementation.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from gold_forecasting.phase10.contracts import ContextSource
from gold_forecasting.phase10.histdata_context import (
    ContextHistDataError,
    import_histdata_context_archives,
    parse_histdata_context_archive,
)

HISTDATA_SILVER_SYMBOL = "XAGUSD"
HISTDATA_SILVER_SOURCE = "histdata"
HISTDATA_SILVER_TIMEFRAME = "1min"
HISTDATA_SILVER_SOURCE_OFFSET = "-05:00"
SilverHistDataError = ContextHistDataError


def parse_histdata_xagusd_archive(
    archive_path: str | Path,
    *,
    year: int,
    source: ContextSource,
    ingested_at_utc: datetime | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Parse an XAGUSD archive with the original silver identity contract."""
    return parse_histdata_context_archive(
        archive_path, year=year, source=source, symbol=HISTDATA_SILVER_SYMBOL,
        ingested_at_utc=ingested_at_utc,
    )


def import_histdata_silver_archives(
    archive_directory: str | Path,
    output_directory: str | Path,
    source: ContextSource,
    *,
    years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024),
    ingested_at_utc: datetime | pd.Timestamp | None = None,
) -> dict[str, object]:
    """Import silver through the shared parser without changing bundle semantics."""
    return import_histdata_context_archives(
        archive_directory, output_directory, source, symbol=HISTDATA_SILVER_SYMBOL,
        years=years, ingested_at_utc=ingested_at_utc,
    )


__all__ = [
    "HISTDATA_SILVER_SOURCE_OFFSET",
    "HISTDATA_SILVER_SYMBOL",
    "SilverHistDataError",
    "import_histdata_silver_archives",
    "parse_histdata_xagusd_archive",
]
