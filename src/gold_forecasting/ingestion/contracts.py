"""Provider-neutral contracts and registry for annual M1 candle adapters."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

from gold_forecasting.config import ProviderConfig


class ProviderConfigurationError(ValueError):
    """Raised when no safe adapter matches the configured provider."""


class RawArtifact(Protocol):
    """Minimum raw-artifact provenance exposed by every M1 adapter."""

    @property
    def path(self) -> Path: ...

    @property
    def sha256(self) -> str: ...

    @property
    def size_bytes(self) -> int: ...

    @property
    def year(self) -> int: ...

    @property
    def observed_at_utc(self) -> datetime: ...

    @property
    def source(self) -> str: ...

    @property
    def source_symbol(self) -> str: ...

    @property
    def timeframe(self) -> str: ...

    @property
    def source_page_url(self) -> str: ...


class M1IngestionBatch(Protocol):
    """Provider-neutral result consumed by the data pipeline."""

    @property
    def archive(self) -> RawArtifact: ...

    @property
    def candles(self) -> pd.DataFrame: ...


class M1Provider(Protocol):
    """Strict adapter contract for one immutable annual M1 acquisition."""

    def ingest_year(
        self,
        year: int,
        raw_directory: str | Path,
        *,
        ingested_at_utc: datetime | None = None,
        as_of_utc: datetime | None = None,
    ) -> M1IngestionBatch: ...


ProviderFactory = Callable[[ProviderConfig], M1Provider]


def _histdata_factory(config: ProviderConfig) -> M1Provider:
    if config.id != "histdata":
        raise ProviderConfigurationError(
            "adapter 'histdata_ascii_m1' requires provider.id='histdata'"
        )
    if config.source_symbol != "XAUUSD":
        raise ProviderConfigurationError(
            "adapter 'histdata_ascii_m1' requires provider.source_symbol='XAUUSD'"
        )
    fixed_contract = {
        "source_timezone_offset": "-05:00",
        "source_observes_dst": False,
        "timestamp_semantics": "candle_open",
        "price_side": "bid",
        "volume_reliable": False,
    }
    for field_name, expected in fixed_contract.items():
        if getattr(config, field_name) != expected:
            raise ProviderConfigurationError(
                "adapter 'histdata_ascii_m1' requires "
                f"provider.{field_name}={expected!r}"
            )
    parsed = urlsplit(str(config.source_url))
    if parsed.scheme != "https" or parsed.hostname not in {
        "histdata.com",
        "www.histdata.com",
    }:
        raise ProviderConfigurationError(
            "adapter 'histdata_ascii_m1' requires an official HTTPS histdata.com source_url"
        )
    base_url = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))

    # Import lazily so provider modules can themselves import these contracts.
    from gold_forecasting.ingestion.histdata import HistDataM1Provider

    return HistDataM1Provider(base_url=base_url)


_PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "histdata_ascii_m1": _histdata_factory,
}


def register_m1_provider(adapter: str, factory: ProviderFactory) -> None:
    """Register a new adapter without changing downstream pipeline code."""

    normalized = adapter.strip()
    if not normalized:
        raise ProviderConfigurationError("provider adapter name must not be empty")
    if normalized in _PROVIDER_FACTORIES:
        raise ProviderConfigurationError(f"provider adapter is already registered: {normalized}")
    _PROVIDER_FACTORIES[normalized] = factory


def create_m1_provider(config: ProviderConfig) -> M1Provider:
    """Instantiate the adapter selected by the versioned provider config."""

    try:
        factory = _PROVIDER_FACTORIES[config.adapter]
    except KeyError as exc:
        available = ", ".join(sorted(_PROVIDER_FACTORIES))
        raise ProviderConfigurationError(
            f"unknown M1 provider adapter {config.adapter!r}; registered adapters: {available}"
        ) from exc
    return factory(config)


def registered_m1_adapters() -> tuple[str, ...]:
    """Return stable adapter identifiers for diagnostics and tests."""

    return tuple(sorted(_PROVIDER_FACTORIES))


__all__ = [
    "M1IngestionBatch",
    "M1Provider",
    "ProviderConfigurationError",
    "ProviderFactory",
    "RawArtifact",
    "create_m1_provider",
    "register_m1_provider",
    "registered_m1_adapters",
]
