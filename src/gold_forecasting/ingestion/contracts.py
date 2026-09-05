"""Provider-neutral contracts and registry for annual M1 candle adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable
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
    def ingested_at_utc(self) -> datetime: ...

    @property
    def provider_revision(self) -> str: ...

    @property
    def reused(self) -> bool: ...

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


PaginationMode = Literal["none", "cursor", "page_number"]
ResumeMode = Literal["completed_resource_cache", "http_range", "cursor"]
RevisionMode = Literal["provider_metadata", "content_sha256"]


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Machine-readable acquisition behaviour exposed by an adapter.

    ``pagination_mode='none'`` is an explicit capability value: it prevents a
    caller from pretending that a provider with discrete annual archives has
    pages.  ``available_fields`` likewise distinguishes absent ask/spread/tick
    data from fields that merely happen to be empty in one response.
    """

    annual_immutable_resources: bool
    pagination_mode: PaginationMode
    resume_mode: ResumeMode
    revision_mode: RevisionMode
    available_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.pagination_mode not in {"none", "cursor", "page_number"}:
            raise ValueError("unsupported pagination_mode")
        if self.resume_mode not in {"completed_resource_cache", "http_range", "cursor"}:
            raise ValueError("unsupported resume_mode")
        if self.revision_mode not in {"provider_metadata", "content_sha256"}:
            raise ValueError("unsupported revision_mode")
        if not self.available_fields:
            raise ValueError("available_fields must describe at least one source field")
        normalized = tuple(field.strip() for field in self.available_fields)
        if any(not field for field in normalized):
            raise ValueError("available_fields must contain non-empty names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("available_fields must not contain duplicates")


@dataclass(frozen=True, slots=True)
class M1Resource:
    """One provider resource covering a half-open UTC interval."""

    resource_id: str
    period_start_utc: datetime
    period_end_utc: datetime
    immutable: bool
    partition: str

    def __post_init__(self) -> None:
        if not self.resource_id.strip():
            raise ValueError("resource_id must not be empty")
        if not self.partition.strip():
            raise ValueError("partition must not be empty")
        for name in ("period_start_utc", "period_end_utc"):
            value = getattr(self, name)
            if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
                raise ValueError(f"{name} must be timezone-aware UTC")
        if self.period_start_utc >= self.period_end_utc:
            raise ValueError("resource period must be non-empty")


@dataclass(frozen=True, slots=True)
class M1ResourcePage:
    """One deterministic page from a provider resource catalogue."""

    resources: tuple[M1Resource, ...]
    next_cursor: str | None = None

    def __post_init__(self) -> None:
        if self.next_cursor is not None and not self.next_cursor.strip():
            raise ValueError("next_cursor must be non-empty when supplied")
        resource_ids = [resource.resource_id for resource in self.resources]
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("one resource page must not contain duplicate resource ids")


@runtime_checkable
class IncrementalM1Provider(M1Provider, Protocol):
    """Extended contract used by the idempotent incremental updater."""

    @property
    def capabilities(self) -> ProviderCapabilities: ...

    def list_resource_page(
        self,
        *,
        period_start_utc: datetime,
        period_end_utc: datetime,
        cursor: str | None = None,
    ) -> M1ResourcePage: ...

    def ingest_resource(
        self,
        resource: M1Resource,
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
                f"adapter 'histdata_ascii_m1' requires provider.{field_name}={expected!r}"
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
    "IncrementalM1Provider",
    "M1IngestionBatch",
    "M1Provider",
    "M1Resource",
    "M1ResourcePage",
    "PaginationMode",
    "ProviderCapabilities",
    "ProviderConfigurationError",
    "ProviderFactory",
    "RawArtifact",
    "ResumeMode",
    "RevisionMode",
    "create_m1_provider",
    "register_m1_provider",
    "registered_m1_adapters",
]
