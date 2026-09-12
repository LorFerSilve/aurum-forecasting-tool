"""Provider-neutral scalar observations with explicit release and vintage evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

OBSERVATION_COLUMNS = (
    "source_id", "observation_id", "observed_at_utc", "available_at_utc",
    "ingested_at_utc", "value", "revision_id", "source_uri", "raw_sha256",
)
TIME_COLUMNS = ("observed_at_utc", "available_at_utc", "ingested_at_utc")
DEVELOPMENT_START = pd.Timestamp("2020-01-01", tz="UTC")
DEVELOPMENT_END = pd.Timestamp("2025-01-01", tz="UTC")


class ContextSource(BaseModel):
    """Declared evidence is audited separately; modeled latency is never strict PIT."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    market_hours: str = Field(min_length=1)
    source_timezone: str = Field(min_length=1)
    publication_delay_seconds: int = Field(ge=0, strict=True)
    stale_after_seconds: int = Field(gt=0, strict=True)
    availability_basis: Literal["provider_timestamp", "modeled_latency", "synthetic"]
    revision_policy: Literal["append_only", "vintages"]
    availability_evidence: str = Field(min_length=1)
    missing_policy: Literal["price_only_fallback"] = "price_only_fallback"
    enabled: bool = Field(default=False, strict=True)


class ContextProvider(Protocol):
    """Adapters return the same validated table regardless of the upstream provider."""

    def load_observations(self) -> pd.DataFrame: ...


def utc_series(values: pd.Series, name: str) -> pd.Series:
    """Reject implicit localization, nulls and non-UTC zones at the core boundary."""
    if (
        not isinstance(values.dtype, pd.DatetimeTZDtype)
        or str(values.dtype.tz) != "UTC"
        or values.isna().any()
    ):
        raise ValueError(f"{name} must be non-null timezone-aware UTC timestamps")
    return values.astype("datetime64[ns, UTC]")


def validate_observations(frame: pd.DataFrame, source: ContextSource) -> pd.DataFrame:
    """Validate without rewriting release times or discarding older vintages.

    Observation time means candle close for market bars and the reference-period
    timestamp for macro data. Availability is the individual version's release,
    never the calendar date to which a revised macro value relates.
    """
    if not frame.columns.is_unique:
        raise ValueError("observation columns must be unique")
    missing = set(OBSERVATION_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"missing observation columns: {sorted(missing)}")
    result = frame.loc[:, list(OBSERVATION_COLUMNS)].copy().reset_index(drop=True)
    for name in TIME_COLUMNS:
        result[name] = utc_series(result[name], name)
    for name in ("source_id", "observation_id", "revision_id", "source_uri", "raw_sha256"):
        if not result[name].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError(f"{name} must contain nonempty strings")
    if not result["source_id"].eq(source.source_id).all():
        raise ValueError("observation source_id differs from source contract")
    if not result["raw_sha256"].str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("raw_sha256 must identify the original source artifact")
    if (
        not pd.api.types.is_numeric_dtype(result["value"])
        or pd.api.types.is_bool_dtype(result["value"])
        or np.iscomplexobj(result["value"].to_numpy())
    ):
        raise ValueError("value must be numeric (null is an explicit missing observation)")
    result["value"] = result["value"].astype("float64")
    if np.isinf(result["value"].to_numpy()).any():
        raise ValueError("observation value must not be infinite")
    earliest = result["observed_at_utc"] + pd.to_timedelta(
        source.publication_delay_seconds, unit="s"
    )
    if (result["available_at_utc"] < earliest).any():
        raise ValueError("availability precedes observation close or declared publication delay")
    if (result["ingested_at_utc"] < result["available_at_utc"]).any():
        raise ValueError("ingestion precedes claimed availability")
    if (
        (result["observed_at_utc"] < DEVELOPMENT_START)
        | (result["observed_at_utc"] >= DEVELOPMENT_END)
    ).any():
        raise ValueError("context observations must remain in development 2020-2024")
    if result.duplicated(["observed_at_utc", "available_at_utc"]).any():
        raise ValueError("ambiguous versions at the same observation and availability time")
    if result.duplicated(["observation_id", "revision_id"]).any():
        raise ValueError("duplicate observation version identity")
    if (
        result.groupby("observation_id")["observed_at_utc"].nunique().gt(1).any()
        or result.groupby("observed_at_utc")["observation_id"].nunique().gt(1).any()
    ):
        raise ValueError("observation identity must map one-to-one to observation time")
    if source.revision_policy == "append_only" and result["observed_at_utc"].duplicated().any():
        raise ValueError("revisions require an explicit vintages contract")
    return result.sort_values(["available_at_utc", "observed_at_utc"]).reset_index(drop=True)


def require_strict_pit(source: ContextSource) -> None:
    """Prevent hypothetical historical availability from crossing the formal gate."""
    if source.availability_basis != "provider_timestamp":
        raise ValueError("strict point-in-time admission requires provider release evidence")


def load_source(path: str | Path) -> ContextSource:
    from gold_forecasting.config import _load_yaml_mapping

    return ContextSource.model_validate(_load_yaml_mapping(Path(path)))
