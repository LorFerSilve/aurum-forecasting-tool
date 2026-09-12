"""Frozen source-specific hypotheses consumed by the shared Phase-10 machinery.

Profiles are code contracts, not configurable model-selection dimensions. The
dollar observation remains the authentic EURUSD bid close; only its causal
return direction is inverted in the explicitly registered feature builder.
The rate profile uses the published DFII10 percentage level and additive yield
changes rather than treating a yield as a tradable price. The CPI profile uses
the BLS CPI-U NSA All items index and only causally released monthly transforms.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gold_forecasting.phase10.features import SILVER_FEATURE_NAMES, build_silver_features


@dataclass(frozen=True, slots=True)
class ContextProfile:
    protocol: str
    source_id: str
    symbol: str
    title: str

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self.source_id == "silver":
            return SILVER_FEATURE_NAMES
        if self.source_id == "dollar":
            from gold_forecasting.phase10.dollar_features import DOLLAR_FEATURE_NAMES

            return DOLLAR_FEATURE_NAMES
        if self.source_id == "rate":
            from gold_forecasting.phase10.rate_features import RATE_FEATURE_NAMES

            return RATE_FEATURE_NAMES
        if self.source_id == "cpi":
            from gold_forecasting.phase10.cpi_features import CPI_FEATURE_NAMES

            return CPI_FEATURE_NAMES
        raise ValueError(f"unsupported frozen Phase-10 source: {self.source_id}")

    @property
    def model_features(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.feature_names
            if name not in {f"{self.source_id}_is_missing", f"{self.source_id}_is_stale"}
        )

    def build_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.source_id == "silver":
            return build_silver_features(frame)
        if self.source_id == "dollar":
            from gold_forecasting.phase10.dollar_features import build_dollar_features

            return build_dollar_features(frame)
        if self.source_id == "rate":
            from gold_forecasting.phase10.rate_features import build_rate_features

            return build_rate_features(frame)
        if self.source_id == "cpi":
            from gold_forecasting.phase10.cpi_features import build_cpi_features

            return build_cpi_features(frame)
        raise ValueError(f"unsupported frozen Phase-10 source: {self.source_id}")


SILVER_PROFILE = ContextProfile("phase10-silver-modeled-v1", "silver", "XAGUSD", "XAGUSD silver")
DOLLAR_PROFILE = ContextProfile(
    "phase10-dollar-eurusd-modeled-v1", "dollar", "EURUSD", "inverse EURUSD dollar proxy"
)
RATE_PROFILE = ContextProfile(
    "phase10-rate-dfii10-modeled-v1",
    "rate",
    "DFII10",
    "10-year TIPS real-yield proxy",
)
CPI_PROFILE = ContextProfile(
    "phase10-cpi-cuur0000sa0-modeled-v1",
    "cpi",
    "CUUR0000SA0",
    "U.S. CPI-U NSA All items",
)


def profile_for_protocol(protocol: str) -> ContextProfile:
    for profile in (SILVER_PROFILE, DOLLAR_PROFILE, RATE_PROFILE, CPI_PROFILE):
        if profile.protocol == protocol:
            return profile
    raise ValueError(f"unsupported frozen Phase-10 protocol: {protocol}")


def profile_for_source(source_id: str) -> ContextProfile:
    for profile in (SILVER_PROFILE, DOLLAR_PROFILE, RATE_PROFILE, CPI_PROFILE):
        if profile.source_id == source_id:
            return profile
    raise ValueError(f"unsupported frozen Phase-10 source: {source_id}")
