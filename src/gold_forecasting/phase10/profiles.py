"""Frozen source-specific hypotheses consumed by the shared Phase-10 machinery.

Profiles are code contracts, not configurable model-selection dimensions. The
dollar observation remains the authentic EURUSD bid close; only its causal
return direction is inverted in the explicitly registered feature builder.
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
        from gold_forecasting.phase10.dollar_features import DOLLAR_FEATURE_NAMES

        return DOLLAR_FEATURE_NAMES

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
        from gold_forecasting.phase10.dollar_features import build_dollar_features

        return build_dollar_features(frame)


SILVER_PROFILE = ContextProfile("phase10-silver-modeled-v1", "silver", "XAGUSD", "XAGUSD silver")
DOLLAR_PROFILE = ContextProfile(
    "phase10-dollar-eurusd-modeled-v1", "dollar", "EURUSD", "inverse EURUSD dollar proxy"
)


def profile_for_protocol(protocol: str) -> ContextProfile:
    for profile in (SILVER_PROFILE, DOLLAR_PROFILE):
        if profile.protocol == protocol:
            return profile
    raise ValueError(f"unsupported frozen Phase-10 protocol: {protocol}")


def profile_for_source(source_id: str) -> ContextProfile:
    for profile in (SILVER_PROFILE, DOLLAR_PROFILE):
        if profile.source_id == source_id:
            return profile
    raise ValueError(f"unsupported frozen Phase-10 source: {source_id}")
