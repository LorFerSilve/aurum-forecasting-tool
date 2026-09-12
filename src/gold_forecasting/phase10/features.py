"""Causal, gap-aware price and silver features for the Phase 10 ablation."""

from __future__ import annotations

import pandas as pd

from gold_forecasting.phase10.paired_features import paired_price_features

PRICE_FEATURE_NAMES = ("gold_return_1_bps", "gold_momentum_5_bps")
SILVER_FEATURE_NAMES = (
    "silver_return_1_bps",
    "silver_momentum_5_bps",
    "gold_silver_log_ratio",
    "gold_silver_correlation_20",
    "silver_is_missing",
    "silver_is_stale",
    "silver_age_seconds",
)


def build_silver_features(
    frame: pd.DataFrame,
    *,
    cadence_seconds: int = 180,
    momentum_steps: int = 5,
    correlation_steps: int = 20,
) -> pd.DataFrame:
    """Append features while retaining every input row, index and audit column.

    Input prices must already be point-in-time joined to each prediction cutoff.
    Price changes are log returns in basis points, consistent with Phase 7.
    The fixed feature names require five-step momentum and twenty-step Pearson
    correlation. A momentum needs six consecutive observations; correlation
    needs twenty paired one-step returns, hence twenty-one observations. Any
    cadence gap restarts history. Missing or stale silver never contributes to
    derived silver features, and no value is filled or interpolated.
    """
    values = paired_price_features(
        frame,
        source_id="silver",
        cadence_seconds=cadence_seconds,
        momentum_steps=momentum_steps,
        correlation_steps=correlation_steps,
    )
    output = frame.copy(deep=True)
    for name, value in values.items():
        output[name.replace("context", "silver")] = value
    return output
