"""Causal USD-strength features from the explicitly chosen inverse-EURUSD proxy."""

from __future__ import annotations

import pandas as pd

from gold_forecasting.phase10.paired_features import paired_price_features

DOLLAR_FEATURE_NAMES = (
    "dollar_return_1_bps",
    "dollar_momentum_5_bps",
    "gold_eur_return_1_bps",
    "gold_dollar_correlation_20",
    "dollar_is_missing",
    "dollar_is_stale",
    "dollar_age_seconds",
)
DOLLAR_MODEL_FEATURES = tuple(
    name for name in DOLLAR_FEATURE_NAMES if name not in {"dollar_is_missing", "dollar_is_stale"}
)


def build_dollar_features(
    frame: pd.DataFrame,
    *,
    cadence_seconds: int = 180,
) -> pd.DataFrame:
    """Append the frozen five-feature EURUSD dollar-proxy contract.

    ``dollar_value`` retains the raw EURUSD bid close in USD per EUR. Negated
    log returns and momentum therefore mean USD strengthening when positive.
    The gold interaction is the log change of XAUUSD / EURUSD, an indicative
    gold-in-EUR change from the available bid observations. It is not an
    executable cross quote. No inverse bid/ask or DXY index level is invented.

    All windows use the shared Phase-10 cadence and observed-time rules.
    Missing, stale, repeated or discontinuous quotes cannot manufacture
    returns; their original audit columns and every gold sample remain intact.
    """
    values = paired_price_features(frame, source_id="dollar", cadence_seconds=cadence_seconds)
    output = frame.copy(deep=True)
    output["gold_return_1_bps"] = values["gold_return_1_bps"]
    output["gold_momentum_5_bps"] = values["gold_momentum_5_bps"]
    output["dollar_return_1_bps"] = -values["context_return_1_bps"]
    output["dollar_momentum_5_bps"] = -values["context_momentum_5_bps"]
    output["gold_eur_return_1_bps"] = (
        values["gold_return_1_bps"] - values["context_return_1_bps"]
    )
    output["gold_dollar_correlation_20"] = -values["gold_context_correlation_20"]
    return output
