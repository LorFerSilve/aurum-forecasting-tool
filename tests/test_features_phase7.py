from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gold_forecasting.config import load_project_config
from gold_forecasting.features.phase7 import (
    Phase7FeatureConfig,
    build_phase7_feature_row,
    build_phase7_features,
    load_phase7_feature_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MINUTES = {
    "1min": 1,
    "3min": 3,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1h": 60,
    "3h": 180,
}


def _candles(timeframe: str, *, hours: int = 30) -> pd.DataFrame:
    minutes = _MINUTES[timeframe]
    periods = max(8, hours * 60 // minutes)
    opens = pd.date_range("2024-01-02T00:00:00Z", periods=periods, freq=f"{minutes}min")
    row = np.arange(periods, dtype=np.float64)
    bid_open = pd.Series(2_000.0 + row * 0.05 + np.sin(row / 7.0) * 0.2)
    bid_close = bid_open + np.sin(row / 3.0) * 0.08
    return pd.DataFrame(
        {
            "timestamp_open_utc": opens,
            "timestamp_close_utc": opens + pd.Timedelta(minutes=minutes),
            "bid_open": bid_open,
            "bid_high": np.maximum(bid_open, bid_close) + 0.10,
            "bid_low": np.minimum(bid_open, bid_close) - 0.10,
            "bid_close": bid_close,
            "instrument": "XAU_USD",
            "timeframe": timeframe,
            "is_complete": True,
            "source": "fixture",
            "raw_file_hash": "a" * 64,
            "ingested_at_utc": pd.Timestamp("2026-01-01T00:00:00Z"),
            "dataset_version": "sha256:" + "b" * 64,
        }
    )


def _inputs() -> dict[str, pd.DataFrame]:
    return {timeframe: _candles(timeframe) for timeframe in _MINUTES}


def _mvp_config():
    return load_project_config(PROJECT_ROOT / "configs" / "mvp.yaml").features


def _phase7_config() -> Phase7FeatureConfig:
    return Phase7FeatureConfig(
        timeframe_windows={
            "1min": (2, 3),
            "3min": (2, 3),
            "5min": (2, 3),
            "15min": (2, 3),
            "30min": (2, 3),
            "1h": (2, 3),
            "3h": (2, 3),
        },
        return_lags=(1, 2),
        regime_lookback_candles=20,
    )


def test_repository_phase7_feature_config_is_valid_and_microstructure_is_disabled() -> None:
    config = load_phase7_feature_config(PROJECT_ROOT / "configs" / "features_phase7.yaml")
    assert config.anchor_timeframe == "3min"
    assert tuple(config.timeframe_windows) == (
        "1min",
        "3min",
        "5min",
        "15min",
        "30min",
        "1h",
        "3h",
    )
    assert config.microstructure_enabled is False
    with pytest.raises(ValueError):
        Phase7FeatureConfig(
            timeframe_windows=config.timeframe_windows,
            microstructure_enabled=True,  # type: ignore[arg-type]
        )


def test_phase7_build_is_finite_causal_and_uses_one_common_ablation_universe() -> None:
    result = build_phase7_features(_inputs(), _mvp_config(), _phase7_config())
    features = result.features

    assert not features.empty
    assert result.diagnostics["nonfinite_output_values"] == 0
    assert features["feature_available_at_utc"].le(features["prediction_time_utc"]).all()
    assert tuple(result.catalog.variants) == (
        "mvp",
        "price_3min",
        "price_session_3min",
        "price_session_regime_3min",
        "price_session_regime_multitimeframe",
    )
    assert "microstructure" in result.catalog.unavailable_groups
    volatility_flags = features[
        [
            "p7_regime_volatility_low",
            "p7_regime_volatility_mid",
            "p7_regime_volatility_high",
        ]
    ]
    trend_flags = features[
        [
            "p7_regime_trend_down",
            "p7_regime_trend_range",
            "p7_regime_trend_up",
        ]
    ]
    assert volatility_flags.sum(axis=1).eq(1.0).all()
    assert trend_flags.sum(axis=1).eq(1.0).all()
    final_names = set(result.catalog.variants["price_session_regime_multitimeframe"])
    assert final_names == set(result.catalog.feature_names)
    for names in result.catalog.variants.values():
        assert set(names) <= final_names


def test_phase7_future_mutation_does_not_change_older_feature_rows() -> None:
    inputs = _inputs()
    config = _phase7_config()
    original = build_phase7_features(inputs, _mvp_config(), config).features
    cutoff = original.iloc[len(original) // 2]["prediction_time_utc"]

    changed: dict[str, pd.DataFrame] = {}
    for timeframe, frame in inputs.items():
        mutated = frame.copy(deep=True)
        future = mutated["timestamp_close_utc"].gt(cutoff)
        price_columns = ["bid_open", "bid_high", "bid_low", "bid_close"]
        mutated.loc[future, price_columns] = mutated.loc[future, price_columns] * 1.25
        changed[timeframe] = mutated

    rebuilt = build_phase7_features(changed, _mvp_config(), config).features
    before = original.loc[original["prediction_time_utc"].le(cutoff)].reset_index(drop=True)
    after = rebuilt.loc[rebuilt["prediction_time_utc"].le(cutoff)].reset_index(drop=True)
    assert_frame_equal(after, before)


def test_phase7_online_helper_matches_batch_builder_exactly() -> None:
    inputs = _inputs()
    config = _phase7_config()
    batch = build_phase7_features(inputs, _mvp_config(), config).features
    cutoff = batch.iloc[-5]["prediction_time_utc"]
    expected = batch.loc[batch["prediction_time_utc"].eq(cutoff)].reset_index(drop=True)

    online = build_phase7_feature_row(
        inputs,
        _mvp_config(),
        config,
        prediction_time_utc=cutoff,
    )

    assert_frame_equal(online, expected)


def test_phase7_three_minute_momentum_matches_hand_calculation() -> None:
    inputs = _inputs()
    result = build_phase7_features(inputs, _mvp_config(), _phase7_config())
    row = result.features.iloc[0]
    prediction = row["prediction_time_utc"]
    source = inputs["3min"]
    index = source.index[source["timestamp_close_utc"].eq(prediction)].item()
    expected = 10_000.0 * np.log(
        source.loc[index, "bid_close"] / source.loc[index - 3, "bid_close"]
    )
    assert row["p7_3min_momentum_3_bps"] == pytest.approx(expected)


def test_phase7_slow_timeframe_never_uses_an_unclosed_candle() -> None:
    inputs = _inputs()
    config = _phase7_config()
    result = build_phase7_features(inputs, _mvp_config(), config)
    cutoff = result.features.iloc[-10]["prediction_time_utc"]
    original = result.features.loc[
        result.features["prediction_time_utc"].eq(cutoff),
        list(result.catalog.variants["price_session_regime_multitimeframe"]),
    ].reset_index(drop=True)

    changed = {name: frame.copy(deep=True) for name, frame in inputs.items()}
    one_hour = changed["1h"]
    future = one_hour["timestamp_close_utc"].gt(cutoff)
    one_hour.loc[future, ["bid_open", "bid_high", "bid_low", "bid_close"]] *= 3.0
    rebuilt = build_phase7_features(changed, _mvp_config(), config).features
    after = rebuilt.loc[
        rebuilt["prediction_time_utc"].eq(cutoff),
        list(result.catalog.variants["price_session_regime_multitimeframe"]),
    ].reset_index(drop=True)
    assert_frame_equal(after, original)
