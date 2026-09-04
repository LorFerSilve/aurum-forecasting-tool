"""Causal, price-only features computed from closed three-minute candles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from gold_forecasting.artifacts import content_version
from gold_forecasting.config import FeaturesConfig
from gold_forecasting.validation import CandleValidationError, validate_candles

FEATURE_METADATA_COLUMNS = (
    "instrument",
    "source",
    "prediction_time_utc",
    "feature_available_at_utc",
    "feature_window_start_utc",
    "source_candle_open_utc",
    "source_raw_file_hash",
    "source_dataset_version",
)


class FeatureBuildError(ValueError):
    """Raised when causal MVP features cannot be constructed safely."""


class FeatureDefinition(BaseModel):
    """One auditable input column in the frozen MVP feature schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    dtype: Literal["float64"] = "float64"
    formula: str = Field(min_length=1)
    source_timeframe: Literal["3min"] = "3min"
    lookback_candles: int = Field(ge=1)
    lookback_minutes: int = Field(ge=3)
    available_at_rule: Literal["source_candle_close"] = "source_candle_close"
    missing_policy: Literal["drop_until_complete_contiguous_history"] = (
        "drop_until_complete_contiguous_history"
    )


class FeatureCatalog(BaseModel):
    """Versioned, ordered catalog used as the only valid model-input schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    feature_spec_version: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    input_timeframe: Literal["3min"] = "3min"
    definitions: tuple[FeatureDefinition, ...]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(definition.name for definition in self.definitions)


@dataclass(frozen=True, slots=True)
class FeatureBuildResult:
    features: pd.DataFrame
    catalog: FeatureCatalog
    input_row_count: int
    output_row_count: int
    dropped_for_history: int


def _definition(name: str, formula: str, lookback_candles: int) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        formula=formula,
        lookback_candles=lookback_candles,
        lookback_minutes=lookback_candles * 3,
    )


def build_feature_catalog(config: FeaturesConfig) -> FeatureCatalog:
    """Build the deterministic catalog implied by the current feature config."""

    definitions: list[FeatureDefinition] = [
        _definition("close_log_return_1_bps", "10000 * ln(close[t] / close[t-1])", 2),
    ]
    definitions.extend(
        _definition(
            f"close_log_return_lag_{lag}_bps",
            f"close_log_return_1_bps shifted by {lag} closed 3min candle(s)",
            lag + 2,
        )
        for lag in config.return_lags
    )
    definitions.extend(
        [
            _definition("candle_body_bps", "10000 * (close-open) / open", 1),
            _definition("candle_range_bps", "10000 * (high-low) / open", 1),
            _definition(
                "upper_wick_bps",
                "10000 * (high-max(open,close)) / open",
                1,
            ),
            _definition(
                "lower_wick_bps",
                "10000 * (min(open,close)-low) / open",
                1,
            ),
            _definition("close_position", "(close-low)/(high-low); constant on zero range", 1),
        ]
    )
    definitions.extend(
        _definition(
            f"momentum_{window}_bps",
            f"10000 * ln(close[t] / close[t-{window}])",
            window + 1,
        )
        for window in config.momentum_windows
    )
    rv_window = config.realized_volatility_window
    definitions.append(
        _definition(
            f"realized_volatility_{rv_window}_bps",
            f"sqrt(sum(last {rv_window} close_log_return_1_bps squared))",
            rv_window + 1,
        )
    )
    definitions.extend(
        _definition(
            f"distance_to_sma_{window}_bps",
            f"10000 * ln(close[t] / mean(close[t-{window}:t-1]))",
            window + 1,
        )
        for window in config.moving_average_windows
    )
    definitions.extend(
        [
            _definition("hour_sin", "sin(2*pi*UTC fractional hour/24)", 1),
            _definition("hour_cos", "cos(2*pi*UTC fractional hour/24)", 1),
            _definition("weekday_sin", "sin(2*pi*UTC weekday/7)", 1),
            _definition("weekday_cos", "cos(2*pi*UTC weekday/7)", 1),
        ]
    )
    stable = {
        "schema_version": 1,
        "input_timeframe": config.input_timeframe,
        "config": config.model_dump(mode="json"),
        "definitions": [item.model_dump(mode="json") for item in definitions],
    }
    return FeatureCatalog(
        feature_spec_version=content_version(stable),
        definitions=tuple(definitions),
    )


def _segment_ids(frame: pd.DataFrame) -> pd.Series:
    group_changed = (
        frame["instrument"].ne(frame["instrument"].shift())
        | frame["source"].ne(frame["source"].shift())
    )
    cadence_broken = frame["timestamp_open_utc"].diff().ne(pd.Timedelta(minutes=3))
    return (group_changed | cadence_broken).cumsum().astype("int64")


def build_mvp_features(
    candles_3min: pd.DataFrame,
    config: FeaturesConfig,
) -> FeatureBuildResult:
    """Build features whose latest source value is available at prediction time.

    Rolling calculations are segmented at every market-data gap, so neither a
    weekend nor a missing candle is silently treated as contiguous history.
    """

    if config.input_timeframe != "3min":
        raise FeatureBuildError("the MVP feature builder requires input_timeframe='3min'")
    if config.tick_count_enabled:
        raise FeatureBuildError("tick-count features are unavailable for the bid-only MVP source")
    try:
        validated = validate_candles(candles_3min, expected_timeframe="3min").candles
    except CandleValidationError as exc:
        raise FeatureBuildError(f"invalid 3min feature input: {exc}") from exc
    if validated.empty:
        raise FeatureBuildError("at least one 3min candle is required")

    frame = validated.sort_values(
        ["instrument", "source", "timestamp_open_utc"], kind="stable"
    ).reset_index(drop=True)
    frame["_segment_id"] = _segment_ids(frame)
    segment = frame["_segment_id"]
    close = frame["bid_close"].astype("float64")
    log_close = pd.Series(
        np.log(close.to_numpy(dtype=np.float64)),
        index=frame.index,
        dtype="float64",
    )
    returns = log_close.groupby(segment, sort=False).diff() * 10_000.0

    values = pd.DataFrame(index=frame.index)
    values["close_log_return_1_bps"] = returns
    for lag in config.return_lags:
        values[f"close_log_return_lag_{lag}_bps"] = returns.groupby(
            segment, sort=False
        ).shift(lag)

    open_price = frame["bid_open"].astype("float64")
    high = frame["bid_high"].astype("float64")
    low = frame["bid_low"].astype("float64")
    values["candle_body_bps"] = 10_000.0 * (close - open_price) / open_price
    values["candle_range_bps"] = 10_000.0 * (high - low) / open_price
    values["upper_wick_bps"] = 10_000.0 * (
        high - pd.concat([open_price, close], axis=1).max(axis=1)
    ) / open_price
    values["lower_wick_bps"] = 10_000.0 * (
        pd.concat([open_price, close], axis=1).min(axis=1) - low
    ) / open_price
    candle_range = high - low
    values["close_position"] = np.where(
        candle_range.eq(0.0),
        config.zero_range_close_position,
        (close - low) / candle_range,
    )

    for window in config.momentum_windows:
        prior = log_close.groupby(segment, sort=False).shift(window)
        values[f"momentum_{window}_bps"] = 10_000.0 * (log_close - prior)

    rv_window = config.realized_volatility_window
    squared = returns.pow(2)
    rv = (
        squared.groupby(segment, sort=False)
        .rolling(rv_window, min_periods=rv_window)
        .sum()
        .reset_index(level=0, drop=True)
    )
    values[f"realized_volatility_{rv_window}_bps"] = np.sqrt(rv)

    prior_close = close.groupby(segment, sort=False).shift(1)
    for window in config.moving_average_windows:
        moving_average = (
            prior_close.groupby(segment, sort=False)
            .rolling(window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
        )
        values[f"distance_to_sma_{window}_bps"] = 10_000.0 * np.log(
            close / moving_average
        )

    prediction_time = frame["timestamp_close_utc"]
    fractional_hour = prediction_time.dt.hour + prediction_time.dt.minute / 60.0
    values["hour_sin"] = np.sin(2.0 * np.pi * fractional_hour / 24.0)
    values["hour_cos"] = np.cos(2.0 * np.pi * fractional_hour / 24.0)
    values["weekday_sin"] = np.sin(2.0 * np.pi * prediction_time.dt.dayofweek / 7.0)
    values["weekday_cos"] = np.cos(2.0 * np.pi * prediction_time.dt.dayofweek / 7.0)

    catalog = build_feature_catalog(config)
    if tuple(values.columns) != catalog.feature_names:
        raise FeatureBuildError("generated feature order differs from the feature catalog")
    window_start = frame["timestamp_open_utc"].groupby(segment, sort=False).shift(
        config.maximum_prior_candles
    )
    valid = values.notna().all(axis=1) & window_start.notna()
    numeric = values.to_numpy(dtype=np.float64)
    valid &= pd.Series(np.isfinite(numeric).all(axis=1), index=values.index)

    metadata = pd.DataFrame(
        {
            "instrument": frame["instrument"],
            "source": frame["source"],
            "prediction_time_utc": prediction_time,
            "feature_available_at_utc": prediction_time,
            "feature_window_start_utc": window_start,
            "source_candle_open_utc": frame["timestamp_open_utc"],
            "source_raw_file_hash": frame["raw_file_hash"],
            "source_dataset_version": frame["dataset_version"],
        }
    )
    output = pd.concat([metadata, values], axis=1).loc[valid].reset_index(drop=True)
    if output.empty:
        raise FeatureBuildError(
            "no feature rows have the required complete contiguous history"
        )
    if not output["feature_available_at_utc"].le(output["prediction_time_utc"]).all():
        raise FeatureBuildError("a feature became available after prediction time")
    if output.duplicated(
        subset=["instrument", "source", "prediction_time_utc"]
    ).any():
        raise FeatureBuildError("feature rows have duplicate prediction keys")

    return FeatureBuildResult(
        features=output,
        catalog=catalog,
        input_row_count=len(frame),
        output_row_count=len(output),
        dropped_for_history=len(frame) - len(output),
    )


__all__ = [
    "FEATURE_METADATA_COLUMNS",
    "FeatureBuildError",
    "FeatureBuildResult",
    "FeatureCatalog",
    "FeatureDefinition",
    "build_feature_catalog",
    "build_mvp_features",
]
