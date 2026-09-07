"""Causal multi-timeframe feature construction for phase 7.

The phase-7 builder is deliberately separate from the frozen MVP builder.  It
reuses the exact MVP columns as the baseline ablation, then adds auditable
price, session and regime features.  Bid/ask, spread and tick-count features
are not synthesized when the provider cannot supply them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gold_forecasting.artifacts import content_version
from gold_forecasting.config import FeaturesConfig, _load_yaml_mapping
from gold_forecasting.features.mvp import FEATURE_METADATA_COLUMNS, build_mvp_features
from gold_forecasting.validation import CandleValidationError, validate_candles

_ALLOWED_TIMEFRAMES = ("1min", "3min", "5min", "15min", "30min", "1h", "3h")
_TIMEFRAME_MINUTES = {
    "1min": 1,
    "3min": 3,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1h": 60,
    "3h": 180,
}
_VARIANT_ORDER = (
    "mvp",
    "price_3min",
    "price_session_3min",
    "price_session_regime_3min",
    "price_session_regime_multitimeframe",
)


class Phase7FeatureBuildError(ValueError):
    """Raised when phase-7 features cannot be built without violating the time contract."""


class Phase7FeatureConfig(BaseModel):
    """Explicit research-only feature budget for phase 7."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    feature_protocol: Literal["phase7-v1"] = "phase7-v1"
    anchor_timeframe: Literal["3min"] = "3min"
    timeframe_windows: dict[str, tuple[int, ...]]
    return_lags: tuple[int, ...] = (1, 2, 3)
    regime_lookback_candles: int = Field(default=80, ge=20, le=240)
    staleness_multiplier: float = Field(default=1.0, ge=1.0, le=2.0)
    zero_range_close_position: float = Field(default=0.5, ge=0.0, le=1.0)
    microstructure_enabled: Literal[False] = False

    @field_validator("return_lags")
    @classmethod
    def validate_lags(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if not values or any(value <= 0 for value in values):
            raise ValueError("return_lags must contain positive integers")
        if tuple(sorted(set(values))) != values:
            raise ValueError("return_lags must be sorted and unique")
        return values

    @model_validator(mode="after")
    def validate_timeframe_budget(self) -> Self:
        if tuple(self.timeframe_windows) != _ALLOWED_TIMEFRAMES:
            raise ValueError(
                "timeframe_windows must contain exactly, in order: "
                + ", ".join(_ALLOWED_TIMEFRAMES)
            )
        for timeframe, windows in self.timeframe_windows.items():
            if not windows or any(value <= 1 for value in windows):
                raise ValueError(f"{timeframe} windows must be integers greater than one")
            if tuple(sorted(set(windows))) != windows:
                raise ValueError(f"{timeframe} windows must be sorted and unique")
        return self


class Phase7FeatureDefinition(BaseModel):
    """One auditable model input in the phase-7 feature catalog."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    group: Literal["mvp", "price", "session", "regime"]
    source_timeframe: str = Field(min_length=1)
    formula: str = Field(min_length=1)
    lookback_candles: int = Field(ge=1)
    lookback_minutes: int = Field(ge=0)
    available_at_rule: Literal["closed_candle_or_prediction_time"] = (
        "closed_candle_or_prediction_time"
    )


class Phase7FeatureCatalog(BaseModel):
    """Ordered feature catalog plus frozen ablation variants."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    feature_protocol: Literal["phase7-v1"] = "phase7-v1"
    feature_spec_version: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    anchor_timeframe: Literal["3min"] = "3min"
    definitions: tuple[Phase7FeatureDefinition, ...]
    variants: dict[str, tuple[str, ...]]
    unavailable_groups: dict[str, str]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.definitions)


@dataclass(frozen=True, slots=True)
class Phase7FeatureBuildResult:
    features: pd.DataFrame
    catalog: Phase7FeatureCatalog
    diagnostics: dict[str, object]


def load_phase7_feature_config(path: str | Path) -> Phase7FeatureConfig:
    return Phase7FeatureConfig.model_validate(_load_yaml_mapping(Path(path)))


def _segment_ids(frame: pd.DataFrame, *, timeframe: str) -> pd.Series:
    minutes = _TIMEFRAME_MINUTES[timeframe]
    changed = (
        frame["instrument"].ne(frame["instrument"].shift())
        | frame["source"].ne(frame["source"].shift())
    )
    broken = frame["timestamp_open_utc"].diff().ne(pd.Timedelta(minutes=minutes))
    return (changed | broken).cumsum().astype("int64")


def _feature_segments(frame: pd.DataFrame) -> pd.Series:
    changed = (
        frame["instrument"].ne(frame["instrument"].shift())
        | frame["source"].ne(frame["source"].shift())
    )
    broken = frame["prediction_time_utc"].diff().ne(pd.Timedelta(minutes=3))
    return (changed | broken).cumsum().astype("int64")


def _definition(
    name: str,
    *,
    group: Literal["price", "session", "regime"],
    timeframe: str,
    formula: str,
    lookback_candles: int,
) -> Phase7FeatureDefinition:
    minutes = _TIMEFRAME_MINUTES.get(timeframe, 0)
    return Phase7FeatureDefinition(
        name=name,
        group=group,
        source_timeframe=timeframe,
        formula=formula,
        lookback_candles=lookback_candles,
        lookback_minutes=max(0, lookback_candles * minutes),
    )


def _price_feature_frame(
    candles: pd.DataFrame,
    *,
    timeframe: str,
    config: Phase7FeatureConfig,
) -> tuple[pd.DataFrame, tuple[Phase7FeatureDefinition, ...]]:
    try:
        validated = validate_candles(candles, expected_timeframe=timeframe).candles
    except CandleValidationError as exc:
        raise Phase7FeatureBuildError(f"invalid {timeframe} input: {exc}") from exc
    if validated.empty:
        raise Phase7FeatureBuildError(f"phase 7 requires non-empty {timeframe} curated data")

    frame = validated.sort_values(
        ["instrument", "source", "timestamp_open_utc"], kind="stable"
    ).reset_index(drop=True)
    segment = _segment_ids(frame, timeframe=timeframe)
    close = frame["bid_close"].astype("float64")
    open_price = frame["bid_open"].astype("float64")
    high = frame["bid_high"].astype("float64")
    low = frame["bid_low"].astype("float64")
    log_close = pd.Series(np.log(close.to_numpy()), index=frame.index, dtype="float64")
    returns = log_close.groupby(segment, sort=False).diff() * 10_000.0
    slug = timeframe
    values = pd.DataFrame(index=frame.index)
    definitions: list[Phase7FeatureDefinition] = []

    def add(name: str, values_: pd.Series | np.ndarray, formula: str, lookback: int) -> None:
        values[name] = values_
        definitions.append(
            _definition(
                name,
                group="price",
                timeframe=timeframe,
                formula=formula,
                lookback_candles=lookback,
            )
        )

    add(
        f"p7_{slug}_log_return_1_bps",
        returns,
        "10000 * ln(close[t] / close[t-1])",
        2,
    )
    for lag in config.return_lags:
        add(
            f"p7_{slug}_log_return_lag_{lag}_bps",
            returns.groupby(segment, sort=False).shift(lag),
            f"one-candle log return shifted by {lag} closed candle(s)",
            lag + 2,
        )

    add(
        f"p7_{slug}_candle_body_bps",
        10_000.0 * (close - open_price) / open_price,
        "10000 * (close-open) / open",
        1,
    )
    add(
        f"p7_{slug}_candle_range_bps",
        10_000.0 * (high - low) / open_price,
        "10000 * (high-low) / open",
        1,
    )
    add(
        f"p7_{slug}_upper_wick_bps",
        10_000.0 * (high - pd.concat([open_price, close], axis=1).max(axis=1)) / open_price,
        "10000 * (high-max(open,close)) / open",
        1,
    )
    add(
        f"p7_{slug}_lower_wick_bps",
        10_000.0 * (pd.concat([open_price, close], axis=1).min(axis=1) - low) / open_price,
        "10000 * (min(open,close)-low) / open",
        1,
    )
    candle_range = high - low
    add(
        f"p7_{slug}_close_position",
        pd.Series(
            np.where(
                candle_range.eq(0.0),
                config.zero_range_close_position,
                (close - low) / candle_range,
            ),
            index=frame.index,
        ),
        "(close-low)/(high-low), configured constant on zero range",
        1,
    )

    windows = config.timeframe_windows[timeframe]
    for window in windows:
        prior = log_close.groupby(segment, sort=False).shift(window)
        add(
            f"p7_{slug}_momentum_{window}_bps",
            10_000.0 * (log_close - prior),
            f"10000 * ln(close[t] / close[t-{window}])",
            window + 1,
        )
        squared = returns.pow(2)
        realized = (
            squared.groupby(segment, sort=False)
            .rolling(window, min_periods=window)
            .sum()
            .reset_index(level=0, drop=True)
        )
        add(
            f"p7_{slug}_realized_volatility_{window}_bps",
            np.sqrt(realized),
            f"sqrt(sum(last {window} one-candle log returns squared))",
            window + 1,
        )

    longest = windows[-1]
    prior_close = close.groupby(segment, sort=False).shift(1)
    rolling_mean = (
        prior_close.groupby(segment, sort=False)
        .rolling(longest, min_periods=longest)
        .mean()
        .reset_index(level=0, drop=True)
    )
    rolling_std = (
        prior_close.groupby(segment, sort=False)
        .rolling(longest, min_periods=longest)
        .std(ddof=0)
        .reset_index(level=0, drop=True)
    )
    add(
        f"p7_{slug}_distance_to_sma_{longest}_bps",
        10_000.0 * np.log(close / rolling_mean),
        f"10000 * ln(close[t] / mean(previous {longest} closes))",
        longest + 1,
    )
    add(
        f"p7_{slug}_price_zscore_{longest}",
        (close - rolling_mean) / rolling_std.replace(0.0, np.nan),
        f"(close[t]-mean(previous {longest} closes))/std(previous {longest} closes)",
        longest + 1,
    )
    prior_high = high.groupby(segment, sort=False).shift(1)
    prior_low = low.groupby(segment, sort=False).shift(1)
    rolling_high = (
        prior_high.groupby(segment, sort=False)
        .rolling(longest, min_periods=longest)
        .max()
        .reset_index(level=0, drop=True)
    )
    rolling_low = (
        prior_low.groupby(segment, sort=False)
        .rolling(longest, min_periods=longest)
        .min()
        .reset_index(level=0, drop=True)
    )
    breakout_range = rolling_high - rolling_low
    add(
        f"p7_{slug}_breakout_position_{longest}",
        (close - rolling_low) / breakout_range.replace(0.0, np.nan),
        f"(close[t]-min(previous {longest} lows))/(max(previous {longest} highs)-min(...))",
        longest + 1,
    )

    required_prior = max(max(config.return_lags) + 1, longest)
    window_start = frame["timestamp_open_utc"].groupby(segment, sort=False).shift(required_prior)
    output = pd.concat(
        [
            frame[["instrument", "source"]].copy(),
            pd.DataFrame(
                {
                    "_p7_available_at_utc": frame["timestamp_close_utc"],
                    "_p7_window_start_utc": window_start,
                }
            ),
            values,
        ],
        axis=1,
    )
    numeric = values.to_numpy(dtype=np.float64)
    valid = values.notna().all(axis=1) & np.isfinite(numeric).all(axis=1) & window_start.notna()
    output = output.loc[valid].reset_index(drop=True)
    return output, tuple(definitions)


def _session_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[Phase7FeatureDefinition, ...]]:
    prediction = frame["prediction_time_utc"]
    hour = prediction.dt.hour + prediction.dt.minute / 60.0
    minute = prediction.dt.minute.astype("float64")
    month = prediction.dt.month.astype("float64") - 1.0
    values = pd.DataFrame(index=frame.index)
    values["p7_session_month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    values["p7_session_month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    values["p7_session_minute_sin"] = np.sin(2.0 * np.pi * minute / 60.0)
    values["p7_session_minute_cos"] = np.cos(2.0 * np.pi * minute / 60.0)
    values["p7_session_asia"] = ((hour >= 0.0) & (hour < 8.0)).astype("float64")
    values["p7_session_europe"] = ((hour >= 7.0) & (hour < 16.0)).astype("float64")
    values["p7_session_us"] = ((hour >= 13.0) & (hour < 22.0)).astype("float64")
    values["p7_session_europe_us_overlap"] = (
        (hour >= 13.0) & (hour < 16.0)
    ).astype("float64")
    formulas = {
        "p7_session_month_sin": "sin(2*pi*(UTC month-1)/12)",
        "p7_session_month_cos": "cos(2*pi*(UTC month-1)/12)",
        "p7_session_minute_sin": "sin(2*pi*UTC minute/60)",
        "p7_session_minute_cos": "cos(2*pi*UTC minute/60)",
        "p7_session_asia": "1 for fixed UTC hour in [00:00,08:00), else 0",
        "p7_session_europe": "1 for fixed UTC hour in [07:00,16:00), else 0",
        "p7_session_us": "1 for fixed UTC hour in [13:00,22:00), else 0",
        "p7_session_europe_us_overlap": "1 for fixed UTC hour in [13:00,16:00), else 0",
    }
    definitions = tuple(
        _definition(
            name,
            group="session",
            timeframe="3min",
            formula=formulas[name],
            lookback_candles=1,
        )
        for name in formulas
    )
    return values, definitions


def _regime_features(
    frame: pd.DataFrame,
    *,
    config: Phase7FeatureConfig,
) -> tuple[pd.DataFrame, tuple[Phase7FeatureDefinition, ...]]:
    segment = _feature_segments(frame)
    volatility = frame["realized_volatility_20_bps"].astype("float64")
    momentum = frame["momentum_20_bps"].astype("float64")
    one_return = frame["close_log_return_1_bps"].astype("float64")

    prior_volatility = volatility.groupby(segment, sort=False).shift(1)
    low_threshold = (
        prior_volatility.groupby(segment, sort=False)
        .rolling(config.regime_lookback_candles, min_periods=config.regime_lookback_candles)
        .quantile(1.0 / 3.0)
        .reset_index(level=0, drop=True)
    )
    high_threshold = (
        prior_volatility.groupby(segment, sort=False)
        .rolling(config.regime_lookback_candles, min_periods=config.regime_lookback_candles)
        .quantile(2.0 / 3.0)
        .reset_index(level=0, drop=True)
    )
    prior_abs_return = one_return.abs().groupby(segment, sort=False).shift(1)
    shock_scale = (
        prior_abs_return.groupby(segment, sort=False)
        .rolling(20, min_periods=20)
        .median()
        .reset_index(level=0, drop=True)
    )
    trend_score = momentum / volatility.replace(0.0, np.nan)
    shock_score = one_return.abs() / shock_scale.replace(0.0, np.nan)

    values = pd.DataFrame(index=frame.index)
    volatility_ready = low_threshold.notna() & high_threshold.notna()
    trend_ready = trend_score.notna()
    shock_ready = shock_score.notna()

    values["p7_regime_volatility_low"] = (volatility < low_threshold).astype("float64")
    values["p7_regime_volatility_mid"] = (
        (volatility >= low_threshold) & (volatility <= high_threshold)
    ).astype("float64")
    values["p7_regime_volatility_high"] = (volatility > high_threshold).astype("float64")
    values.loc[
        ~volatility_ready,
        [
            "p7_regime_volatility_low",
            "p7_regime_volatility_mid",
            "p7_regime_volatility_high",
        ],
    ] = np.nan

    values["p7_regime_trend_score"] = trend_score
    values["p7_regime_trend_down"] = (trend_score < -1.0).astype("float64")
    values["p7_regime_trend_range"] = trend_score.abs().le(1.0).astype("float64")
    values["p7_regime_trend_up"] = (trend_score > 1.0).astype("float64")
    values.loc[
        ~trend_ready,
        ["p7_regime_trend_down", "p7_regime_trend_range", "p7_regime_trend_up"],
    ] = np.nan

    values["p7_regime_shock_score"] = shock_score
    values["p7_regime_shock"] = (shock_score >= 3.0).astype("float64")
    values.loc[~shock_ready, "p7_regime_shock"] = np.nan

    formulas = {
        "p7_regime_volatility_low": "rv20 below trailing prior one-third quantile",
        "p7_regime_volatility_mid": "rv20 between trailing prior one-third/two-third quantiles",
        "p7_regime_volatility_high": "rv20 above trailing prior two-third quantile",
        "p7_regime_trend_score": "momentum20 / realized_volatility20",
        "p7_regime_trend_down": "1 when trend_score < -1",
        "p7_regime_trend_range": "1 when abs(trend_score) <= 1",
        "p7_regime_trend_up": "1 when trend_score > 1",
        "p7_regime_shock_score": "abs(return1) / median(abs(previous 20 returns))",
        "p7_regime_shock": "1 when shock_score >= 3",
    }
    definitions = tuple(
        _definition(
            name,
            group="regime",
            timeframe="3min",
            formula=formulas[name],
            lookback_candles=(
                config.regime_lookback_candles + 1
                if name.startswith("p7_regime_volatility_")
                else 21
            ),
        )
        for name in formulas
    )
    return values, definitions


def _merge_asof_features(
    anchor: pd.DataFrame,
    right: pd.DataFrame,
    *,
    timeframe: str,
    feature_names: tuple[str, ...],
    staleness_multiplier: float,
) -> tuple[pd.DataFrame, str, str, int]:
    available_column = f"_p7_{timeframe}_available_at_utc"
    window_column = f"_p7_{timeframe}_window_start_utc"
    right_frame = right.rename(
        columns={
            "_p7_available_at_utc": available_column,
            "_p7_window_start_utc": window_column,
        }
    )
    left = anchor.sort_values(
        ["prediction_time_utc", "instrument", "source"], kind="stable"
    ).reset_index(drop=True)
    right_frame = right_frame.sort_values(
        [available_column, "instrument", "source"], kind="stable"
    ).reset_index(drop=True)
    merged = pd.merge_asof(
        left,
        right_frame,
        by=["instrument", "source"],
        left_on="prediction_time_utc",
        right_on=available_column,
        direction="backward",
        allow_exact_matches=True,
    )
    age_minutes = (
        merged["prediction_time_utc"] - merged[available_column]
    ).dt.total_seconds() / 60.0
    maximum_age = _TIMEFRAME_MINUTES[timeframe] * staleness_multiplier
    stale = age_minutes.lt(0.0) | age_minutes.gt(maximum_age) | age_minutes.isna()
    stale_count = int(stale.sum())
    if stale_count:
        merged.loc[stale, list(feature_names)] = np.nan
        merged.loc[stale, [available_column, window_column]] = pd.NaT
    return merged, available_column, window_column, stale_count


def build_phase7_features(
    candles_by_timeframe: dict[str, pd.DataFrame],
    mvp_config: FeaturesConfig,
    config: Phase7FeatureConfig,
) -> Phase7FeatureBuildResult:
    """Build the common causal sample universe used by every phase-7 ablation."""
    started = time.perf_counter()
    missing = [name for name in config.timeframe_windows if name not in candles_by_timeframe]
    if missing:
        raise Phase7FeatureBuildError(f"missing configured candle inputs: {', '.join(missing)}")
    if mvp_config.input_timeframe != config.anchor_timeframe:
        raise Phase7FeatureBuildError(
            "phase-7 anchor must preserve the MVP 3min prediction cadence"
        )

    base = build_mvp_features(candles_by_timeframe["3min"], mvp_config)
    combined = base.features.copy()
    definitions: list[Phase7FeatureDefinition] = [
        Phase7FeatureDefinition(
            name=item.name,
            group="mvp",
            source_timeframe=item.source_timeframe,
            formula=item.formula,
            lookback_candles=item.lookback_candles,
            lookback_minutes=item.lookback_minutes,
        )
        for item in base.catalog.definitions
    ]
    price_names_by_timeframe: dict[str, tuple[str, ...]] = {}
    availability_columns: list[str] = []
    window_columns: list[str] = []
    stale_counts: dict[str, int] = {}

    for timeframe in config.timeframe_windows:
        price_frame, price_definitions = _price_feature_frame(
            candles_by_timeframe[timeframe], timeframe=timeframe, config=config
        )
        names = tuple(item.name for item in price_definitions)
        combined, available, window, stale_count = _merge_asof_features(
            combined,
            price_frame,
            timeframe=timeframe,
            feature_names=names,
            staleness_multiplier=config.staleness_multiplier,
        )
        price_names_by_timeframe[timeframe] = names
        definitions.extend(price_definitions)
        availability_columns.append(available)
        window_columns.append(window)
        stale_counts[timeframe] = stale_count

    session_values, session_definitions = _session_features(combined)
    combined = pd.concat([combined, session_values], axis=1)
    definitions.extend(session_definitions)
    regime_values, regime_definitions = _regime_features(combined, config=config)
    combined = pd.concat([combined, regime_values], axis=1)
    definitions.extend(regime_definitions)

    mvp_names = base.catalog.feature_names
    session_names = tuple(item.name for item in session_definitions)
    regime_names = tuple(item.name for item in regime_definitions)
    price_3min = price_names_by_timeframe["3min"]
    higher_price = tuple(
        name
        for timeframe in config.timeframe_windows
        if timeframe != "3min"
        for name in price_names_by_timeframe[timeframe]
    )
    variants = {
        "mvp": mvp_names,
        "price_3min": mvp_names + price_3min,
        "price_session_3min": mvp_names + price_3min + session_names,
        "price_session_regime_3min": mvp_names + price_3min + session_names + regime_names,
        "price_session_regime_multitimeframe": (
            mvp_names + price_3min + session_names + regime_names + higher_price
        ),
    }
    if tuple(variants) != _VARIANT_ORDER:
        raise Phase7FeatureBuildError("ablation variant order changed unexpectedly")

    unavailable_groups: dict[str, str] = {
        "microstructure": (
            "disabled: HistData development source is bid-only and does not provide "
            "reliable ask, spread or tick-count history"
        )
    }
    stable_catalog: dict[str, object] = {
        "schema_version": 1,
        "feature_protocol": config.feature_protocol,
        "anchor_timeframe": config.anchor_timeframe,
        "config": config.model_dump(mode="json"),
        "definitions": [item.model_dump(mode="json") for item in definitions],
        "variants": {key: list(value) for key, value in variants.items()},
        "unavailable_groups": unavailable_groups,
    }
    catalog = Phase7FeatureCatalog(
        feature_spec_version=content_version(stable_catalog),
        definitions=tuple(definitions),
        variants=variants,
        unavailable_groups=unavailable_groups,
    )
    if len(set(catalog.feature_names)) != len(catalog.feature_names):
        raise Phase7FeatureBuildError("phase-7 feature names are not unique")

    feature_matrix = combined.loc[:, list(catalog.feature_names)]
    finite = pd.Series(
        np.isfinite(feature_matrix.to_numpy(dtype=np.float64)).all(axis=1),
        index=combined.index,
    )
    source_complete = combined[availability_columns + window_columns].notna().all(axis=1)
    valid = finite & source_complete
    for available in availability_columns:
        valid &= combined[available].le(combined["prediction_time_utc"])

    prefilter_rows = len(combined)
    combined = combined.loc[valid].copy()
    if combined.empty:
        raise Phase7FeatureBuildError("no common rows remain after phase-7 feature alignment")

    all_window_columns = ["feature_window_start_utc", *window_columns]
    combined["feature_window_start_utc"] = combined[all_window_columns].min(axis=1)
    combined["feature_available_at_utc"] = combined["prediction_time_utc"]
    internal = [*availability_columns, *window_columns]
    combined = combined.drop(columns=internal)
    if combined.duplicated(["instrument", "source", "prediction_time_utc"]).any():
        raise Phase7FeatureBuildError("phase-7 features contain duplicate prediction keys")

    ordered = [
        *FEATURE_METADATA_COLUMNS,
        *catalog.feature_names,
    ]
    combined = combined.loc[:, ordered].sort_values(
        ["instrument", "source", "prediction_time_utc"], kind="stable"
    ).reset_index(drop=True)
    diagnostics: dict[str, object] = {
        "feature_protocol": config.feature_protocol,
        "input_rows": {name: len(candles_by_timeframe[name]) for name in config.timeframe_windows},
        "anchor_mvp_rows": base.output_row_count,
        "prefilter_rows": prefilter_rows,
        "output_rows": len(combined),
        "dropped_for_common_universe": prefilter_rows - len(combined),
        "stale_alignment_rows": stale_counts,
        "feature_count": len(catalog.feature_names),
        "variant_feature_counts": {key: len(value) for key, value in variants.items()},
        "nonfinite_output_values": int(
            (~np.isfinite(combined.loc[:, list(catalog.feature_names)].to_numpy())).sum()
        ),
        "build_seconds": time.perf_counter() - started,
    }
    return Phase7FeatureBuildResult(features=combined, catalog=catalog, diagnostics=diagnostics)


def build_phase7_feature_row(
    candles_by_timeframe: dict[str, pd.DataFrame],
    mvp_config: FeaturesConfig,
    config: Phase7FeatureConfig,
    *,
    prediction_time_utc: pd.Timestamp,
) -> pd.DataFrame:
    """Online-parity helper: run the same builder using only data known by one timestamp."""
    cutoff = pd.Timestamp(prediction_time_utc)
    if cutoff.tzinfo is None:
        raise Phase7FeatureBuildError("prediction_time_utc must be timezone-aware")
    truncated = {
        timeframe: frame.loc[frame["timestamp_close_utc"].le(cutoff)].copy()
        for timeframe, frame in candles_by_timeframe.items()
    }
    built = build_phase7_features(truncated, mvp_config, config)
    row = built.features.loc[built.features["prediction_time_utc"].eq(cutoff)]
    if len(row) != 1:
        raise Phase7FeatureBuildError(
            f"expected exactly one phase-7 feature row at {cutoff.isoformat()}, found {len(row)}"
        )
    return row.reset_index(drop=True)


__all__ = [
    "Phase7FeatureBuildError",
    "Phase7FeatureBuildResult",
    "Phase7FeatureCatalog",
    "Phase7FeatureConfig",
    "Phase7FeatureDefinition",
    "build_phase7_feature_row",
    "build_phase7_features",
    "load_phase7_feature_config",
]
