"""Execution-price-aware, fixed-notional walk-forward research backtester.

The output is a sum of arithmetic P&L basis points on one entry-bid notional
per trade, not a compounded account return. Bid-only data uses a documented
constant-spread ask proxy. No future quote enters the decision policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from gold_forecasting.classification import ClassificationContractError, PredictionBatch

INPUT_COLUMNS = (
    "instrument",
    "prediction_time_utc",
    "entry_time_utc",
    "label_end_time_utc",
    "entry_bid_open",
    "exit_bid_open",
    "predicted_class",
    "p_down",
    "p_neutral",
    "p_up",
    "expected_return_bps",
    "horizon_minutes",
)
_TIME_COLUMNS = ("prediction_time_utc", "entry_time_utc", "label_end_time_utc")
_PROBABILITY_COLUMNS = ("p_down", "p_neutral", "p_up")
_RESULT_COLUMNS = (
    "signal_time_utc",
    "order_time_utc",
    "fill_time_utc",
    "exit_fill_time_utc",
    "modeled_latency_seconds",
    "signal_confidence",
    "side",
    "decision_status",
    "estimated_cost_bps",
    "expected_directional_return_bps",
    "expected_net_return_bps",
    "entry_ask_basis",
    "exit_ask_basis",
    "entry_fill_price",
    "exit_fill_price",
    "gross_return_bps",
    "spread_cost_bps",
    "slippage_cost_bps",
    "commission_cost_bps",
    "cost_bps",
    "net_return_bps",
    "utc_hour",
    "utc_session",
)


class BacktestV1Error(ValueError):
    """Inputs violate the execution or prediction contract."""


def _nonnegative(value: float, name: str) -> None:
    if isinstance(value, bool) or not np.isfinite(value) or value < 0.0:
        raise BacktestV1Error(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class BacktestCosts:
    """Spread proxy and adverse slippage; commissions use fixed entry notional."""

    spread_bps: float = 3.0
    slippage_per_side_bps: float = 0.5
    commission_per_side_bps: float = 0.0

    def __post_init__(self) -> None:
        for name in ("spread_bps", "slippage_per_side_bps", "commission_per_side_bps"):
            _nonnegative(getattr(self, name), name)
        if self.slippage_per_side_bps >= 10_000:
            raise BacktestV1Error("slippage_per_side_bps must be less than 10000")

    @property
    def estimated_round_trip_bps(self) -> float:
        """Ex-ante small-return approximation, independent of future quotes."""
        return self.spread_bps + 2 * (self.slippage_per_side_bps + self.commission_per_side_bps)


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    confidence_threshold: float = 0.5
    min_expected_net_bps: float = 0.0

    def __post_init__(self) -> None:
        _nonnegative(self.confidence_threshold, "confidence_threshold")
        _nonnegative(self.min_expected_net_bps, "min_expected_net_bps")
        if self.confidence_threshold > 1.0:
            raise BacktestV1Error("confidence_threshold must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class BacktestV1Result:
    decisions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, object]


def _utc_column(series: pd.Series, name: str) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        if series.isna().any():
            raise BacktestV1Error(f"{name} must contain valid timezone-aware timestamps")
        return cast(pd.Series, series.dt.tz_convert("UTC").astype("datetime64[ns, UTC]"))
    timestamps: list[pd.Timestamp] = []
    for value in series:
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise BacktestV1Error(f"invalid {name} timestamp") from exc
        if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset() is None:
            raise BacktestV1Error(f"{name} must contain valid timezone-aware timestamps")
        timestamps.append(stamp.tz_convert("UTC"))
    return pd.Series(pd.DatetimeIndex(timestamps, dtype="datetime64[ns, UTC]"), index=series.index)


def _numeric_column(frame: pd.DataFrame, name: str, *, positive: bool = False) -> None:
    try:
        values = pd.to_numeric(frame[name], errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise BacktestV1Error(f"{name} must be numeric") from exc
    if not np.isfinite(values.to_numpy()).all():
        raise BacktestV1Error(f"{name} must contain finite values")
    if positive and values.le(0.0).any():
        raise BacktestV1Error(f"{name} must contain positive values")
    frame[name] = values


def _validate(records: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(records, pd.DataFrame):
        raise BacktestV1Error("records must be a pandas DataFrame")
    if not records.columns.is_unique:
        raise BacktestV1Error("record columns must be unique")
    missing = sorted(set(INPUT_COLUMNS) - set(records.columns))
    if missing:
        raise BacktestV1Error(f"records miss columns: {', '.join(missing)}")
    frame = records.copy(deep=True)
    for name in _TIME_COLUMNS:
        frame[name] = _utc_column(frame[name], name)
    if frame["instrument"].map(lambda x: not isinstance(x, str) or not x.strip()).any():
        raise BacktestV1Error("instrument must contain non-empty strings")
    for name in ("entry_bid_open", "exit_bid_open", "horizon_minutes"):
        _numeric_column(frame, name, positive=True)
    _numeric_column(frame, "expected_return_bps")
    for name in ("entry_ask_open", "exit_ask_open"):
        if name in frame:
            _numeric_column(frame, name, positive=True)
            if frame[name].lt(frame[name.replace("ask", "bid")]).any():
                raise BacktestV1Error(f"{name} must be greater than or equal to its bid")
    if not frame["horizon_minutes"].mod(1.0).eq(0.0).all():
        raise BacktestV1Error("horizon_minutes must contain positive integers")
    frame["horizon_minutes"] = frame["horizon_minutes"].astype("int64")
    if frame.empty:
        return frame
    try:
        PredictionBatch(
            predicted_class=tuple(frame["predicted_class"]),
            probabilities=frame.loc[:, list(_PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64),
        )
    except (ClassificationContractError, TypeError, ValueError) as exc:
        raise BacktestV1Error(f"invalid prediction contract: {exc}") from exc
    if not frame["entry_time_utc"].gt(frame["prediction_time_utc"]).all():
        raise BacktestV1Error("entry time must be strictly after prediction time")
    duration = frame["label_end_time_utc"] - frame["entry_time_utc"]
    if not duration.eq(pd.to_timedelta(frame["horizon_minutes"], unit="min")).all():
        raise BacktestV1Error("label interval must equal horizon_minutes after entry")
    keys = ["instrument", "prediction_time_utc", "horizon_minutes"]
    if frame.duplicated(keys).any():
        raise BacktestV1Error("duplicate instrument/prediction/horizon records")
    frame = frame.sort_values(
        ["prediction_time_utc", "instrument", "horizon_minutes"], kind="stable"
    ).reset_index(drop=True)
    for _, group in frame.groupby("instrument", sort=False):
        if not group["entry_time_utc"].is_monotonic_increasing:
            raise BacktestV1Error("entry times must preserve prediction order per instrument")
    return frame


def _trade_summary(trades: pd.DataFrame) -> dict[str, object]:
    ordered = trades.sort_values(["exit_fill_time_utc", "instrument"], kind="stable")
    net = ordered["net_return_bps"].to_numpy(dtype=np.float64)
    gross = ordered["gross_return_bps"].to_numpy(dtype=np.float64)
    # Concurrent exits on different instruments realize together. Arbitrary
    # alphabetic ordering must not create fictitious within-timestamp drawdown.
    timestamp_net = ordered.groupby("exit_fill_time_utc", sort=True)["net_return_bps"].sum()
    cumulative = np.concatenate(([0.0], timestamp_net.to_numpy(dtype=np.float64).cumsum()))
    gains = float(net[net > 0.0].sum())
    losses = float(-net[net < 0.0].sum())
    return {
        "executed_trade_count": len(trades),
        "cumulative_gross_return_bps": float(gross.sum()),
        "cumulative_net_return_bps": float(net.sum()),
        "cumulative_cost_bps": float(trades["cost_bps"].sum()),
        "max_drawdown_bps": float(np.max(np.maximum.accumulate(cumulative) - cumulative)),
        "hit_rate": float((net > 0.0).mean()) if len(net) else None,
        "average_gross_return_bps": float(gross.mean()) if len(net) else None,
        "average_net_return_bps": float(net.mean()) if len(net) else None,
        "profit_factor": gains / losses if losses else None,
        "turnover": float(2 * len(trades)),
        "exposure_minutes": float(trades["horizon_minutes"].sum()),
    }


def _metrics(decisions: pd.DataFrame, trades: pd.DataFrame) -> dict[str, object]:
    result = _trade_summary(trades)
    signal = decisions["decision_status"].isin(["executed", "suppressed_overlap"])
    observation_minutes = 0.0
    for _, group in decisions.groupby("instrument", sort=True):
        observation_minutes += (
            group["label_end_time_utc"].max() - group["prediction_time_utc"].min()
        ).total_seconds() / 60.0
    exposure_minutes = float(trades["horizon_minutes"].sum())
    result.update(
        {
            "record_count": len(decisions),
            "signal_count": int(signal.sum()),
            "no_signal_count": int((~signal).sum()),
            "suppressed_overlap_count": int(
                decisions["decision_status"].eq("suppressed_overlap").sum()
            ),
            "decision_counts": {
                str(k): int(v) for k, v in decisions["decision_status"].value_counts().items()
            },
            "observation_instrument_minutes": observation_minutes,
            "exposure_fraction": exposure_minutes / observation_minutes
            if observation_minutes
            else 0.0,
            "return_convention": "arithmetic_fixed_entry_bid_notional_bps_not_account_return",
            "drawdown_convention": "realized_exit_pnl_only_not_intratrade_mark_to_market",
            "position_policy": "single_non_overlapping_per_instrument_shortest_horizon_tie_break",
            "session_convention": "fixed_UTC_hour_buckets_not_DST_adjusted_market_sessions",
        }
    )
    for column in ("utc_session", "utc_hour", "horizon_minutes", "fold", "instrument"):
        if column in trades:
            result[f"by_{column}"] = {
                str(key): _trade_summary(group)
                for key, group in trades.groupby(column, sort=True, dropna=False)
            }
    return result


_DEFAULT_COSTS = BacktestCosts()
_DEFAULT_POLICY = DecisionPolicy()


def run_backtest_v1(
    records: pd.DataFrame,
    *,
    costs: BacktestCosts = _DEFAULT_COSTS,
    policy: DecisionPolicy = _DEFAULT_POLICY,
    decision_costs: BacktestCosts | None = None,
) -> BacktestV1Result:
    """Evaluate frozen decisions against executable-side historical prices.

    Entry is strictly later than prediction; signal and order are assumed to
    be generated instantaneously at prediction. All configured latency is
    represented by the entry timestamp. Longs buy ask and sell bid; shorts
    sell bid and buy ask. Adverse slippage adjusts both fills. Commission is
    charged twice on the original normalized bid notional. No additional
    constant spread is subtracted from the price-based return.

    A directional probability must pass confidence, and its signed expected
    bid return must exceed estimated cost plus minimum edge (strictly).
    Estimation uses configured costs, never realized spreads/price outcomes.
    Decisions may combine horizons, but should represent a single strategy.
    Supply base ``decision_costs`` with stressed execution ``costs`` to retain
    the original decision set while testing worse fills on those same trades.
    """
    frame = _validate(records)
    spread_fraction = costs.spread_bps / 10_000.0
    slippage_fraction = costs.slippage_per_side_bps / 10_000.0
    estimated_cost = (decision_costs or costs).estimated_round_trip_bps
    confidence = frame.loc[:, list(_PROBABILITY_COLUMNS)].max(axis=1)
    direction = frame["predicted_class"].map({"up": 1.0, "down": -1.0, "neutral": 0.0})
    expected_directional = direction * frame["expected_return_bps"]
    expected_net = expected_directional - estimated_cost
    status = np.select(
        [
            frame["predicted_class"].eq("neutral"),
            confidence.lt(policy.confidence_threshold),
            expected_directional.le(0.0),
            expected_net.le(policy.min_expected_net_bps),
        ],
        [
            "no_signal_neutral",
            "no_signal_low_confidence",
            "no_signal_incompatible_expected_return",
            "no_signal_insufficient_edge",
        ],
        default="executed",
    )
    # Only eligible signals need a sequential position state. Work on integer
    # nanosecond arrays to avoid constructing a Timestamp/dict for every row.
    entry_ns = frame["entry_time_utc"].astype("int64").to_numpy()
    end_ns = frame["label_end_time_utc"].astype("int64").to_numpy()
    instruments = frame["instrument"].to_numpy()
    active_until: dict[str, int] = {}
    signal = status == "executed"
    for position in np.flatnonzero(signal):
        instrument = str(instruments[position])
        if instrument in active_until and entry_ns[position] < active_until[instrument]:
            status[position] = "suppressed_overlap"
        else:
            active_until[instrument] = int(end_ns[position])
    executed = status == "executed"
    entry_bid = frame["entry_bid_open"]
    exit_bid = frame["exit_bid_open"]
    entry_observed = "entry_ask_open" in frame
    exit_observed = "exit_ask_open" in frame
    entry_ask = frame["entry_ask_open"] if entry_observed else entry_bid * (1 + spread_fraction)
    exit_ask = frame["exit_ask_open"] if exit_observed else exit_bid * (1 + spread_fraction)
    is_long = direction.gt(0.0)
    entry_side = entry_ask.where(is_long, entry_bid)
    exit_side = exit_bid.where(is_long, exit_ask)
    entry_fill = entry_side * (1.0 + direction * slippage_fraction)
    exit_fill = exit_side * (1.0 - direction * slippage_fraction)
    spread_cost = (entry_ask - entry_bid).where(is_long, exit_ask - exit_bid) / entry_bid * 10000
    slip_cost = (entry_side + exit_side) * slippage_fraction / entry_bid * 10000
    gross = direction * (exit_bid - entry_bid) / entry_bid * 10000
    commission = costs.commission_per_side_bps * 2.0
    net = direction * (exit_fill - entry_fill) / entry_bid * 10000 - commission
    hour = frame["entry_time_utc"].dt.hour
    audit = pd.DataFrame(
        {
            "signal_time_utc": frame["prediction_time_utc"].where(signal),
            "order_time_utc": frame["prediction_time_utc"].where(executed),
            "fill_time_utc": frame["entry_time_utc"].where(executed),
            "exit_fill_time_utc": frame["label_end_time_utc"].where(executed),
            "modeled_latency_seconds": (
                frame["entry_time_utc"] - frame["prediction_time_utc"]
            ).dt.total_seconds(),
            "signal_confidence": confidence,
            "side": frame["predicted_class"].map({"up": "long", "down": "short"}),
            "decision_status": status,
            "estimated_cost_bps": estimated_cost,
            "expected_directional_return_bps": expected_directional,
            "expected_net_return_bps": expected_net,
            "entry_ask_basis": "observed" if entry_observed else "constant_spread_proxy",
            "exit_ask_basis": "observed" if exit_observed else "constant_spread_proxy",
            "entry_fill_price": entry_fill.where(executed),
            "exit_fill_price": exit_fill.where(executed),
            "gross_return_bps": gross.where(executed),
            "spread_cost_bps": spread_cost.where(executed),
            "slippage_cost_bps": slip_cost.where(executed),
            "commission_cost_bps": np.where(executed, commission, np.nan),
            "cost_bps": (spread_cost + slip_cost + commission).where(executed),
            "net_return_bps": net.where(executed),
            "utc_hour": hour,
            "utc_session": np.select(
                [hour.lt(7), hour.lt(13), hour.lt(21)],
                ["00-07_UTC", "07-13_UTC", "13-21_UTC"],
                default="21-24_UTC",
            ),
        },
        index=frame.index,
    )
    frame["entry_ask_open"] = entry_ask
    frame["exit_ask_open"] = exit_ask
    decisions = pd.concat(
        [frame.drop(columns=list(_RESULT_COLUMNS), errors="ignore"), audit], axis=1
    )
    trades = decisions.loc[executed].reset_index(drop=True)
    return BacktestV1Result(decisions=decisions, trades=trades, metrics=_metrics(decisions, trades))


__all__ = [
    "INPUT_COLUMNS",
    "BacktestCosts",
    "BacktestV1Error",
    "BacktestV1Result",
    "DecisionPolicy",
    "run_backtest_v1",
]
