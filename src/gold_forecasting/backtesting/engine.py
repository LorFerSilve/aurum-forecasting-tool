"""A deliberately small, costs-aware backtester for the research MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd

from gold_forecasting.classification import ClassificationContractError, PredictionBatch

TradeSide: TypeAlias = Literal["long", "short"]
DecisionStatus: TypeAlias = Literal[
    "executed",
    "no_signal_neutral",
    "no_signal_low_confidence",
    "suppressed_overlap",
]

BACKTEST_INPUT_COLUMNS = (
    "instrument",
    "prediction_time_utc",
    "entry_time_utc",
    "label_end_time_utc",
    "predicted_class",
    "p_down",
    "p_neutral",
    "p_up",
    "future_return_bps",
)

_PROBABILITY_COLUMNS = ("p_down", "p_neutral", "p_up")


class BacktestError(ValueError):
    """Raised when inputs cannot support a causal and auditable backtest."""


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """Aggregate MVP metrics; returns and drawdown are expressed in bps.

    ``turnover`` is the sum of entry and exit notionals.  With the default
    normalized notional of one, every executed round trip contributes 2.0.
    Rates and averages are ``None`` when there are no executed trades because
    those quantities are mathematically undefined in that case.
    """

    record_count: int
    signal_count: int
    executed_trade_count: int
    no_signal_count: int
    neutral_prediction_count: int
    low_confidence_count: int
    suppressed_overlap_count: int
    hit_rate: float | None
    average_gross_return_bps: float | None
    average_net_return_bps: float | None
    cumulative_gross_return_bps: float
    cumulative_net_return_bps: float
    max_drawdown_bps: float
    turnover: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Decision audit trail, executed trades, and aggregate metrics."""

    decisions: pd.DataFrame
    trades: pd.DataFrame
    metrics: BacktestMetrics
    confidence_threshold: float
    round_trip_cost_bps: float
    normalized_notional: float
    position_policy: Literal["single_non_overlapping"] = "single_non_overlapping"


def _finite_number(value: object, *, name: str, minimum: float) -> float:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise BacktestError(f"{name} must be numeric") from exc
    if not np.isfinite(numeric) or numeric < minimum:
        qualifier = "non-negative" if minimum == 0.0 else f"at least {minimum}"
        raise BacktestError(f"{name} must be finite and {qualifier}")
    return numeric


def _normalize_utc_column(frame: pd.DataFrame, column: str) -> pd.Series:
    normalized: list[pd.Timestamp] = []
    for index, raw in frame[column].items():
        try:
            timestamp = pd.Timestamp(raw)
        except (TypeError, ValueError) as exc:
            raise BacktestError(f"invalid {column} at index {index!r}") from exc
        if pd.isna(timestamp):
            raise BacktestError(f"invalid {column} at index {index!r}")
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise BacktestError(f"{column} must contain timezone-aware timestamps")
        normalized.append(timestamp.tz_convert("UTC"))
    return pd.Series(
        pd.DatetimeIndex(normalized, dtype="datetime64[ns, UTC]"),
        index=frame.index,
        name=column,
    )


def _validate_records(records: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(records, pd.DataFrame):
        raise BacktestError("records must be a pandas DataFrame")
    if records.empty:
        raise BacktestError("at least one backtest record is required")
    missing = sorted(set(BACKTEST_INPUT_COLUMNS).difference(records.columns))
    if missing:
        raise BacktestError(f"backtest records miss columns: {', '.join(missing)}")

    frame = records.copy(deep=True)
    for column in ("prediction_time_utc", "entry_time_utc", "label_end_time_utc"):
        frame[column] = _normalize_utc_column(frame, column)

    if frame["instrument"].map(lambda value: not isinstance(value, str) or not value.strip()).any():
        raise BacktestError("instrument values must be non-empty strings")
    instruments = set(frame["instrument"])
    if len(instruments) != 1:
        raise BacktestError("MVP backtests must contain exactly one instrument")

    try:
        probabilities = frame.loc[:, list(_PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
        labels = tuple(frame["predicted_class"].tolist())
        PredictionBatch(predicted_class=labels, probabilities=probabilities)
    except (ClassificationContractError, TypeError, ValueError) as exc:
        raise BacktestError(f"invalid prediction contract: {exc}") from exc

    try:
        future_returns = pd.to_numeric(frame["future_return_bps"], errors="raise").astype(
            "float64"
        )
    except (TypeError, ValueError) as exc:
        raise BacktestError("future_return_bps must be numeric") from exc
    if not np.isfinite(future_returns.to_numpy()).all():
        raise BacktestError("future_return_bps must contain only finite values")
    frame["future_return_bps"] = future_returns

    if not frame["entry_time_utc"].gt(frame["prediction_time_utc"]).all():
        raise BacktestError("every entry time must be strictly after prediction time")
    if not frame["label_end_time_utc"].gt(frame["entry_time_utc"]).all():
        raise BacktestError("every exit time must be strictly after entry time")
    duplicate_key = ["instrument", "prediction_time_utc"]
    if frame.duplicated(subset=duplicate_key).any():
        raise BacktestError("backtest records contain duplicate prediction keys")

    frame = frame.sort_values(duplicate_key, kind="stable").reset_index(drop=True)
    if not frame["entry_time_utc"].is_monotonic_increasing:
        raise BacktestError("entry times must preserve prediction-time order")
    return frame


def _maximum_drawdown(cumulative_net_bps: pd.Series) -> float:
    if cumulative_net_bps.empty:
        return 0.0
    equity = np.concatenate(([0.0], cumulative_net_bps.to_numpy(dtype=np.float64)))
    running_peak = np.maximum.accumulate(equity)
    return float(np.max(running_peak - equity))


def run_mvp_backtest(
    records: pd.DataFrame,
    *,
    confidence_threshold: float,
    round_trip_cost_bps: float,
    normalized_notional: float = 1.0,
) -> BacktestResult:
    """Run the causal MVP policy on aligned predictions and future outcomes.

    Directional predictions become signals only when their winning class
    probability is at least ``confidence_threshold``.  Neutral and
    below-threshold predictions do not signal.  Under the fixed
    ``single_non_overlapping`` policy, a new trade can enter exactly when the
    prior trade exits, but never before it exits.

    ``future_return_bps`` is the long return already calculated from the
    executable entry and exit bid opens.  Long trades retain its sign, short
    trades reverse it, and the fixed round-trip cost is subtracted exactly
    once from every executed trade.
    """

    threshold = _finite_number(
        confidence_threshold,
        name="confidence_threshold",
        minimum=0.0,
    )
    if threshold > 1.0:
        raise BacktestError("confidence_threshold must lie in [0, 1]")
    cost_bps = _finite_number(
        round_trip_cost_bps,
        name="round_trip_cost_bps",
        minimum=0.0,
    )
    notional = _finite_number(
        normalized_notional,
        name="normalized_notional",
        minimum=np.nextafter(0.0, 1.0),
    )
    frame = _validate_records(records)

    confidences = frame.loc[:, list(_PROBABILITY_COLUMNS)].max(axis=1)
    decisions: list[dict[str, object]] = []
    active_until: pd.Timestamp | None = None
    for position, (_, row) in enumerate(frame.iterrows()):
        predicted_class = str(row["predicted_class"])
        confidence = float(confidences.iloc[position])
        side: TradeSide | None = None
        status: DecisionStatus
        gross_return_bps: float | None = None
        net_return_bps: float | None = None

        if predicted_class == "neutral":
            status = "no_signal_neutral"
        elif confidence < threshold:
            status = "no_signal_low_confidence"
        else:
            side = "long" if predicted_class == "up" else "short"
            entry_time = row["entry_time_utc"]
            if active_until is not None and entry_time < active_until:
                status = "suppressed_overlap"
            else:
                status = "executed"
                direction = 1.0 if side == "long" else -1.0
                gross_return_bps = direction * float(row["future_return_bps"])
                net_return_bps = gross_return_bps - cost_bps
                active_until = row["label_end_time_utc"]

        decision = {column: row[column] for column in BACKTEST_INPUT_COLUMNS}
        decision.update(
            {
                "signal_confidence": confidence,
                "side": side,
                "decision_status": status,
                "gross_return_bps": gross_return_bps,
                "cost_bps": cost_bps if status == "executed" else None,
                "net_return_bps": net_return_bps,
            }
        )
        decisions.append(decision)

    decision_frame = pd.DataFrame(decisions)
    trade_frame = decision_frame.loc[
        decision_frame["decision_status"].eq("executed")
    ].reset_index(drop=True)
    signal_mask = decision_frame["decision_status"].isin(
        ["executed", "suppressed_overlap"]
    )
    executed_count = len(trade_frame)
    if executed_count:
        gross = trade_frame["gross_return_bps"].astype("float64")
        net = trade_frame["net_return_bps"].astype("float64")
        cumulative_net = net.cumsum()
        hit_rate: float | None = float(net.gt(0.0).mean())
        average_gross: float | None = float(gross.mean())
        average_net: float | None = float(net.mean())
        cumulative_gross = float(gross.sum())
        cumulative_net_return = float(net.sum())
        max_drawdown = _maximum_drawdown(cumulative_net)
    else:
        hit_rate = None
        average_gross = None
        average_net = None
        cumulative_gross = 0.0
        cumulative_net_return = 0.0
        max_drawdown = 0.0

    metrics = BacktestMetrics(
        record_count=len(decision_frame),
        signal_count=int(signal_mask.sum()),
        executed_trade_count=executed_count,
        no_signal_count=int((~signal_mask).sum()),
        neutral_prediction_count=int(
            decision_frame["decision_status"].eq("no_signal_neutral").sum()
        ),
        low_confidence_count=int(
            decision_frame["decision_status"].eq("no_signal_low_confidence").sum()
        ),
        suppressed_overlap_count=int(
            decision_frame["decision_status"].eq("suppressed_overlap").sum()
        ),
        hit_rate=hit_rate,
        average_gross_return_bps=average_gross,
        average_net_return_bps=average_net,
        cumulative_gross_return_bps=cumulative_gross,
        cumulative_net_return_bps=cumulative_net_return,
        max_drawdown_bps=max_drawdown,
        turnover=2.0 * notional * executed_count,
    )
    return BacktestResult(
        decisions=decision_frame,
        trades=trade_frame,
        metrics=metrics,
        confidence_threshold=threshold,
        round_trip_cost_bps=cost_bps,
        normalized_notional=notional,
    )


__all__ = [
    "BACKTEST_INPUT_COLUMNS",
    "BacktestError",
    "BacktestMetrics",
    "BacktestResult",
    "DecisionStatus",
    "TradeSide",
    "run_mvp_backtest",
]
