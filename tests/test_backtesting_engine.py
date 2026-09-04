from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gold_forecasting.backtesting import BacktestError, run_mvp_backtest


def _row(
    prediction_minute: int,
    *,
    predicted_class: str = "up",
    probabilities: tuple[float, float, float] = (0.1, 0.1, 0.8),
    future_return_bps: float = 10.0,
    entry_minute: int | None = None,
    exit_minute: int | None = None,
    instrument: str = "XAU_USD",
) -> dict[str, object]:
    origin = pd.Timestamp("2024-01-02T00:00:00Z")
    entry = prediction_minute + 1 if entry_minute is None else entry_minute
    exit_ = entry + 15 if exit_minute is None else exit_minute
    return {
        "instrument": instrument,
        "prediction_time_utc": origin + pd.Timedelta(minutes=prediction_minute),
        "entry_time_utc": origin + pd.Timedelta(minutes=entry),
        "label_end_time_utc": origin + pd.Timedelta(minutes=exit_),
        "predicted_class": predicted_class,
        "p_down": probabilities[0],
        "p_neutral": probabilities[1],
        "p_up": probabilities[2],
        "future_return_bps": future_return_bps,
    }


def _frame(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_long_short_neutral_and_low_confidence_have_correct_costed_results() -> None:
    records = _frame(
        [
            _row(0, future_return_bps=10.0),
            _row(
                20,
                predicted_class="down",
                probabilities=(0.7, 0.2, 0.1),
                future_return_bps=-12.0,
            ),
            _row(
                40,
                predicted_class="neutral",
                probabilities=(0.1, 0.8, 0.1),
            ),
            _row(60, probabilities=(0.2, 0.3, 0.5)),
        ]
    )

    result = run_mvp_backtest(
        records,
        confidence_threshold=0.6,
        round_trip_cost_bps=4.0,
    )

    assert result.decisions["decision_status"].tolist() == [
        "executed",
        "executed",
        "no_signal_neutral",
        "no_signal_low_confidence",
    ]
    assert result.trades["side"].tolist() == ["long", "short"]
    assert result.trades["gross_return_bps"].tolist() == [10.0, 12.0]
    assert result.trades["net_return_bps"].tolist() == [6.0, 8.0]
    assert result.trades["cost_bps"].tolist() == [4.0, 4.0]
    assert result.metrics.record_count == 4
    assert result.metrics.signal_count == 2
    assert result.metrics.executed_trade_count == 2
    assert result.metrics.no_signal_count == 2
    assert result.metrics.neutral_prediction_count == 1
    assert result.metrics.low_confidence_count == 1
    assert result.metrics.suppressed_overlap_count == 0
    assert result.metrics.hit_rate == 1.0
    assert result.metrics.average_gross_return_bps == 11.0
    assert result.metrics.average_net_return_bps == 7.0
    assert result.metrics.cumulative_gross_return_bps == 22.0
    assert result.metrics.cumulative_net_return_bps == 14.0
    assert result.metrics.max_drawdown_bps == 0.0
    assert result.metrics.turnover == 4.0


def test_single_position_policy_suppresses_overlap_and_allows_boundary_entry() -> None:
    records = _frame(
        [
            _row(0, entry_minute=1, exit_minute=16),
            _row(3, entry_minute=4, exit_minute=19),
            _row(15, entry_minute=16, exit_minute=31),
        ]
    )

    result = run_mvp_backtest(
        records,
        confidence_threshold=0.5,
        round_trip_cost_bps=4.0,
    )

    assert result.decisions["decision_status"].tolist() == [
        "executed",
        "suppressed_overlap",
        "executed",
    ]
    assert result.metrics.signal_count == 3
    assert result.metrics.executed_trade_count == 2
    assert result.metrics.suppressed_overlap_count == 1
    assert result.metrics.no_signal_count == 0


def test_max_drawdown_includes_initial_zero_equity_and_uses_net_returns() -> None:
    records = _frame(
        [
            _row(0, future_return_bps=10.0),
            _row(20, future_return_bps=-6.0),
            _row(
                40,
                predicted_class="down",
                probabilities=(0.8, 0.1, 0.1),
                future_return_bps=8.0,
            ),
        ]
    )

    result = run_mvp_backtest(
        records,
        confidence_threshold=0.8,
        round_trip_cost_bps=4.0,
        normalized_notional=1.5,
    )

    assert result.trades["net_return_bps"].tolist() == [6.0, -10.0, -12.0]
    assert result.metrics.hit_rate == pytest.approx(1 / 3)
    assert result.metrics.cumulative_gross_return_bps == -4.0
    assert result.metrics.cumulative_net_return_bps == -16.0
    assert result.metrics.max_drawdown_bps == 22.0
    assert result.metrics.turnover == 9.0


def test_all_no_signal_metrics_are_explicitly_undefined_or_zero() -> None:
    records = _frame(
        [
            _row(
                0,
                predicted_class="neutral",
                probabilities=(0.1, 0.8, 0.1),
            )
        ]
    )

    result = run_mvp_backtest(
        records,
        confidence_threshold=0.5,
        round_trip_cost_bps=4.0,
    )

    assert result.trades.empty
    assert result.metrics.hit_rate is None
    assert result.metrics.average_gross_return_bps is None
    assert result.metrics.average_net_return_bps is None
    assert result.metrics.cumulative_gross_return_bps == 0.0
    assert result.metrics.cumulative_net_return_bps == 0.0
    assert result.metrics.max_drawdown_bps == 0.0
    assert result.metrics.turnover == 0.0


def test_threshold_is_inclusive_and_input_is_sorted_without_mutation() -> None:
    records = _frame(
        [
            _row(20, probabilities=(0.1, 0.1, 0.8)),
            _row(0, probabilities=(0.1, 0.1, 0.8)),
        ]
    )
    original = records.copy(deep=True)

    result = run_mvp_backtest(
        records,
        confidence_threshold=0.8,
        round_trip_cost_bps=0.0,
    )

    assert result.metrics.executed_trade_count == 2
    assert result.decisions["prediction_time_utc"].is_monotonic_increasing
    assert_frame_equal(records, original)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("entry_time_utc", pd.Timestamp("2024-01-02T00:00:00Z"), "strictly after"),
        ("entry_time_utc", pd.Timestamp("2024-01-01T23:59:00Z"), "strictly after"),
        ("label_end_time_utc", pd.Timestamp("2024-01-02T00:01:00Z"), "exit time"),
        ("prediction_time_utc", pd.Timestamp("2024-01-02T00:00:00"), "timezone-aware"),
        ("future_return_bps", np.nan, "finite"),
        ("p_up", 0.7, "sum to one"),
        ("predicted_class", "down", "argmax"),
    ],
)
def test_invalid_record_values_fail_closed(column: str, value: object, message: str) -> None:
    records = _frame([_row(0)])
    if column.endswith("_utc"):
        records[column] = records[column].astype("object")
    records.loc[0, column] = value

    with pytest.raises(BacktestError, match=message):
        run_mvp_backtest(
            records,
            confidence_threshold=0.5,
            round_trip_cost_bps=4.0,
        )


def test_duplicate_predictions_multiple_instruments_and_noncausal_entry_order_fail() -> None:
    duplicate = _frame([_row(0), _row(0)])
    with pytest.raises(BacktestError, match="duplicate prediction"):
        run_mvp_backtest(duplicate, confidence_threshold=0.5, round_trip_cost_bps=4.0)

    multiple = _frame([_row(0), _row(20, instrument="EUR_USD")])
    with pytest.raises(BacktestError, match="exactly one instrument"):
        run_mvp_backtest(multiple, confidence_threshold=0.5, round_trip_cost_bps=4.0)

    noncausal_order = _frame(
        [
            _row(0, entry_minute=10, exit_minute=25),
            _row(3, entry_minute=4, exit_minute=19),
        ]
    )
    with pytest.raises(BacktestError, match="preserve prediction-time order"):
        run_mvp_backtest(noncausal_order, confidence_threshold=0.5, round_trip_cost_bps=4.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"confidence_threshold": -0.1}, "confidence_threshold"),
        ({"confidence_threshold": 1.1}, "confidence_threshold"),
        ({"confidence_threshold": np.nan}, "confidence_threshold"),
        ({"round_trip_cost_bps": -0.1}, "round_trip_cost_bps"),
        ({"round_trip_cost_bps": np.inf}, "round_trip_cost_bps"),
        ({"normalized_notional": 0.0}, "normalized_notional"),
    ],
)
def test_invalid_policy_arguments_fail_closed(
    kwargs: dict[str, float],
    message: str,
) -> None:
    arguments = {
        "confidence_threshold": 0.5,
        "round_trip_cost_bps": 4.0,
        "normalized_notional": 1.0,
    }
    arguments.update(kwargs)

    with pytest.raises(BacktestError, match=message):
        run_mvp_backtest(_frame([_row(0)]), **arguments)


def test_missing_columns_and_empty_input_are_rejected() -> None:
    records = _frame([_row(0)]).drop(columns="p_down")
    with pytest.raises(BacktestError, match="miss columns: p_down"):
        run_mvp_backtest(records, confidence_threshold=0.5, round_trip_cost_bps=4.0)

    with pytest.raises(BacktestError, match="at least one"):
        run_mvp_backtest(pd.DataFrame(), confidence_threshold=0.5, round_trip_cost_bps=4.0)
