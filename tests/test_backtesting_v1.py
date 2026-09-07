"""Hand-calculated execution P&L and causal decision contract checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold_forecasting.backtesting.v1 import (
    BacktestCosts,
    BacktestV1Error,
    DecisionPolicy,
    run_backtest_v1,
)


def _records(*, label: str = "up", count: int = 1) -> pd.DataFrame:
    times = pd.date_range("2024-01-02T08:00:00Z", periods=count, freq="20min")
    probabilities = {"up": (0.1, 0.1, 0.8), "down": (0.8, 0.1, 0.1), "neutral": (0.1, 0.8, 0.1)}
    down, neutral, up = probabilities[label]
    return pd.DataFrame(
        {
            "instrument": "XAU/USD",
            "prediction_time_utc": times,
            "entry_time_utc": times + pd.Timedelta(minutes=1),
            "label_end_time_utc": times + pd.Timedelta(minutes=16),
            "entry_bid_open": 100.0,
            "exit_bid_open": 101.0,
            "predicted_class": label,
            "p_down": down,
            "p_neutral": neutral,
            "p_up": up,
            "expected_return_bps": -10.0 if label == "down" else 10.0,
            "horizon_minutes": 15,
            "fold": "2024",
        }
    )


@pytest.mark.parametrize("label,direction", [("up", 1.0), ("down", -1.0)])
def test_proxy_prices_costs_exactly_once(label: str, direction: float) -> None:
    result = run_backtest_v1(_records(label=label), costs=BacktestCosts(3, 0.5, 0.2))
    trade = result.trades.iloc[0]
    if label == "up":
        expected_entry = 100.03 * 1.00005
        expected_exit = 101.0 * 0.99995
        expected_spread = 3.0
    else:
        expected_entry = 100.0 * 0.99995
        expected_exit = 101.0303 * 1.00005
        expected_spread = 3.03
    expected_net = direction * (expected_exit - expected_entry) * 100.0 - 0.4
    assert trade["entry_fill_price"] == pytest.approx(expected_entry)
    assert trade["exit_fill_price"] == pytest.approx(expected_exit)
    assert trade["spread_cost_bps"] == pytest.approx(expected_spread)
    assert trade["net_return_bps"] == pytest.approx(expected_net)
    assert trade["gross_return_bps"] == pytest.approx(direction * 100.0)
    assert trade["gross_return_bps"] - trade["net_return_bps"] == pytest.approx(trade["cost_bps"])
    assert trade["entry_ask_basis"] == "constant_spread_proxy"
    assert trade["exit_ask_basis"] == "constant_spread_proxy"
    assert trade["estimated_cost_bps"] == pytest.approx(4.4)


@pytest.mark.parametrize("label,expected_net", [("up", 90.0), ("down", -130.0)])
def test_observed_asks_replace_proxy_spread(label: str, expected_net: float) -> None:
    frame = _records(label=label).assign(entry_ask_open=100.10, exit_ask_open=101.30)
    trade = run_backtest_v1(frame, costs=BacktestCosts(3, 0, 0)).trades.iloc[0]
    assert trade["net_return_bps"] == pytest.approx(expected_net)
    assert trade["entry_ask_basis"] == "observed"
    assert trade["exit_ask_basis"] == "observed"
    assert trade["slippage_cost_bps"] == 0.0
    assert trade["commission_cost_bps"] == 0.0


def test_future_quote_changes_never_change_signal_gating() -> None:
    original = _records(count=2)
    changed = original.assign(exit_bid_open=[1.0, 10000.0], exit_ask_open=[9.0, 20000.0])
    original_result = run_backtest_v1(original)
    changed_result = run_backtest_v1(changed)
    for column in ("decision_status", "expected_net_return_bps", "estimated_cost_bps"):
        pd.testing.assert_series_equal(
            original_result.decisions[column], changed_result.decisions[column]
        )
    assert not original_result.trades["net_return_bps"].equals(
        changed_result.trades["net_return_bps"]
    )


def test_stress_execution_preserves_base_decisions_and_input() -> None:
    frame = _records(count=3).assign(expected_return_bps=[4.1, 5.0, 10.0])
    snapshot = frame.copy(deep=True)
    base_costs = BacktestCosts()
    base = run_backtest_v1(frame, costs=base_costs)
    stress = run_backtest_v1(frame, costs=BacktestCosts(spread_bps=4.5), decision_costs=base_costs)
    pd.testing.assert_frame_equal(frame, snapshot)
    for name in ("decision_status", "fill_time_utc", "exit_fill_time_utc"):
        pd.testing.assert_series_equal(base.decisions[name], stress.decisions[name])
    assert len(stress.trades) == 3
    assert stress.trades["net_return_bps"].lt(base.trades["net_return_bps"]).all()
    assert stress.trades["estimated_cost_bps"].eq(4.0).all()


def test_datetime_resolution_and_timezones_normalize_consistently() -> None:
    frame = _records(count=2)
    expected = run_backtest_v1(frame)
    for name in ("prediction_time_utc", "entry_time_utc", "label_end_time_utc"):
        frame[name] = frame[name].dt.tz_convert("Europe/Brussels")
    frame["entry_time_utc"] = frame["entry_time_utc"].astype("datetime64[us, Europe/Brussels]")
    actual = run_backtest_v1(frame)
    pd.testing.assert_frame_equal(actual.decisions, expected.decisions)


def test_timestamps_latency_and_metadata_are_auditable() -> None:
    frame = _records().assign(model="linear")
    result = run_backtest_v1(frame)
    trade = result.trades.iloc[0]
    assert trade["signal_time_utc"] == frame.iloc[0]["prediction_time_utc"]
    assert trade["order_time_utc"] == frame.iloc[0]["prediction_time_utc"]
    assert trade["fill_time_utc"] == frame.iloc[0]["entry_time_utc"]
    assert trade["exit_fill_time_utc"] == frame.iloc[0]["label_end_time_utc"]
    assert trade["modeled_latency_seconds"] == 60.0
    assert trade["model"] == "linear"
    assert trade["utc_session"] == "07-13_UTC"
    assert trade["utc_hour"] == 8
    assert result.metrics["exposure_fraction"] == pytest.approx(15 / 16)
    assert result.metrics["turnover"] == 2.0
    assert result.metrics["by_fold"]["2024"]["executed_trade_count"] == 1  # type: ignore[index]


def test_nonoverlap_per_instrument_includes_all_horizons() -> None:
    frame = _records(count=4)
    times = pd.to_datetime(
        ["2024-01-02T08:00Z", "2024-01-02T08:03Z", "2024-01-02T08:15Z", "2024-01-02T08:03Z"]
    )
    frame["prediction_time_utc"] = times
    frame["entry_time_utc"] = times + pd.Timedelta(minutes=1)
    frame["horizon_minutes"] = [15, 3, 15, 3]
    frame["label_end_time_utc"] = frame["entry_time_utc"] + pd.to_timedelta(
        frame["horizon_minutes"], unit="min"
    )
    frame.loc[3, "instrument"] = "OTHER"
    decisions = run_backtest_v1(frame).decisions
    gold = decisions.loc[decisions["instrument"].eq("XAU/USD")]
    assert gold["decision_status"].tolist() == ["executed", "suppressed_overlap", "executed"]
    assert decisions.loc[decisions["instrument"].eq("OTHER"), "decision_status"].tolist() == [
        "executed"
    ]


def test_simultaneous_horizons_are_deterministic_shortest_first() -> None:
    long = _records()
    short = long.assign(horizon_minutes=3)
    short["label_end_time_utc"] = short["entry_time_utc"] + pd.Timedelta(minutes=3)
    first = run_backtest_v1(pd.concat([long, short], ignore_index=True))
    second = run_backtest_v1(pd.concat([short, long], ignore_index=True))
    pd.testing.assert_frame_equal(first.decisions, second.decisions)
    assert first.trades["horizon_minutes"].tolist() == [3]


@pytest.mark.parametrize(
    "label,expected,threshold,edge,status",
    [
        ("neutral", 10, 0.5, 0.0, "no_signal_neutral"),
        ("up", 10, 0.81, 0.0, "no_signal_low_confidence"),
        ("up", -10, 0.5, 0.0, "no_signal_incompatible_expected_return"),
        ("down", 10, 0.5, 0.0, "no_signal_incompatible_expected_return"),
        ("down", 0, 0.5, 0.0, "no_signal_incompatible_expected_return"),
        ("up", 4, 0.5, 0.0, "no_signal_insufficient_edge"),
        ("up", 6, 0.5, 2.0, "no_signal_insufficient_edge"),
        ("up", 6.0001, 0.8, 2.0, "executed"),
    ],
)
def test_frozen_policy_thresholds(
    label: str, expected: float, threshold: float, edge: float, status: str
) -> None:
    frame = _records(label=label).assign(expected_return_bps=expected)
    result = run_backtest_v1(frame, policy=DecisionPolicy(threshold, edge))
    assert result.decisions.iloc[0]["decision_status"] == status
    if status != "executed":
        assert result.trades.empty
        assert result.metrics["cumulative_net_return_bps"] == 0.0
        assert result.metrics["max_drawdown_bps"] == 0.0
        assert result.metrics["hit_rate"] is None
        assert result.decisions["order_time_utc"].isna().all()


def test_empty_records_have_complete_empty_result_schema() -> None:
    result = run_backtest_v1(_records().iloc[:0])
    assert result.trades.empty
    assert result.decisions.empty
    assert "net_return_bps" in result.trades
    assert result.metrics["record_count"] == 0
    assert result.metrics["exposure_fraction"] == 0.0


def test_drawdown_is_exit_order_realized_fixed_notional_not_account_return() -> None:
    frame = _records(count=3).assign(exit_bid_open=[101, 98, 101])
    result = run_backtest_v1(frame, costs=BacktestCosts(0, 0, 0))
    assert result.metrics["cumulative_gross_return_bps"] == pytest.approx(0.0)
    assert result.metrics["cumulative_net_return_bps"] == pytest.approx(0.0)
    assert result.metrics["max_drawdown_bps"] == pytest.approx(200.0)
    assert result.metrics["hit_rate"] == pytest.approx(2 / 3)
    assert result.metrics["profit_factor"] == pytest.approx(1.0)
    assert result.metrics["turnover"] == 6.0
    assert "not_account_return" in str(result.metrics["return_convention"])


def test_simultaneous_instrument_exits_do_not_add_fictitious_drawdown() -> None:
    gain = _records().assign(instrument="B", exit_bid_open=101.0)
    loss = _records().assign(instrument="A", exit_bid_open=99.0)
    result = run_backtest_v1(
        pd.concat([gain, loss], ignore_index=True), costs=BacktestCosts(0, 0, 0)
    )
    assert result.metrics["max_drawdown_bps"] == pytest.approx(0.0)
    assert result.metrics["cumulative_net_return_bps"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "column,value,match",
    [
        ("entry_bid_open", 0, "positive"),
        ("exit_bid_open", -1, "positive"),
        ("entry_bid_open", np.nan, "finite"),
        ("exit_bid_open", np.inf, "finite"),
        ("expected_return_bps", "bad", "numeric"),
        ("expected_return_bps", np.nan, "finite"),
        ("entry_ask_open", 99.9, "greater than"),
        ("exit_ask_open", 100.9, "greater than"),
        ("horizon_minutes", 1.5, "integers"),
        ("horizon_minutes", 12, "interval"),
        ("predicted_class", "down", "argmax"),
        ("p_up", 0.7, "sum"),
        ("p_down", np.nan, "finite"),
        ("instrument", "", "non-empty"),
        ("prediction_time_utc", "2024-01-02 08:00:00", "timezone-aware"),
        ("prediction_time_utc", "bad", "invalid"),
        ("entry_time_utc", "2024-01-02T08:00:00Z", "strictly after"),
    ],
)
def test_invalid_inputs_are_rejected(column: str, value: object, match: str) -> None:
    frame = _records()
    frame[column] = value
    with pytest.raises(BacktestV1Error, match=match):
        run_backtest_v1(frame)


def test_duplicate_keys_and_missing_columns_rejected() -> None:
    with pytest.raises(BacktestV1Error, match="duplicate"):
        run_backtest_v1(pd.concat([_records(), _records()], ignore_index=True))
    with pytest.raises(BacktestV1Error, match="miss columns"):
        run_backtest_v1(_records().drop(columns="entry_bid_open"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spread_bps": -1},
        {"slippage_per_side_bps": 10000},
        {"commission_per_side_bps": np.nan},
        {"spread_bps": True},
    ],
)
def test_invalid_costs(kwargs: dict[str, float]) -> None:
    with pytest.raises(BacktestV1Error):
        BacktestCosts(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence_threshold": 1.1},
        {"confidence_threshold": np.nan},
        {"min_expected_net_bps": -1},
    ],
)
def test_invalid_policy(kwargs: dict[str, float]) -> None:
    with pytest.raises(BacktestV1Error):
        DecisionPolicy(**kwargs)
