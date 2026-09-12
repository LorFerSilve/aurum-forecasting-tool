from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gold_forecasting.phase10.contracts import ContextSource
from gold_forecasting.phase10.dollar_features import (
    DOLLAR_FEATURE_NAMES,
    DOLLAR_MODEL_FEATURES,
    build_dollar_features,
)
from gold_forecasting.phase10.features import PRICE_FEATURE_NAMES, build_silver_features
from gold_forecasting.phase10.point_in_time import join_context


def _frame(periods: int = 60) -> pd.DataFrame:
    steps = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "prediction_time_utc": pd.date_range(
                "2024-01-02", periods=periods, freq="3min", tz="UTC"
            ),
            "gold_close": 2_000.0 * np.exp(np.cumsum(0.0001 + np.sin(steps) * 0.001)),
            "dollar_value": 1.10 * np.exp(np.cumsum(0.0002 + np.cos(steps) * 0.002)),
            "dollar_is_missing": False,
            "dollar_is_stale": False,
            "dollar_age_seconds": 60.0,
            "sample_id": [f"sample-{step}" for step in range(periods)],
        }
    )


def _source() -> ContextSource:
    return ContextSource(
        source_id="dollar",
        description="Synthetic EURUSD vintage stream for dollar feature tests",
        source_url="https://example.test/eurusd",
        market_hours="FX sessions",
        source_timezone="UTC",
        publication_delay_seconds=60,
        stale_after_seconds=600,
        availability_basis="synthetic",
        revision_policy="vintages",
        availability_evidence="Explicit synthetic release timestamps",
        enabled=True,
    )


def _observations(frame: pd.DataFrame) -> pd.DataFrame:
    available = frame["prediction_time_utc"].copy()
    return pd.DataFrame(
        {
            "source_id": "dollar",
            "observation_id": [f"quote-{step}" for step in range(len(frame))],
            "observed_at_utc": available - pd.Timedelta(seconds=60),
            "available_at_utc": available,
            "ingested_at_utc": pd.Timestamp("2024-12-31", tz="UTC"),
            "value": frame["dollar_value"],
            "revision_id": "v1",
            "source_uri": "https://example.test/eurusd/raw.csv",
            "raw_sha256": "a" * 64,
        }
    )


def _predictions(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ["prediction_time_utc", "gold_close", "sample_id"]]


def test_frozen_contract_and_dollar_strength_sign() -> None:
    assert DOLLAR_MODEL_FEATURES == (
        "dollar_return_1_bps",
        "dollar_momentum_5_bps",
        "gold_eur_return_1_bps",
        "gold_dollar_correlation_20",
        "dollar_age_seconds",
    )
    frame = _frame()
    frame["dollar_value"] = np.exp(-np.arange(len(frame)) * 0.001)
    result = build_dollar_features(frame)
    np.testing.assert_allclose(result["dollar_return_1_bps"].iloc[1:], 10.0, atol=1e-10)
    np.testing.assert_allclose(result["dollar_momentum_5_bps"].iloc[5:], 50.0, atol=1e-10)
    assert_frame_equal(result.loc[:, frame.columns], frame)


def test_independent_eur_cross_and_paired_correlation_calculation() -> None:
    frame = _frame()
    result = build_dollar_features(frame)
    cross = frame["gold_close"] / frame["dollar_value"]
    np.testing.assert_allclose(
        result["gold_eur_return_1_bps"].iloc[1:],
        np.diff(np.log(cross)) * 10_000,
        atol=2e-11,
    )
    assert result.loc[5, "dollar_momentum_5_bps"] == pytest.approx(
        -10_000 * np.log(frame.loc[5, "dollar_value"] / frame.loc[0, "dollar_value"])
    )
    gold_returns = np.diff(np.log(frame.loc[:20, "gold_close"]))
    dollar_returns = -np.diff(np.log(frame.loc[:20, "dollar_value"]))
    assert result.loc[20, "gold_dollar_correlation_20"] == pytest.approx(
        np.corrcoef(gold_returns, dollar_returns)[0, 1]
    )
    assert result["dollar_return_1_bps"].iloc[:1].isna().all()
    assert result["dollar_momentum_5_bps"].iloc[:5].isna().all()
    assert result["gold_dollar_correlation_20"].iloc[:20].isna().all()
    assert result.loc[20:, list(DOLLAR_MODEL_FEATURES)].notna().all().all()


def test_preserves_every_row_index_and_audit_field_without_mutating_input() -> None:
    frame = _frame()
    frame.index = pd.Index(frame["sample_id"], name="original_gold_sample")
    frame["dollar_raw_sha256"] = "b" * 64
    frame["dollar_revision_id"] = "v0"
    original = frame.copy(deep=True)
    result = build_dollar_features(frame)
    assert_frame_equal(frame, original)
    assert_frame_equal(result.loc[:, frame.columns], frame)
    assert tuple(result.columns[len(frame.columns) :]) == (
        *PRICE_FEATURE_NAMES,
        *DOLLAR_FEATURE_NAMES[:4],
    )
    assert not result.columns.has_duplicates


def test_future_quotes_flags_and_ages_cannot_change_past_features() -> None:
    frame = _frame()
    baseline = build_dollar_features(frame)
    changed = frame.copy(deep=True)
    changed.loc[35:, "gold_close"] *= 2
    changed.loc[35:, "dollar_value"] *= 0.5
    changed.loc[38:, "dollar_is_stale"] = True
    changed.loc[38:, "dollar_age_seconds"] = 3_600.0
    changed.loc[41:, "dollar_is_missing"] = True
    changed.loc[41:, "dollar_value"] = np.nan
    changed.loc[41:, "dollar_age_seconds"] = np.nan
    assert_frame_equal(build_dollar_features(changed).iloc[:35], baseline.iloc[:35])
    assert_frame_equal(build_dollar_features(frame.iloc[:35]), baseline.iloc[:35])


@pytest.mark.parametrize("flag", ["dollar_is_missing", "dollar_is_stale"])
def test_outage_preserves_gold_samples_and_restarts_context_windows(flag: str) -> None:
    frame = _frame()
    before = build_dollar_features(frame)
    frame.loc[25, flag] = True
    frame.loc[25, "dollar_value"] = 9_999.0
    frame.loc[25, "dollar_age_seconds"] = 3_600.0
    result = build_dollar_features(frame)
    assert_frame_equal(result[list(PRICE_FEATURE_NAMES)], before[list(PRICE_FEATURE_NAMES)])
    assert result.loc[25:26, ["dollar_return_1_bps", "gold_eur_return_1_bps"]].isna().all().all()
    assert pd.notna(result.loc[27, "dollar_return_1_bps"])
    assert result.loc[25:30, "dollar_momentum_5_bps"].isna().all()
    assert pd.notna(result.loc[31, "dollar_momentum_5_bps"])
    assert result.loc[25:45, "gold_dollar_correlation_20"].isna().all()
    assert pd.notna(result.loc[46, "gold_dollar_correlation_20"])
    assert result.loc[25, "dollar_value"] == 9_999.0
    assert result["sample_id"].equals(frame["sample_id"])


def test_prediction_gap_restarts_gold_and_dollar_windows() -> None:
    frame = _frame().drop(index=25)
    result = build_dollar_features(frame)
    for name in ("gold_return_1_bps", "dollar_return_1_bps", "gold_eur_return_1_bps"):
        assert pd.isna(result.loc[26, name])
        assert pd.notna(result.loc[27, name])
    for name in ("gold_momentum_5_bps", "dollar_momentum_5_bps"):
        assert result.loc[26:30, name].isna().all()
        assert pd.notna(result.loc[31, name])
    assert result.loc[26:45, "gold_dollar_correlation_20"].isna().all()
    assert pd.notna(result.loc[46, "gold_dollar_correlation_20"])


def test_fresh_repeated_asof_quote_does_not_invent_zero_return() -> None:
    frame = _frame()
    frame.loc[25:26, "dollar_value"] = frame.loc[24, "dollar_value"]
    frame.loc[25:26, "dollar_age_seconds"] = [240.0, 420.0]
    result = build_dollar_features(frame)
    assert result.loc[25:27, ["dollar_return_1_bps", "gold_eur_return_1_bps"]].isna().all().all()
    assert pd.notna(result.loc[28, "dollar_return_1_bps"])
    assert result.loc[25:31, "dollar_momentum_5_bps"].isna().all()
    assert pd.notna(result.loc[32, "dollar_momentum_5_bps"])
    assert result.loc[25:46, "gold_dollar_correlation_20"].isna().all()
    assert pd.notna(result.loc[47, "gold_dollar_correlation_20"])
    assert result.loc[25:26, "dollar_value"].notna().all()


def test_absent_source_retains_gold_universe_and_has_no_usable_dollar_features() -> None:
    frame = _frame()
    joined = join_context(_predictions(frame), None, _source())
    result = build_dollar_features(joined)
    assert result[list(DOLLAR_MODEL_FEATURES)].isna().all().all()
    assert result["dollar_is_missing"].all()
    assert_frame_equal(result[_predictions(frame).columns], _predictions(frame))


def test_constant_quotes_do_not_fabricate_correlation() -> None:
    frame = _frame()
    frame["dollar_value"] = 1.1
    result = build_dollar_features(frame)
    assert result["gold_dollar_correlation_20"].isna().all()
    assert result["dollar_return_1_bps"].iloc[1:].eq(0.0).all()


def test_late_release_uses_only_available_quote_and_restarts_windows() -> None:
    frame = _frame()
    observations = _observations(frame)
    observations.loc[25, "available_at_utc"] += pd.Timedelta(1, unit="ns")
    result = build_dollar_features(join_context(_predictions(frame), observations, _source()))
    assert result.loc[25, "dollar_observation_id"] == "quote-24"
    assert result.loc[25, "dollar_age_seconds"] == 240.0
    assert result.loc[25:26, "dollar_return_1_bps"].isna().all()
    assert pd.notna(result.loc[27, "dollar_return_1_bps"])
    assert result.loc[25:45, "gold_dollar_correlation_20"].isna().all()
    assert pd.notna(result.loc[46, "gold_dollar_correlation_20"])
    assert (result["dollar_available_at_utc"] <= result["prediction_time_utc"]).all()


def test_revisions_never_rewrite_history_or_rejuvenate_quote_age() -> None:
    frame = _frame()
    observations = _observations(frame)
    predictions = _predictions(frame)
    # The latest observation is revised while its next expected candle is absent.
    observations = observations.drop(index=26)
    revision = observations.loc[[25]].copy()
    revision["revision_id"] = "v2"
    revision["raw_sha256"] = "b" * 64
    revision["value"] = 1.5
    revision["available_at_utc"] = frame.loc[26, "prediction_time_utc"]
    revised = pd.concat([observations, revision], ignore_index=True)
    baseline = build_dollar_features(join_context(predictions, observations, _source()))
    result = build_dollar_features(join_context(predictions, revised, _source()))
    assert_frame_equal(result.iloc[:26], baseline.iloc[:26])
    assert result.loc[26, "dollar_value"] == 1.5
    assert result.loc[26, "dollar_revision_id"] == "v2"
    assert result.loc[26, "dollar_age_seconds"] == 240.0
    assert pd.isna(result.loc[26, "dollar_return_1_bps"])
    # A still later vintage of an older candle cannot replace the latest quote.
    old = observations.loc[[10]].copy()
    old["revision_id"] = "v2"
    old["value"] = 99.0
    old["available_at_utc"] = frame.loc[30, "prediction_time_utc"] + pd.Timedelta(seconds=1)
    assert_frame_equal(
        result,
        build_dollar_features(
            join_context(predictions, pd.concat([revised, old], ignore_index=True), _source())
        ),
    )


def test_future_release_value_and_provenance_mutation_preserves_past_features() -> None:
    frame = _frame()
    observations = _observations(frame)
    changed = observations.copy(deep=True)
    changed.loc[35:, "value"] *= 0.5
    changed.loc[35:, "raw_sha256"] = "b" * 64
    changed.loc[35:, "revision_id"] = "future-mutated"
    baseline = build_dollar_features(join_context(_predictions(frame), observations, _source()))
    result = build_dollar_features(join_context(_predictions(frame), changed, _source()))
    assert_frame_equal(result.iloc[:35], baseline.iloc[:35])


def test_empty_frame_keeps_feature_schema() -> None:
    result = build_dollar_features(_frame().iloc[:0])
    assert result.empty
    assert set((*PRICE_FEATURE_NAMES, *DOLLAR_FEATURE_NAMES)).issubset(result.columns)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("gold_close", np.nan),
        ("gold_close", np.inf),
        ("gold_close", 0.0),
        ("dollar_value", -1.0),
        ("dollar_value", np.inf),
        ("dollar_age_seconds", -1.0),
        ("dollar_age_seconds", np.inf),
        ("dollar_age_seconds", np.nan),
    ],
)
def test_invalid_price_and_age_fail_closed(column: str, value: float) -> None:
    frame = _frame()
    frame.loc[5, column] = value
    with pytest.raises(ValueError, match=column):
        build_dollar_features(frame)


@pytest.mark.parametrize("column", ["gold_close", "dollar_value", "dollar_age_seconds"])
@pytest.mark.parametrize("value", ["25.0", True])
def test_nonnumeric_or_boolean_prices_fail_closed(column: str, value: str | bool) -> None:
    frame = _frame()
    frame[column] = value
    with pytest.raises(ValueError, match="numeric"):
        build_dollar_features(frame)


@pytest.mark.parametrize("column", ["dollar_is_missing", "dollar_is_stale"])
def test_invalid_context_flags_fail_closed(column: str) -> None:
    frame = _frame()
    frame[column] = 0
    with pytest.raises(ValueError, match="booleans"):
        build_dollar_features(frame)
    frame[column] = pd.Series(False, index=frame.index, dtype="boolean")
    frame.loc[5, column] = pd.NA
    with pytest.raises(ValueError, match="booleans"):
        build_dollar_features(frame)


def test_missing_or_duplicate_columns_fail_closed() -> None:
    frame = _frame()
    with pytest.raises(ValueError, match="missing columns"):
        build_dollar_features(frame.drop(columns="dollar_value"))
    with pytest.raises(ValueError, match="duplicate columns"):
        build_dollar_features(pd.concat([frame, frame[["dollar_value"]]], axis=1))


@pytest.mark.parametrize("problem", ["naive", "non_utc", "unsorted", "duplicate", "missing"])
def test_invalid_prediction_timestamps_fail_closed(problem: str) -> None:
    frame = _frame()
    if problem == "naive":
        frame["prediction_time_utc"] = frame["prediction_time_utc"].dt.tz_localize(None)
    elif problem == "non_utc":
        frame["prediction_time_utc"] = frame["prediction_time_utc"].dt.tz_convert("Europe/Brussels")
    elif problem == "unsorted":
        frame = frame.iloc[::-1]
    elif problem == "duplicate":
        frame.loc[5, "prediction_time_utc"] = frame.loc[4, "prediction_time_utc"]
    else:
        frame.loc[5, "prediction_time_utc"] = pd.NaT
    with pytest.raises(ValueError, match="prediction_time_utc"):
        build_dollar_features(frame)


@pytest.mark.parametrize("cadence", [0, -180, True, 180.0])
def test_invalid_cadence_fails_closed(cadence: int) -> None:
    with pytest.raises(ValueError, match="cadence_seconds"):
        build_dollar_features(_frame(), cadence_seconds=cadence)


def test_shared_extraction_preserves_pre_refactor_silver_golden_values() -> None:
    # Values captured from the completed silver implementation before extraction.
    # This protects the frozen operation order as well as the independent formulas.
    frame = _frame().rename(columns=lambda name: name.replace("dollar", "silver"))
    steps = np.arange(len(frame), dtype=float)
    frame["silver_value"] = 25.0 * np.exp(np.cumsum(0.0002 + np.cos(steps) * 0.002))
    frame["silver_age_seconds"] = 0.0
    expected = np.array(
        [
            [9.414709848085678, np.nan, 12.806046117361802, np.nan, 4.379587501046954, np.nan],
            [-8.589242746639414, 6.761616497223954, 7.673243709267474,
             -14.716369253595651, 4.382074433248963, np.nan],
            [10.129452507277037, -4.374655949259676, 10.161641236270391,
             26.485611397384368, 4.377845639424435, 0.042847520929869],
            [-8.055783620060097, 4.889508137919662, 10.483580146742888,
             -14.965240756037446, 4.379834333186730, -0.033998158849194],
            [7.367380071388041, 10.444692350262130, -13.421604459513681,
             32.466272068534252, 4.376571513097574, -0.051152216988315],
        ]
    )
    names = [
        *PRICE_FEATURE_NAMES,
        "silver_return_1_bps",
        "silver_momentum_5_bps",
        "gold_silver_log_ratio",
        "gold_silver_correlation_20",
    ]
    np.testing.assert_allclose(
        build_silver_features(frame).loc[[1, 5, 20, 24, 59], names],
        expected,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )
