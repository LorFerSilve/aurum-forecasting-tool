"""Common-universe parity and mutation guards for phase 9."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold_forecasting.backtesting.v1 import DecisionPolicy
from gold_forecasting.phase8.sequences import build_phase8_sequences
from gold_forecasting.phase9.dataset import build_phase9_dataset
from gold_forecasting.phase9.parity import (
    Phase9ParityError,
    evaluate_frozen_reference_on_phase9_universe,
    restrict_reference_to_phase9_universe,
)
from gold_forecasting.phase9.targets import build_future_path_targets
from tests.test_phase9_targets import _candles, _predictions


def _reference_and_phase9() -> tuple[pd.DataFrame, pd.DataFrame]:
    times = pd.Series(
        pd.date_range(
            "2024-01-02T01:00:00Z",
            periods=3,
            freq="15min",
        )
    )
    phase9 = pd.DataFrame(
        {
            "sample_id": [
                "sample-b",
                "sample-d",
            ],
            "prediction_time_utc": (
                times.iloc[[0, 2]]
                .reset_index(drop=True)
            ),
            "target_class_id": [0, 2],
            "arithmetic_return_bps": [
                -8.0,
                9.0,
            ],
        }
    )
    reference = pd.DataFrame(
        {
            "sample_id": [
                "sample-a",
                "sample-b",
                "sample-c",
                "sample-d",
            ],
            "prediction_time_utc": [
                times.iloc[0]
                - pd.Timedelta(minutes=15),
                times.iloc[0],
                times.iloc[1],
                times.iloc[2],
            ],
            "target_class_id": [
                1,
                0,
                1,
                2,
            ],
            "target_class": [
                "neutral",
                "down",
                "neutral",
                "up",
            ],
            "arithmetic_return_bps": [
                0.0,
                -8.0,
                1.0,
                9.0,
            ],
            "expected_return_bps": [
                0.0,
                -5.0,
                0.5,
                6.0,
            ],
            "p_down": [
                0.1,
                0.8,
                0.2,
                0.1,
            ],
            "p_neutral": [
                0.8,
                0.1,
                0.7,
                0.1,
            ],
            "p_up": [
                0.1,
                0.1,
                0.1,
                0.8,
            ],
            "entry_time_utc": [
                value + pd.Timedelta(minutes=1)
                for value in [
                    times.iloc[0]
                    - pd.Timedelta(minutes=15),
                    times.iloc[0],
                    times.iloc[1],
                    times.iloc[2],
                ]
            ],
            "label_end_time_utc": [
                value + pd.Timedelta(minutes=16)
                for value in [
                    times.iloc[0]
                    - pd.Timedelta(minutes=15),
                    times.iloc[0],
                    times.iloc[1],
                    times.iloc[2],
                ]
            ],
            "entry_bid_open": [
                2000.0,
                2000.0,
                2000.0,
                2000.0,
            ],
            "exit_bid_open": [
                2000.0,
                1998.4,
                2000.2,
                2001.8,
            ],
            "horizon_minutes": [
                15,
                15,
                15,
                15,
            ],
            "instrument": [
                "XAU_USD",
            ] * 4,
            "source": [
                "fixture",
            ] * 4,
        }
    )
    return reference, phase9


def test_reference_is_reordered_and_subset_to_exact_phase9_ids() -> None:
    reference, phase9 = _reference_and_phase9()

    restricted = (
        restrict_reference_to_phase9_universe(
            reference,
            phase9,
        )
    )

    assert restricted[
        "sample_id"
    ].tolist() == [
        "sample-b",
        "sample-d",
    ]


def test_reference_missing_phase9_id_fails_closed() -> None:
    reference, phase9 = _reference_and_phase9()
    changed = reference.loc[
        reference["sample_id"].ne(
            "sample-d"
        )
    ]

    with pytest.raises(
        Phase9ParityError,
        match="missing",
    ):
        restrict_reference_to_phase9_universe(
            changed,
            phase9,
        )


def test_reference_target_mutation_fails_closed() -> None:
    reference, phase9 = _reference_and_phase9()
    changed = reference.copy()
    changed.loc[
        changed["sample_id"].eq(
            "sample-d"
        ),
        "arithmetic_return_bps",
    ] = 99.0

    with pytest.raises(
        Phase9ParityError,
        match="return targets",
    ):
        restrict_reference_to_phase9_universe(
            changed,
            phase9,
        )


def test_frozen_reference_evaluation_never_refits_or_reselects() -> None:
    reference, phase9 = _reference_and_phase9()

    result = (
        evaluate_frozen_reference_on_phase9_universe(
            reference,
            phase9,
            DecisionPolicy(
                confidence_threshold=0.5,
                min_expected_net_bps=0.0,
            ),
        )
    )

    assert result["rows"] == 2
    assert result["reference_refit"] is False
    assert result["policy_reselected"] is False


def test_future_path_mutation_changes_label_but_not_historical_sequences() -> None:
    candles_3min = _candles(periods=80)
    predictions = _predictions(
        candles_3min,
        [20],
    )
    original_labels = (
        build_future_path_targets(
            candles_3min,
            predictions,
        )
    )
    sequence_lengths = {
        "3min": 5,
    }
    original_sequences = (
        build_phase8_sequences(
            {"3min": candles_3min},
            predictions,
            sequence_lengths,
        )
    )

    prediction_time = predictions.loc[
        0,
        "prediction_time_utc",
    ]
    changed = candles_3min.copy()
    future_mask = changed[
        "timestamp_open_utc"
    ].eq(prediction_time)
    changed.loc[
        future_mask,
        [
            "bid_open",
            "bid_high",
            "bid_low",
            "bid_close",
        ],
    ] += 10.0
    changed.loc[
        future_mask,
        "bid_high",
    ] += 1.0
    mutated_labels = (
        build_future_path_targets(
            changed,
            predictions,
        )
    )
    mutated_sequences = (
        build_phase8_sequences(
            {"3min": changed},
            predictions,
            sequence_lengths,
        )
    )

    assert not original_labels.labels.equals(
        mutated_labels.labels
    )
    np.testing.assert_array_equal(
        original_sequences.by_timeframe[
            "3min"
        ].values,
        mutated_sequences.by_timeframe[
            "3min"
        ].values,
    )
