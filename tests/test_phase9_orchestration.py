"""Structural tests for phase-9 walk-forward orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold_forecasting.phase8.sequences import (
    Phase8SequenceBuildResult,
    SequenceFrame,
)
from gold_forecasting.phase9.config import Phase9Config
from gold_forecasting.phase9.orchestration import (
    GAP_MINUTES,
    Phase9OrchestrationError,
    build_phase9_schedule,
    validate_phase9_alignment,
)


def _alignment_fixture() -> tuple[pd.DataFrame, Phase8SequenceBuildResult]:
    times = pd.Series(
        pd.date_range(
            "2024-01-02T00:00:00Z",
            periods=4,
            freq="3min",
        )
    )
    table = pd.DataFrame(
        {
            "prediction_time_utc": times,
            "label_end_time_utc": (
                times + pd.Timedelta(minutes=16)
            ),
            "entry_time_utc": (
                times + pd.Timedelta(minutes=1)
            ),
            "sample_id": [
                f"sample-{index}"
                for index in range(4)
            ],
        }
    )
    frame = SequenceFrame(
        timeframe="3min",
        values=np.ones((4, 3, 6), dtype=np.float32),
        available=np.ones(4, dtype=bool),
        source_close_utc=times.copy(),
        window_start_utc=(
            times - pd.Timedelta(minutes=9)
        ),
    )
    sequences = Phase8SequenceBuildResult(
        prediction_times=times.copy(),
        by_timeframe={"3min": frame},
        diagnostics={"rows": 4},
    )
    return table, sequences


def test_phase9_schedule_preserves_phase6_outer_years_and_gap() -> None:
    config = Phase9Config()

    folds = build_phase9_schedule(config)

    assert tuple(
        fold.name
        for fold in folds
    ) == (
        "test_2022",
        "test_2023",
        "test_2024",
    )
    assert GAP_MINUTES == 181
    assert all(
        fold.calibration.end
        == fold.test.start
        for fold in folds
    )


def test_phase9_alignment_accepts_exact_prediction_index() -> None:
    table, sequences = _alignment_fixture()

    validate_phase9_alignment(
        table,
        sequences,
    )


def test_phase9_alignment_rejects_reordered_sequence_rows() -> None:
    table, sequences = _alignment_fixture()
    changed = Phase8SequenceBuildResult(
        prediction_times=(
            sequences.prediction_times
            .iloc[::-1]
            .reset_index(drop=True)
        ),
        by_timeframe=sequences.by_timeframe,
        diagnostics=sequences.diagnostics,
    )

    with pytest.raises(
        Phase9OrchestrationError,
        match="not aligned",
    ):
        validate_phase9_alignment(
            table,
            changed,
        )


def test_phase9_alignment_rejects_duplicate_sample_ids() -> None:
    table, sequences = _alignment_fixture()
    changed = table.copy()
    changed.loc[1, "sample_id"] = (
        changed.loc[0, "sample_id"]
    )

    with pytest.raises(
        Phase9OrchestrationError,
        match="unique",
    ):
        validate_phase9_alignment(
            changed,
            sequences,
        )
