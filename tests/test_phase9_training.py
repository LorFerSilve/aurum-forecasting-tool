"""Training smoke tests for the phase-9 future-path challenger."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gold_forecasting.phase8.sequences import (
    SEQUENCE_FEATURE_NAMES,
    Phase8SequenceBuildResult,
    SequenceFrame,
)
from gold_forecasting.phase9.config import Phase9Config
from gold_forecasting.phase9.targets import PATH_COMPONENTS
from gold_forecasting.phase9.training import (
    predict_phase9_model,
    save_phase9_checkpoint,
    train_phase9_model,
)


def _fixture(
    rows: int = 30,
) -> tuple[Phase8SequenceBuildResult, pd.DataFrame]:
    classes = np.arange(rows, dtype=np.int64) % 3
    direction = classes.astype(np.float64) - 1.0
    prediction_times = pd.Series(
        pd.date_range(
            "2024-01-02T01:00:00Z",
            periods=rows,
            freq="3min",
        )
    )
    frames: dict[str, SequenceFrame] = {}
    for timeframe, timesteps in (
        ("1min", 8),
        ("3min", 6),
        ("15min", 4),
    ):
        values = np.zeros(
            (
                rows,
                timesteps,
                len(SEQUENCE_FEATURE_NAMES),
            ),
            dtype=np.float32,
        )
        for class_id in range(3):
            values[
                classes == class_id,
                :,
                class_id,
            ] = 3.0
        frames[timeframe] = SequenceFrame(
            timeframe=timeframe,
            values=values,
            available=np.ones(rows, dtype=bool),
            source_close_utc=prediction_times.copy(),
            window_start_utc=(
                prediction_times
                - pd.Timedelta(minutes=30)
            ),
        )
    table = pd.DataFrame(
        {
            "target_class_id": classes,
            "arithmetic_return_bps": direction * 8.0,
            "future_range_bps": 5.0 + classes,
            "future_realized_vol_bps": 2.0 + classes * 0.5,
        }
    )
    for step in range(5):
        table[
            f"path_step_{step + 1}_gap_log_bps"
        ] = direction * (0.2 + step * 0.05)
        table[
            f"path_step_{step + 1}_body_log_bps"
        ] = direction * (1.0 + step * 0.2)
        table[
            f"path_step_{step + 1}_upper_wick_log_bps"
        ] = 0.8 + classes * 0.1 + step * 0.02
        table[
            f"path_step_{step + 1}_lower_wick_log_bps"
        ] = 0.7 + classes * 0.1 + step * 0.02

    table["path_15m_gap_log_bps"] = table[
        "path_step_1_gap_log_bps"
    ]
    table["path_15m_body_log_bps"] = sum(
        table[
            f"path_step_{step}_body_log_bps"
        ]
        for step in range(1, 6)
    ) + sum(
        table[
            f"path_step_{step}_gap_log_bps"
        ]
        for step in range(2, 6)
    )
    table["path_15m_upper_wick_log_bps"] = (
        1.5 + classes * 0.1
    )
    table["path_15m_lower_wick_log_bps"] = (
        1.4 + classes * 0.1
    )
    table["path_15m_close_return_log_bps"] = (
        table["path_15m_gap_log_bps"]
        + table["path_15m_body_log_bps"]
    )
    return (
        Phase8SequenceBuildResult(
            prediction_times=prediction_times,
            by_timeframe=frames,
            diagnostics={"rows": rows},
        ),
        table,
    )


def _config() -> Phase9Config:
    return Phase9Config(
        encoder_hidden_size=16,
        fusion_size=16,
        parameter_budget=60_000,
        batch_size=64,
        max_epochs=4,
        early_stopping_patience=2,
        learning_rate=0.01,
        weight_decay=0.0,
        device="cpu",
        mixed_precision=False,
        deterministic_algorithms=True,
    )


def test_phase9_training_predicts_all_heads_and_saves_checkpoint(
    tmp_path: Path,
) -> None:
    sequences, table = _fixture()
    rows = np.arange(len(table), dtype=np.int64)
    config = _config()

    result = train_phase9_model(
        sequences,
        table,
        rows,
        None,
        config,
        seed=20260908,
        forced_epochs=2,
    )
    prediction = predict_phase9_model(
        result,
        sequences,
        rows,
        config,
    )

    assert prediction.probabilities.shape == (len(table), 3)
    assert prediction.path_quantiles_log_bps.shape == (
        len(table),
        5,
        len(PATH_COMPONENTS),
        3,
    )
    assert prediction.aggregate_quantiles_log_bps.shape == (
        len(table),
        len(PATH_COMPONENTS),
        3,
    )
    np.testing.assert_allclose(
        prediction.probabilities.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.all(
        prediction.path_quantiles_log_bps[..., 1]
        >= prediction.path_quantiles_log_bps[..., 0]
    )
    assert np.all(
        prediction.path_quantiles_log_bps[..., 2]
        >= prediction.path_quantiles_log_bps[..., 1]
    )
    assert result.optimizer_steps > 0
    assert result.amp_skipped_steps == 0
    assert result.sequence_normalizer.fit_rows == len(rows)
    assert result.path_normalizer.fit_rows == len(rows)

    checkpoint = tmp_path / "phase9.pt"
    save_phase9_checkpoint(
        result,
        checkpoint,
    )
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    assert payload["parameter_count"] == result.parameter_count
    assert payload["best_epoch"] == result.best_epoch
    assert payload["path_normalizer"]["fit_rows"] == len(rows)


def test_phase9_same_seed_reproduces_cpu_predictions() -> None:
    sequences, table = _fixture(rows=18)
    rows = np.arange(len(table), dtype=np.int64)
    config = _config()

    first = train_phase9_model(
        sequences,
        table,
        rows,
        None,
        config,
        seed=77,
        forced_epochs=2,
    )
    second = train_phase9_model(
        sequences,
        table,
        rows,
        None,
        config,
        seed=77,
        forced_epochs=2,
    )
    first_prediction = predict_phase9_model(
        first,
        sequences,
        rows,
        config,
    )
    second_prediction = predict_phase9_model(
        second,
        sequences,
        rows,
        config,
    )

    np.testing.assert_allclose(
        second_prediction.probabilities,
        first_prediction.probabilities,
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        second_prediction.path_quantiles_log_bps,
        first_prediction.path_quantiles_log_bps,
        rtol=0.0,
        atol=1e-6,
    )


def test_recursive_phase9_training_uses_same_train_predict_contract() -> None:
    sequences, table = _fixture(rows=18)
    rows = np.arange(len(table), dtype=np.int64)
    config = _config()

    result = train_phase9_model(
        sequences,
        table,
        rows,
        None,
        config,
        seed=91,
        forced_epochs=1,
        model_variant="recursive",
    )
    prediction = predict_phase9_model(
        result,
        sequences,
        rows,
        config,
    )

    assert result.model_variant == "recursive"
    assert prediction.path_quantiles_log_bps.shape == (
        len(table),
        5,
        len(PATH_COMPONENTS),
        3,
    )
    assert np.isfinite(
        prediction.path_quantiles_log_bps
    ).all()
