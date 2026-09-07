"""Training-contract tests for the phase-8 neural challenger."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gold_forecasting.phase8.config import Phase8Config
from gold_forecasting.phase8.sequences import (
    SEQUENCE_FEATURE_NAMES,
    Phase8SequenceBuildResult,
    SequenceFrame,
)
from gold_forecasting.phase8.training import (
    _gradients_are_finite,
    fit_normalizer,
    predict_neural_model,
    save_checkpoint,
    train_neural_model,
)


def _synthetic_training_fixture(
    *,
    rows: int = 36,
) -> tuple[Phase8SequenceBuildResult, pd.DataFrame]:
    classes = np.arange(rows, dtype=np.int64) % 3
    timesteps = 6
    values = np.zeros(
        (rows, timesteps, len(SEQUENCE_FEATURE_NAMES)),
        dtype=np.float32,
    )
    for class_id in range(3):
        values[classes == class_id, :, class_id] = 5.0

    prediction_times = pd.Series(
        pd.date_range(
            "2024-01-02T00:18:00Z",
            periods=rows,
            freq="3min",
        )
    )
    frame = SequenceFrame(
        timeframe="3min",
        values=values,
        available=np.ones(rows, dtype=bool),
        source_close_utc=prediction_times.copy(),
        window_start_utc=prediction_times - pd.Timedelta(minutes=18),
    )
    sequences = Phase8SequenceBuildResult(
        prediction_times=prediction_times,
        by_timeframe={"3min": frame},
        diagnostics={"rows": rows},
    )
    direction = classes.astype(np.float64) - 1.0
    table = pd.DataFrame(
        {
            "target_class_id": classes,
            "arithmetic_return_bps": direction * 10.0,
            "future_range_bps": 4.0 + classes.astype(np.float64),
            "future_realized_vol_bps": 2.0 + classes.astype(np.float64) * 0.5,
        }
    )
    return sequences, table


def _direction_only_config() -> Phase8Config:
    return Phase8Config(
        encoder_hidden_size=16,
        fusion_size=16,
        parameter_budget=50_000,
        batch_size=64,
        max_epochs=40,
        early_stopping_patience=4,
        learning_rate=0.03,
        weight_decay=0.0,
        return_loss_weight=0.0,
        range_loss_weight=0.0,
        volatility_loss_weight=0.0,
        device="cpu",
        mixed_precision=False,
        deterministic_algorithms=True,
    )


def test_normalizer_is_unchanged_when_only_validation_rows_are_mutated() -> None:
    sequences, table = _synthetic_training_fixture()
    train_rows = np.arange(18, dtype=np.int64)
    original = fit_normalizer(sequences, table, train_rows, ("3min",))

    mutated_values = sequences.by_timeframe["3min"].values.copy()
    mutated_values[18:] = mutated_values[18:] * 1_000.0 + 123.0
    source = sequences.by_timeframe["3min"]
    mutated_sequences = Phase8SequenceBuildResult(
        prediction_times=sequences.prediction_times.copy(),
        by_timeframe={
            "3min": SequenceFrame(
                timeframe="3min",
                values=mutated_values,
                available=source.available.copy(),
                source_close_utc=source.source_close_utc.copy(),
                window_start_utc=source.window_start_utc.copy(),
            )
        },
        diagnostics=sequences.diagnostics,
    )
    mutated_table = table.copy()
    mutated_table.loc[18:, "arithmetic_return_bps"] *= 1_000.0
    mutated_table.loc[18:, "future_range_bps"] *= 1_000.0
    mutated_table.loc[18:, "future_realized_vol_bps"] *= 1_000.0

    rebuilt = fit_normalizer(
        mutated_sequences,
        mutated_table,
        train_rows,
        ("3min",),
    )

    np.testing.assert_array_equal(
        rebuilt.medians["3min"],
        original.medians["3min"],
    )
    np.testing.assert_array_equal(
        rebuilt.means["3min"],
        original.means["3min"],
    )
    np.testing.assert_array_equal(
        rebuilt.scales["3min"],
        original.scales["3min"],
    )
    assert rebuilt.target_scales == original.target_scales


def test_tiny_direction_dataset_overfits_and_same_seed_reproduces(
    tmp_path: Path,
) -> None:
    sequences, table = _synthetic_training_fixture()
    rows = np.arange(len(table), dtype=np.int64)
    config = _direction_only_config()

    first = train_neural_model(
        sequences,
        table,
        rows,
        None,
        ("3min",),
        config,
        seed=20260906,
        forced_epochs=config.max_epochs,
    )
    first_prediction = predict_neural_model(
        first,
        sequences,
        rows,
        ("3min",),
        config,
    )
    second = train_neural_model(
        sequences,
        table,
        rows,
        None,
        ("3min",),
        config,
        seed=20260906,
        forced_epochs=config.max_epochs,
    )
    second_prediction = predict_neural_model(
        second,
        sequences,
        rows,
        ("3min",),
        config,
    )

    labels = table["target_class_id"].to_numpy(dtype=np.int64)
    accuracy = float(
        np.mean(first_prediction.probabilities.argmax(axis=1) == labels)
    )
    assert accuracy >= 0.95
    np.testing.assert_allclose(
        first_prediction.probabilities.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert first_prediction.probabilities.dtype == np.float64
    assert first.parameter_count <= config.parameter_budget
    assert first.optimizer_steps > 0
    assert first.amp_skipped_steps == 0
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(first, checkpoint)
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    assert payload["optimizer_steps"] == first.optimizer_steps
    assert payload["amp_skipped_steps"] == first.amp_skipped_steps
    assert payload["parameter_count"] == first.parameter_count
    assert all(
        np.isfinite(float(epoch["max_preclip_gradient_norm"]))
        and float(epoch["max_preclip_gradient_norm"]) > 0.0
        for epoch in first.history
    )
    np.testing.assert_allclose(
        second_prediction.probabilities,
        first_prediction.probabilities,
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        second_prediction.expected_return_bps,
        first_prediction.expected_return_bps,
        rtol=0.0,
        atol=1e-7,
    )


def test_gradient_finiteness_guard_detects_nonfinite_values() -> None:
    sequences, table = _synthetic_training_fixture(rows=6)
    rows = np.arange(len(table), dtype=np.int64)
    config = _direction_only_config()
    result = train_neural_model(
        sequences,
        table,
        rows,
        None,
        ("3min",),
        config,
        seed=7,
        forced_epochs=1,
    )

    parameters = [
        parameter
        for parameter in result.model.parameters()
        if parameter.requires_grad
    ]
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    assert _gradients_are_finite(result.model)

    parameters[0].grad = torch.full_like(
        parameters[0],
        float("inf"),
    )
    assert not _gradients_are_finite(result.model)
