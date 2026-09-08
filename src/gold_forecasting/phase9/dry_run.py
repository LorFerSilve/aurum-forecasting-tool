"""Synthetic end-to-end phase-9 dry run.

No repository market data or frozen benchmark reference is consumed. The flow exists
solely to exercise data/labels/sequences/folds/training/artifacts/verification together.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gold_forecasting.artifacts import write_json_atomic
from gold_forecasting.phase8.sequences import build_phase8_sequences
from gold_forecasting.phase9.artifacts import (
    finalize_phase9_manifest,
    verify_phase9,
    write_phase9_fold_artifacts,
    write_phase9_run_contract,
)
from gold_forecasting.phase9.config import Phase9Config
from gold_forecasting.phase9.dataset import build_phase9_dataset
from gold_forecasting.phase9.orchestration import (
    build_phase9_schedule,
    evaluate_phase9_fold,
)


class Phase9DryRunError(ValueError):
    """Raised when the isolated synthetic integration run cannot be constructed."""


def _prediction_times() -> pd.DatetimeIndex:
    blocks = (
        "2020-02-10T12:00:00Z",
        "2021-04-10T12:00:00Z",
        "2021-07-10T12:00:00Z",
        "2021-10-10T12:00:00Z",
        "2022-02-10T12:00:00Z",
    )
    values: list[pd.Timestamp] = []
    for block in blocks:
        start = pd.Timestamp(block)
        values.extend(
            start + pd.Timedelta(days=2 * index)
            for index in range(6)
        )
    return pd.DatetimeIndex(
        values,
        dtype="datetime64[ns, UTC]",
    )


def _synthetic_candles(
    predictions: pd.DatetimeIndex,
    *,
    timeframe: str,
) -> pd.DataFrame:
    minutes = {
        "1min": 1,
        "3min": 3,
        "15min": 15,
    }[timeframe]
    offsets = {
        "1min": range(-70, 17),
        "3min": range(-22, 6),
        "15min": range(-9, 0),
    }[timeframe]
    rows: list[dict[str, object]] = []
    for sample_index, prediction in enumerate(predictions):
        direction = (sample_index % 3) - 1
        base = 1_900.0 + sample_index * 4.0
        for offset in offsets:
            open_time = (
                prediction
                + pd.Timedelta(
                    minutes=offset * minutes
                )
            )
            relative_minutes = (
                offset * minutes
            )
            slope = (
                0.15
                * float(direction)
            )
            open_price = (
                base
                + slope * relative_minutes
            )
            close_price = (
                open_price
                + slope * minutes
            )
            high = (
                max(
                    open_price,
                    close_price,
                )
                + 0.08
            )
            low = (
                min(
                    open_price,
                    close_price,
                )
                - 0.08
            )
            rows.append(
                {
                    "timestamp_open_utc": (
                        open_time
                    ),
                    "timestamp_close_utc": (
                        open_time
                        + pd.Timedelta(
                            minutes=minutes
                        )
                    ),
                    "bid_open": open_price,
                    "bid_high": high,
                    "bid_low": low,
                    "bid_close": close_price,
                    "instrument": "XAU_USD",
                    "timeframe": timeframe,
                    "is_complete": True,
                    "source": "phase9_synthetic",
                    "raw_file_hash": "a" * 64,
                    "ingested_at_utc": pd.Timestamp(
                        "2026-09-08T00:00:00Z"
                    ),
                    "dataset_version": (
                        "sha256:" + "b" * 64
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        [
            "instrument",
            "source",
            "timestamp_open_utc",
        ],
        kind="stable",
    ).reset_index(drop=True)
    if frame[
        [
            "instrument",
            "source",
            "timestamp_open_utc",
        ]
    ].duplicated().any():
        raise Phase9DryRunError(
            "synthetic candle windows overlap"
        )
    return frame


def _config() -> Phase9Config:
    return Phase9Config(
        seeds=(1101, 1102),
        encoder_hidden_size=8,
        fusion_size=8,
        parameter_budget=40_000,
        batch_size=32,
        max_epochs=2,
        early_stopping_patience=1,
        learning_rate=0.01,
        weight_decay=0.0,
        minimum_policy_trades=1,
        device="cpu",
        mixed_precision=False,
        deterministic_algorithms=True,
    )


def run_phase9_synthetic_dry_run(
    output: str | Path,
) -> dict[str, object]:
    destination = Path(output)
    if destination.exists():
        raise Phase9DryRunError(
            "synthetic dry-run destination must not already exist"
        )
    destination.mkdir(
        parents=True,
    )
    config = _config()
    predictions = _prediction_times()
    candidates = pd.DataFrame(
        {
            "instrument": "XAU_USD",
            "source": "phase9_synthetic",
            "prediction_time_utc": predictions,
        }
    )
    candles = {
        timeframe: _synthetic_candles(
            predictions,
            timeframe=timeframe,
        )
        for timeframe in (
            "1min",
            "3min",
            "15min",
        )
    }
    dataset = build_phase9_dataset(
        candles["1min"],
        candles["3min"],
        candidates,
        neutral_threshold_bps=(
            config.neutral_threshold_bps
        ),
    )
    key = [
        "instrument",
        "source",
        "prediction_time_utc",
    ]
    sequences = build_phase8_sequences(
        candles,
        dataset.table.loc[:, key],
        config.sequence_lengths,
    )
    folds = build_phase9_schedule(config)
    fold = folds[0]

    project_root = Path(__file__).resolve().parents[3]
    write_phase9_run_contract(
        destination,
        config=config,
        folds=folds,
        phase8_reference_run=(
            "synthetic-phase8-reference"
        ),
        code_version="synthetic-dry-run",
        runtime={
            "run_mode": (
                "synthetic_dry_run"
            ),
            "device": "cpu",
        },
        requirements_lock=(
            project_root
            / "requirements.lock"
        ).read_text(encoding="utf-8"),
        neural_lock=(
            project_root
            / "requirements-neural.lock"
        ).read_text(encoding="utf-8"),
    )
    write_json_atomic(
        destination / "dataset_diagnostics.json",
        dataset.diagnostics,
    )
    write_json_atomic(
        destination / "run.json",
        {
            "run_id": destination.name,
            "status": "running",
            "run_mode": "synthetic_dry_run",
        },
    )

    result = evaluate_phase9_fold(
        dataset.table,
        sequences,
        fold,
        config,
    )
    write_phase9_fold_artifacts(
        result,
        destination / fold.name,
    )
    summary = {
        "protocol": "phase9-v1",
        "run_mode": "synthetic_dry_run",
        "holdout_opened": False,
        "test_years": [
            2022,
            2023,
            2024,
        ],
        "phase8_reference_run": (
            "synthetic-phase8-reference"
        ),
        "folds": {
            fold.name: {
                "split": result.split_audit,
                "comparison": (
                    result.comparison
                ),
            }
        },
    }
    finalize_phase9_manifest(
        destination,
        summary,
    )
    write_json_atomic(
        destination / "run.json",
        {
            "run_id": destination.name,
            "status": "succeeded",
            "run_mode": "synthetic_dry_run",
        },
    )
    return verify_phase9(
        destination
    )


__all__ = [
    "Phase9DryRunError",
    "run_phase9_synthetic_dry_run",
]
