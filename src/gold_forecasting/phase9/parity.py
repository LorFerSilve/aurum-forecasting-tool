"""Common-sample and future-only parity guards for phase-9 comparisons."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from gold_forecasting.backtesting.v1 import DecisionPolicy, run_backtest_v1
from gold_forecasting.benchmark.pipeline import (
    BASE_COSTS,
    PROBABILITY_COLUMNS,
    STRESS_COSTS,
    _metrics,
)
from gold_forecasting.datasets.preprocessing import sample_id_digest


class Phase9ParityError(ValueError):
    """Raised when a reference cannot be compared on the phase-9 sample universe."""


def restrict_reference_to_phase9_universe(
    reference_records: pd.DataFrame,
    phase9_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Subset a frozen reference to exactly the phase-9 ids and verify target parity."""

    required_reference = {
        "sample_id",
        "prediction_time_utc",
        "target_class_id",
        "arithmetic_return_bps",
        "expected_return_bps",
        *PROBABILITY_COLUMNS,
    }
    required_phase9 = {
        "sample_id",
        "prediction_time_utc",
        "target_class_id",
        "arithmetic_return_bps",
    }
    missing_reference = required_reference.difference(
        reference_records.columns
    )
    missing_phase9 = required_phase9.difference(
        phase9_rows.columns
    )
    if missing_reference or missing_phase9:
        raise Phase9ParityError(
            "common-sample parity inputs are incomplete: "
            f"reference={sorted(missing_reference)}, "
            f"phase9={sorted(missing_phase9)}"
        )
    if (
        reference_records["sample_id"].duplicated().any()
        or phase9_rows["sample_id"].duplicated().any()
    ):
        raise Phase9ParityError(
            "common-sample parity requires unique sample ids"
        )

    indexed = reference_records.set_index(
        "sample_id",
        drop=False,
    )
    ids = phase9_rows[
        "sample_id"
    ].tolist()
    missing = [
        sample_id
        for sample_id in ids
        if sample_id not in indexed.index
    ]
    if missing:
        raise Phase9ParityError(
            "frozen reference is missing phase-9 samples"
        )
    restricted = indexed.loc[
        ids
    ].reset_index(drop=True)
    phase9 = phase9_rows.reset_index(
        drop=True
    )

    if not restricted[
        "prediction_time_utc"
    ].equals(
        phase9["prediction_time_utc"]
    ):
        raise Phase9ParityError(
            "reference and phase-9 prediction timestamps differ"
        )
    if not np.array_equal(
        restricted[
            "target_class_id"
        ].to_numpy(dtype=np.int64),
        phase9[
            "target_class_id"
        ].to_numpy(dtype=np.int64),
    ):
        raise Phase9ParityError(
            "reference and phase-9 direction targets differ"
        )
    if not np.allclose(
        restricted[
            "arithmetic_return_bps"
        ].to_numpy(dtype=np.float64),
        phase9[
            "arithmetic_return_bps"
        ].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise Phase9ParityError(
            "reference and phase-9 return targets differ"
        )
    expected_digest = sample_id_digest(
        phase9["sample_id"]
    )
    actual_digest = sample_id_digest(
        restricted["sample_id"]
    )
    if actual_digest != expected_digest:
        raise Phase9ParityError(
            "restricted reference digest differs from phase-9 universe"
        )
    return restricted


def evaluate_frozen_reference_on_phase9_universe(
    reference_records: pd.DataFrame,
    phase9_rows: pd.DataFrame,
    policy: DecisionPolicy,
) -> dict[str, Any]:
    """Re-evaluate a frozen model's predictions without refitting or reselection."""

    restricted = (
        restrict_reference_to_phase9_universe(
            reference_records,
            phase9_rows,
        )
    )
    base = run_backtest_v1(
        restricted,
        costs=BASE_COSTS,
        policy=policy,
    )
    stress = run_backtest_v1(
        restricted,
        costs=STRESS_COSTS,
        policy=policy,
        decision_costs=BASE_COSTS,
    )
    return {
        "classification": _metrics(
            restricted
        ),
        "base_backtest": base.metrics,
        "stress_backtest": stress.metrics,
        "policy": asdict(policy),
        "sample_digest": sample_id_digest(
            restricted["sample_id"]
        ),
        "rows": len(restricted),
        "reference_refit": False,
        "policy_reselected": False,
    }


__all__ = [
    "Phase9ParityError",
    "evaluate_frozen_reference_on_phase9_universe",
    "restrict_reference_to_phase9_universe",
]
