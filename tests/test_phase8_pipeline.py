"""Structural guards for the phase-8 benchmark preflight."""

from __future__ import annotations

import numpy as np
import pytest

from gold_forecasting.phase8.pipeline import (
    _validate_phase7_baseline_metrics,
)


def _valid_baseline() -> dict[str, object]:
    return {
        "macro_f1": {"mean": 0.4, "worst": 0.3},
        "brier": {"mean": 0.6},
        "log_loss": {"mean": 1.0},
        "return_mae_bps": {"mean": 12.0},
        "net_bps": {"worst": 0.0},
    }


def test_phase7_baseline_metric_contract_accepts_finite_reference() -> None:
    _validate_phase7_baseline_metrics(
        _valid_baseline(),
        horizon=15,
    )


@pytest.mark.parametrize(
    ("metric", "field", "value"),
    [
        ("macro_f1", "mean", np.nan),
        ("macro_f1", "worst", np.inf),
        ("brier", "mean", None),
        ("log_loss", "mean", "1.0"),
        ("return_mae_bps", "mean", True),
        ("net_bps", "worst", np.nan),
    ],
)
def test_phase7_baseline_metric_contract_rejects_invalid_values(
    metric: str,
    field: str,
    value: object,
) -> None:
    baseline = _valid_baseline()
    section = dict(baseline[metric])  # type: ignore[arg-type]
    section[field] = value
    baseline[metric] = section

    with pytest.raises(ValueError, match=metric):
        _validate_phase7_baseline_metrics(
            baseline,
            horizon=30,
        )


def test_phase7_baseline_metric_contract_rejects_missing_section() -> None:
    baseline = _valid_baseline()
    del baseline["brier"]

    with pytest.raises(ValueError, match="brier"):
        _validate_phase7_baseline_metrics(
            baseline,
            horizon=60,
        )
