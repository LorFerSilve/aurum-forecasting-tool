from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold_forecasting.phase9.audit import (
    Phase9AuditError,
    _path_return_from_quantiles,
)


def test_path_return_audit_sums_q50_gap_and_body_across_five_steps() -> None:
    frame = pd.DataFrame(
        {
            **{
                f"step_{step}_gap_log_bps_q50": [float(step), -float(step)]
                for step in range(1, 6)
            },
            **{
                f"step_{step}_body_log_bps_q50": [2.0, -3.0]
                for step in range(1, 6)
            },
        }
    )

    result = _path_return_from_quantiles(
        frame,
        clip_log_bps=5_000.0,
    )

    np.testing.assert_allclose(
        result,
        np.array([25.0, -30.0]),
    )


def test_path_return_audit_matches_reconstruction_clipping() -> None:
    frame = pd.DataFrame(
        {
            **{
                f"step_{step}_gap_log_bps_q50": [0.0]
                for step in range(1, 6)
            },
            **{
                f"step_{step}_body_log_bps_q50": [0.0]
                for step in range(1, 6)
            },
        }
    )
    frame.loc[0, "step_1_gap_log_bps_q50"] = 8_000.0
    frame.loc[0, "step_2_body_log_bps_q50"] = -9_000.0

    result = _path_return_from_quantiles(
        frame,
        clip_log_bps=5_000.0,
    )

    assert result[0] == pytest.approx(0.0)


def test_path_return_audit_rejects_missing_persisted_column() -> None:
    with pytest.raises(
        Phase9AuditError,
        match="missing step_1_gap_log_bps_q50",
    ):
        _path_return_from_quantiles(
            pd.DataFrame({"sample_id": ["x"]}),
            clip_log_bps=5_000.0,
        )
