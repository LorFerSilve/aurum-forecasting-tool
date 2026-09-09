"""Common 15min sample universe for phase-9 path and direction supervision."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

from gold_forecasting.datasets.preprocessing import sample_id_digest
from gold_forecasting.evaluation.walk_forward import validate_development_frame
from gold_forecasting.labels.multihorizon import build_horizon_labels
from gold_forecasting.phase9.targets import build_future_path_targets


class Phase9DatasetError(ValueError):
    """Raised when standard and path targets cannot share an exact sample universe."""


@dataclass(frozen=True, slots=True)
class Phase9DatasetBuildResult:
    table: pd.DataFrame
    diagnostics: dict[str, object]


def build_phase9_dataset(
    candles_1min: pd.DataFrame,
    candles_3min: pd.DataFrame,
    prediction_candidates: pd.DataFrame,
    *,
    neutral_threshold_bps: float = 6.0,
) -> Phase9DatasetBuildResult:
    """Join executable 15min targets and exact five-candle path labels by prediction key."""

    standard = build_horizon_labels(
        candles_1min,
        prediction_candidates,
        horizon_minutes=15,
        neutral_threshold_bps=neutral_threshold_bps,
    )
    path = build_future_path_targets(
        candles_3min,
        prediction_candidates,
    )
    key = [
        "instrument",
        "source",
        "prediction_time_utc",
    ]
    table = standard.labels.merge(
        path.labels,
        on=key,
        how="inner",
        validate="one_to_one",
    ).sort_values(
        key,
        kind="stable",
    ).reset_index(drop=True)
    if table.empty:
        raise Phase9DatasetError(
            "phase-9 common sample universe is empty"
        )

    expected_path_end = (
        table["prediction_time_utc"]
        + pd.Timedelta(minutes=15)
    )
    if not table[
        "path_end_time_utc"
    ].eq(expected_path_end).all():
        raise Phase9DatasetError(
            "future path does not end exactly 15 minutes after prediction"
        )
    if not table[
        "label_end_time_utc"
    ].eq(
        table["prediction_time_utc"]
        + pd.Timedelta(minutes=16)
    ).all():
        raise Phase9DatasetError(
            "executable 15min label timing changed from phase 8"
        )
    if not table[
        "entry_time_utc"
    ].eq(
        table["prediction_time_utc"]
        + pd.Timedelta(minutes=1)
    ).all():
        raise Phase9DatasetError(
            "phase-9 executable entry must retain the one-minute latency"
        )

    table["sample_id"] = [
        hashlib.sha256(
            (
                f"{instrument}|{source}|"
                f"{stamp.isoformat()}|15"
            ).encode()
        ).hexdigest()
        for instrument, source, stamp
        in table[key].itertuples(
            index=False,
            name=None,
        )
    ]
    validate_development_frame(table)
    diagnostics: dict[str, object] = {
        "candidate_rows": len(prediction_candidates),
        "standard_15m_eligible": (
            standard.output_row_count
        ),
        "path_eligible": path.output_row_count,
        "common_eligible": len(table),
        "dropped_from_common": (
            len(prediction_candidates)
            - len(table)
        ),
        "sample_digest": sample_id_digest(
            table["sample_id"]
        ),
        "timing": {
            "path": "[t,t+15min]",
            "execution_label": "[t+1min,t+16min]",
        },
    }
    return Phase9DatasetBuildResult(
        table=table,
        diagnostics=diagnostics,
    )


__all__ = [
    "Phase9DatasetBuildResult",
    "Phase9DatasetError",
    "build_phase9_dataset",
]
