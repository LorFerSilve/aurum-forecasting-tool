"""Train-only scaling for phase-9 path targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gold_forecasting.phase9.targets import PATH_COMPONENTS, PATH_STEPS


class Phase9NormalizationError(ValueError):
    """Raised when path-target scaling would violate the train-only contract."""


def path_target_array(labels: pd.DataFrame) -> NDArray[np.float64]:
    result = np.empty(
        (len(labels), PATH_STEPS, len(PATH_COMPONENTS)),
        dtype=np.float64,
    )
    for step in range(PATH_STEPS):
        for component_index, component in enumerate(PATH_COMPONENTS):
            column = f"path_step_{step + 1}_{component}"
            if column not in labels:
                raise Phase9NormalizationError(f"missing path target column: {column}")
            result[:, step, component_index] = labels[column].to_numpy(dtype=np.float64)
    if not np.isfinite(result).all():
        raise Phase9NormalizationError("path targets must be finite")
    return result


def aggregate_target_array(labels: pd.DataFrame) -> NDArray[np.float64]:
    columns = [f"path_15m_{component}" for component in PATH_COMPONENTS]
    missing = [column for column in columns if column not in labels]
    if missing:
        raise Phase9NormalizationError(
            "missing aggregate target columns: " + ", ".join(missing)
        )
    result = labels.loc[:, columns].to_numpy(dtype=np.float64)
    if not np.isfinite(result).all():
        raise Phase9NormalizationError("aggregate path targets must be finite")
    return result


@dataclass(frozen=True, slots=True)
class PathTargetNormalizer:
    path_scales: NDArray[np.float64]
    aggregate_scales: NDArray[np.float64]
    cumulative_return_scale: float
    fit_rows: int

    def transform_path(self, values: np.ndarray) -> NDArray[np.float32]:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-2:] != self.path_scales.shape or not np.isfinite(array).all():
            raise Phase9NormalizationError("invalid path array for normalization")
        return (array / self.path_scales).astype(np.float32)

    def inverse_path(self, values: np.ndarray) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-2:] != self.path_scales.shape or not np.isfinite(array).all():
            raise Phase9NormalizationError("invalid normalized path array")
        return array * self.path_scales

    def transform_aggregate(self, values: np.ndarray) -> NDArray[np.float32]:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1:] != self.aggregate_scales.shape or not np.isfinite(array).all():
            raise Phase9NormalizationError("invalid aggregate array for normalization")
        return (array / self.aggregate_scales).astype(np.float32)

    def inverse_aggregate(self, values: np.ndarray) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1:] != self.aggregate_scales.shape or not np.isfinite(array).all():
            raise Phase9NormalizationError("invalid normalized aggregate array")
        return array * self.aggregate_scales

    def as_record(self) -> dict[str, object]:
        return {
            "fit_rows": self.fit_rows,
            "path_components": list(PATH_COMPONENTS),
            "path_scales": self.path_scales.tolist(),
            "aggregate_scales": self.aggregate_scales.tolist(),
            "cumulative_return_scale": self.cumulative_return_scale,
        }


def fit_path_target_normalizer(
    labels: pd.DataFrame,
    row_indices: NDArray[np.int64],
) -> PathTargetNormalizer:
    rows = np.asarray(row_indices, dtype=np.int64)
    if (
        rows.ndim != 1
        or len(rows) == 0
        or (rows < 0).any()
        or (rows >= len(labels)).any()
    ):
        raise Phase9NormalizationError("normalizer requires valid nonempty training rows")
    path = path_target_array(labels)[rows]
    aggregate = aggregate_target_array(labels)[rows]
    path_scales = np.std(path, axis=0)
    aggregate_scales = np.std(aggregate, axis=0)
    path_scales = np.where(path_scales > 1e-3, path_scales, 1.0)
    aggregate_scales = np.where(aggregate_scales > 1e-3, aggregate_scales, 1.0)
    cumulative = labels.iloc[rows][
        "path_15m_close_return_log_bps"
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(cumulative).all():
        raise Phase9NormalizationError("cumulative path returns must be finite")
    cumulative_scale = max(float(np.std(cumulative)), 1e-3)
    return PathTargetNormalizer(
        path_scales=path_scales.astype(np.float64),
        aggregate_scales=aggregate_scales.astype(np.float64),
        cumulative_return_scale=cumulative_scale,
        fit_rows=len(rows),
    )


__all__ = [
    "PathTargetNormalizer",
    "Phase9NormalizationError",
    "aggregate_target_array",
    "fit_path_target_normalizer",
    "path_target_array",
]
