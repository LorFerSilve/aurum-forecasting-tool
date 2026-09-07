"""Leakage-safe training utilities for the phase-8 compact neural challenger."""

from __future__ import annotations

import copy
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.metrics import f1_score  # type: ignore[import-untyped]
from torch import Tensor
from torch.nn import functional as F

from gold_forecasting.classification import validate_probability_matrix
from gold_forecasting.phase8.config import Phase8Config
from gold_forecasting.phase8.model import MultiTimeframeGRU
from gold_forecasting.phase8.sequences import (
    Phase8SequenceBuildResult,
    SEQUENCE_FEATURE_NAMES,
)


class Phase8TrainingError(ValueError):
    """Raised when neural training would violate the phase-8 data contract."""


@dataclass(frozen=True, slots=True)
class SequenceNormalizer:
    medians: dict[str, NDArray[np.float32]]
    means: dict[str, NDArray[np.float32]]
    scales: dict[str, NDArray[np.float32]]
    target_scales: dict[str, float]
    fit_rows: int

    def transform(self, timeframe: str, values: np.ndarray) -> NDArray[np.float32]:
        if timeframe not in self.medians:
            raise Phase8TrainingError(f"normalizer has no statistics for {timeframe}")
        array = np.asarray(values, dtype=np.float32).copy()
        median = self.medians[timeframe]
        mean = self.means[timeframe]
        scale = self.scales[timeframe]
        missing = ~np.isfinite(array)
        if missing.any():
            array = np.where(missing, median.reshape(1, 1, -1), array)
        result = (array - mean.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
        if not np.isfinite(result).all():
            raise Phase8TrainingError("normalizer produced non-finite sequence values")
        return result.astype(np.float32, copy=False)

    def target(self, name: str, values: np.ndarray) -> NDArray[np.float32]:
        scale = self.target_scales[name]
        result = np.asarray(values, dtype=np.float32) / np.float32(scale)
        if not np.isfinite(result).all():
            raise Phase8TrainingError(f"normalized target {name} is non-finite")
        return result

    def inverse_target(self, name: str, values: np.ndarray) -> NDArray[np.float64]:
        return np.asarray(values, dtype=np.float64) * self.target_scales[name]

    def as_record(self) -> dict[str, Any]:
        return {
            "fit_rows": self.fit_rows,
            "feature_names": list(SEQUENCE_FEATURE_NAMES),
            "timeframes": {
                name: {
                    "median": self.medians[name].astype(float).tolist(),
                    "mean": self.means[name].astype(float).tolist(),
                    "scale": self.scales[name].astype(float).tolist(),
                }
                for name in self.medians
            },
            "target_scales": self.target_scales,
        }


@dataclass(frozen=True, slots=True)
class NeuralPrediction:
    probabilities: NDArray[np.float64]
    expected_return_bps: NDArray[np.float64]
    predicted_range_bps: NDArray[np.float64]
    predicted_volatility_bps: NDArray[np.float64]
    mean_fusion_weights: dict[str, float]
    inference_seconds: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: MultiTimeframeGRU
    normalizer: SequenceNormalizer
    history: tuple[dict[str, float | int], ...]
    best_epoch: int
    parameter_count: int
    device: str
    mixed_precision_used: bool
    fit_seconds: float


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise Phase8TrainingError(
                "CUDA was requested but torch.cuda.is_available() is false"
            )
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int, *, deterministic: bool) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def fit_normalizer(
    sequences: Phase8SequenceBuildResult,
    table: pd.DataFrame,
    row_indices: NDArray[np.int64],
    timeframes: tuple[str, ...],
) -> SequenceNormalizer:
    if len(row_indices) == 0:
        raise Phase8TrainingError("normalizer requires nonempty training rows")
    medians: dict[str, NDArray[np.float32]] = {}
    means: dict[str, NDArray[np.float32]] = {}
    scales: dict[str, NDArray[np.float32]] = {}
    for timeframe in timeframes:
        values = sequences.by_timeframe[timeframe].values[row_indices]
        flat = values.reshape(-1, values.shape[-1]).astype(np.float64)
        with np.errstate(all="ignore"):
            median = np.nanmedian(flat, axis=0)
        if not np.isfinite(median).all():
            raise Phase8TrainingError(
                f"training rows contain no observed sequence values for {timeframe}"
            )
        filled = np.where(np.isfinite(flat), flat, median)
        mean = filled.mean(axis=0)
        scale = filled.std(axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        medians[timeframe] = median.astype(np.float32)
        means[timeframe] = mean.astype(np.float32)
        scales[timeframe] = scale.astype(np.float32)
    target_scales: dict[str, float] = {}
    for name in (
        "arithmetic_return_bps",
        "future_range_bps",
        "future_realized_vol_bps",
    ):
        values = table.iloc[row_indices][name].to_numpy(dtype=np.float64)
        scale = float(np.std(values))
        target_scales[name] = max(scale, 1e-3)
    return SequenceNormalizer(
        medians=medians,
        means=means,
        scales=scales,
        target_scales=target_scales,
        fit_rows=len(row_indices),
    )


def _batch_indices(
    rows: NDArray[np.int64],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> Iterator[NDArray[np.int64]]:
    ordered = rows.copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(ordered)
    for start in range(0, len(ordered), batch_size):
        yield ordered[start : start + batch_size]


def _availability_matrix(
    sequences: Phase8SequenceBuildResult,
    rows: NDArray[np.int64],
    timeframes: tuple[str, ...],
) -> NDArray[np.bool_]:
    result = np.column_stack(
        [sequences.by_timeframe[name].available[rows] for name in timeframes]
    ).astype(bool)
    if (~result).all(axis=1).any():
        raise Phase8TrainingError(
            "a training/evaluation sample has no available timeframe"
        )
    return result


def _class_weights(
    table: pd.DataFrame,
    rows: NDArray[np.int64],
    mode: str,
) -> Tensor | None:
    if mode == "none":
        return None
    labels = table.iloc[rows]["target_class_id"].to_numpy(dtype=np.int64)
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if (counts == 0).any():
        raise Phase8TrainingError(
            "balanced class weighting requires all classes in train"
        )
    weights = counts.sum() / (3.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _macro_f1(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(
        f1_score(
            labels,
            probabilities.argmax(axis=1),
            labels=[0, 1, 2],
            average="macro",
        )
    )


def _predict_batches(
    model: MultiTimeframeGRU,
    normalizer: SequenceNormalizer,
    sequences: Phase8SequenceBuildResult,
    rows: NDArray[np.int64],
    timeframes: tuple[str, ...],
    *,
    batch_size: int,
    device: torch.device,
) -> NeuralPrediction:
    model.eval()
    all_probabilities: list[np.ndarray] = []
    all_returns: list[np.ndarray] = []
    all_ranges: list[np.ndarray] = []
    all_volatility: list[np.ndarray] = []
    weight_sums = np.zeros(len(timeframes), dtype=np.float64)
    total = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in _batch_indices(rows, batch_size, shuffle=False, seed=0):
            availability = _availability_matrix(sequences, batch, timeframes)
            tensors = {
                name: torch.from_numpy(
                    normalizer.transform(
                        name,
                        sequences.by_timeframe[name].values[batch],
                    )
                ).to(device)
                for name in timeframes
            }
            output = model(
                tensors,
                torch.from_numpy(availability).to(
                    device=device,
                    dtype=torch.bool,
                ),
            )
            probabilities = torch.softmax(
                output.direction_logits, dim=1
            ).cpu().numpy()
            all_probabilities.append(probabilities)
            all_returns.append(output.normalized_return.cpu().numpy())
            all_ranges.append(output.normalized_range.cpu().numpy())
            all_volatility.append(output.normalized_volatility.cpu().numpy())
            weights = output.fusion_weights.cpu().numpy()
            weight_sums += weights.sum(axis=0)
            total += len(batch)
    probabilities64 = validate_probability_matrix(
        np.concatenate(all_probabilities, axis=0)
    )
    return NeuralPrediction(
        probabilities=probabilities64,
        expected_return_bps=normalizer.inverse_target(
            "arithmetic_return_bps",
            np.concatenate(all_returns),
        ),
        predicted_range_bps=normalizer.inverse_target(
            "future_range_bps",
            np.concatenate(all_ranges),
        ),
        predicted_volatility_bps=normalizer.inverse_target(
            "future_realized_vol_bps",
            np.concatenate(all_volatility),
        ),
        mean_fusion_weights={
            name: float(weight_sums[index] / max(total, 1))
            for index, name in enumerate(timeframes)
        },
        inference_seconds=time.perf_counter() - started,
    )


def train_neural_model(
    sequences: Phase8SequenceBuildResult,
    table: pd.DataFrame,
    train_rows: NDArray[np.int64],
    validation_rows: NDArray[np.int64] | None,
    timeframes: tuple[str, ...],
    config: Phase8Config,
    *,
    seed: int,
    forced_epochs: int | None = None,
) -> TrainingResult:
    """Fit training-only statistics; validation controls epochs but never scaling."""

    if len(train_rows) == 0:
        raise Phase8TrainingError("training rows must be nonempty")
    if forced_epochs is None and (
        validation_rows is None or len(validation_rows) == 0
    ):
        raise Phase8TrainingError(
            "early-stopped training requires nonempty validation rows"
        )
    seed_everything(seed, deterministic=config.deterministic_algorithms)
    device = resolve_device(config.device)
    normalizer = fit_normalizer(sequences, table, train_rows, timeframes)
    model = MultiTimeframeGRU(
        timeframes,
        input_size=len(SEQUENCE_FEATURE_NAMES),
        hidden_size=config.encoder_hidden_size,
        fusion_size=config.fusion_size,
        parameter_budget=config.parameter_budget,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    weights = _class_weights(table, train_rows, config.class_weighting)
    if weights is not None:
        weights = weights.to(device)
    use_amp = bool(config.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    max_epochs = forced_epochs if forced_epochs is not None else config.max_epochs
    if forced_epochs is not None and not 1 <= forced_epochs <= config.max_epochs:
        raise Phase8TrainingError(
            "forced_epochs must be inside the frozen epoch budget"
        )

    history: list[dict[str, float | int]] = []
    best_state: dict[str, Tensor] | None = None
    best_f1 = -math.inf
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        samples = 0
        max_gradient_norm = 0.0
        for batch in _batch_indices(
            train_rows,
            config.batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            availability = _availability_matrix(
                sequences,
                batch,
                timeframes,
            )
            tensors = {
                name: torch.from_numpy(
                    normalizer.transform(
                        name,
                        sequences.by_timeframe[name].values[batch],
                    )
                ).to(device)
                for name in timeframes
            }
            labels = torch.from_numpy(
                table.iloc[batch]["target_class_id"].to_numpy(
                    dtype=np.int64
                )
            ).to(device)
            return_target = torch.from_numpy(
                normalizer.target(
                    "arithmetic_return_bps",
                    table.iloc[batch][
                        "arithmetic_return_bps"
                    ].to_numpy(),
                )
            ).to(device)
            range_target = torch.from_numpy(
                normalizer.target(
                    "future_range_bps",
                    table.iloc[batch]["future_range_bps"].to_numpy(),
                )
            ).to(device)
            volatility_target = torch.from_numpy(
                normalizer.target(
                    "future_realized_vol_bps",
                    table.iloc[batch][
                        "future_realized_vol_bps"
                    ].to_numpy(),
                )
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                output = model(
                    tensors,
                    torch.from_numpy(availability).to(
                        device=device,
                        dtype=torch.bool,
                    ),
                )
                direction_loss = F.cross_entropy(
                    output.direction_logits,
                    labels,
                    weight=weights,
                )
                return_loss = F.huber_loss(
                    output.normalized_return,
                    return_target,
                )
                range_loss = F.huber_loss(
                    output.normalized_range,
                    range_target,
                )
                volatility_loss = F.huber_loss(
                    output.normalized_volatility,
                    volatility_target,
                )
                loss = (
                    config.direction_loss_weight * direction_loss
                    + config.return_loss_weight * return_loss
                    + config.range_loss_weight * range_loss
                    + config.volatility_loss_weight * volatility_loss
                )
            if not torch.isfinite(loss):
                raise Phase8TrainingError("training loss became non-finite")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip_norm,
                )
            )
            if not math.isfinite(gradient_norm):
                raise Phase8TrainingError("gradient norm became non-finite")
            max_gradient_norm = max(
                max_gradient_norm,
                gradient_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu()) * len(batch)
            samples += len(batch)

        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(samples, 1),
            "max_preclip_gradient_norm": max_gradient_norm,
        }
        if forced_epochs is not None:
            history.append(record)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            continue
        assert validation_rows is not None
        validation = _predict_batches(
            model,
            normalizer,
            sequences,
            validation_rows,
            timeframes,
            batch_size=config.batch_size,
            device=device,
        )
        validation_labels = table.iloc[
            validation_rows
        ]["target_class_id"].to_numpy(dtype=np.int64)
        macro_f1 = _macro_f1(
            validation.probabilities,
            validation_labels,
        )
        record["validation_macro_f1"] = macro_f1
        history.append(record)
        if macro_f1 > best_f1 + 1e-12:
            best_f1 = macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break
    if best_state is None or best_epoch < 1:
        raise Phase8TrainingError("training produced no valid checkpoint")
    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        normalizer=normalizer,
        history=tuple(history),
        best_epoch=best_epoch,
        parameter_count=model.parameter_count,
        device=str(device),
        mixed_precision_used=use_amp,
        fit_seconds=time.perf_counter() - started,
    )


def predict_neural_model(
    result: TrainingResult,
    sequences: Phase8SequenceBuildResult,
    rows: NDArray[np.int64],
    timeframes: tuple[str, ...],
    config: Phase8Config,
) -> NeuralPrediction:
    device = next(result.model.parameters()).device
    return _predict_batches(
        result.model,
        result.normalizer,
        sequences,
        rows,
        timeframes,
        batch_size=config.batch_size,
        device=device,
    )


def save_checkpoint(
    result: TrainingResult,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in result.model.state_dict().items()
        },
        "timeframes": result.model.timeframes,
        "parameter_count": result.parameter_count,
        "best_epoch": result.best_epoch,
        "normalizer": result.normalizer.as_record(),
    }
    torch.save(payload, destination)


__all__ = [
    "NeuralPrediction",
    "Phase8TrainingError",
    "SequenceNormalizer",
    "TrainingResult",
    "fit_normalizer",
    "predict_neural_model",
    "resolve_device",
    "save_checkpoint",
    "seed_everything",
    "train_neural_model",
]
