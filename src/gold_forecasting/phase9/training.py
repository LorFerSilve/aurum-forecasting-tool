"""Leakage-safe training for the phase-9 direct future-path challenger."""

from __future__ import annotations

import copy
import math
import os
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.metrics import f1_score  # type: ignore[import-untyped]
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.nn import functional as F

from gold_forecasting.classification import validate_probability_matrix
from gold_forecasting.phase8.sequences import (
    SEQUENCE_FEATURE_NAMES,
    Phase8SequenceBuildResult,
)
from gold_forecasting.phase8.training import (
    SequenceNormalizer,
    fit_normalizer,
    resolve_device,
    seed_everything,
)
from gold_forecasting.phase9.config import Phase9Config
from gold_forecasting.phase9.losses import (
    aggregate_path_representation_torch,
    pinball_loss,
)
from gold_forecasting.phase9.model import (
    FuturePathGRU,
    RecursiveFuturePathGRU,
)
from gold_forecasting.phase9.normalization import (
    PathTargetNormalizer,
    aggregate_target_array,
    fit_path_target_normalizer,
    path_target_array,
)


class Phase9TrainingError(ValueError):
    """Raised when phase-9 training violates its frozen data/training contract."""


@dataclass(frozen=True, slots=True)
class Phase9Prediction:
    probabilities: NDArray[np.float64]
    expected_return_bps: NDArray[np.float64]
    predicted_range_bps: NDArray[np.float64]
    predicted_volatility_bps: NDArray[np.float64]
    path_quantiles_log_bps: NDArray[np.float64]
    aggregate_quantiles_log_bps: NDArray[np.float64]
    mean_fusion_weights: dict[str, float]
    inference_seconds: float


@dataclass(frozen=True, slots=True)
class Phase9TrainingResult:
    model: FuturePathGRU
    sequence_normalizer: SequenceNormalizer
    path_normalizer: PathTargetNormalizer
    history: tuple[dict[str, float | int], ...]
    best_epoch: int
    parameter_count: int
    device: str
    mixed_precision_used: bool
    optimizer_steps: int
    amp_skipped_steps: int
    fit_seconds: float
    model_variant: Literal["direct", "recursive"]


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
        [
            sequences.by_timeframe[name].available[rows]
            for name in timeframes
        ]
    ).astype(bool)
    if (~result).all(axis=1).any():
        raise Phase9TrainingError(
            "a phase-9 sample has no available configured timeframe"
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
        raise Phase9TrainingError(
            "balanced direction loss requires all classes in training"
        )
    weights = counts.sum() / (3.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _gradients_are_finite(model: nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return bool(
        gradients
        and all(
            torch.isfinite(gradient).all().item()
            for gradient in gradients
        )
    )


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
    model: FuturePathGRU,
    sequence_normalizer: SequenceNormalizer,
    path_normalizer: PathTargetNormalizer,
    sequences: Phase8SequenceBuildResult,
    rows: NDArray[np.int64],
    timeframes: tuple[str, ...],
    *,
    batch_size: int,
    device: torch.device,
) -> Phase9Prediction:
    if len(rows) == 0:
        raise Phase9TrainingError("prediction rows must be nonempty")
    model.eval()
    all_probabilities: list[np.ndarray] = []
    all_returns: list[np.ndarray] = []
    all_ranges: list[np.ndarray] = []
    all_volatility: list[np.ndarray] = []
    all_path: list[np.ndarray] = []
    all_aggregate: list[np.ndarray] = []
    weight_sums = np.zeros(len(timeframes), dtype=np.float64)
    total = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in _batch_indices(
            rows,
            batch_size,
            shuffle=False,
            seed=0,
        ):
            availability = _availability_matrix(
                sequences,
                batch,
                timeframes,
            )
            tensors = {
                name: torch.from_numpy(
                    sequence_normalizer.transform(
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
                output.direction_logits.to(dtype=torch.float64),
                dim=1,
            ).cpu().numpy()
            all_probabilities.append(probabilities)
            all_returns.append(
                output.normalized_return.float().cpu().numpy()
            )
            all_ranges.append(
                output.normalized_range.float().cpu().numpy()
            )
            all_volatility.append(
                output.normalized_volatility.float().cpu().numpy()
            )
            all_path.append(
                output.path_quantiles.float().cpu().numpy()
            )
            all_aggregate.append(
                output.aggregate_quantiles.float().cpu().numpy()
            )
            weights = output.fusion_weights.float().cpu().numpy()
            weight_sums += weights.sum(axis=0)
            total += len(batch)

    probabilities64 = validate_probability_matrix(
        np.concatenate(all_probabilities, axis=0)
    )
    path_normalized = np.concatenate(all_path, axis=0)
    aggregate_normalized = np.concatenate(all_aggregate, axis=0)
    return Phase9Prediction(
        probabilities=probabilities64,
        expected_return_bps=sequence_normalizer.inverse_target(
            "arithmetic_return_bps",
            np.concatenate(all_returns),
        ),
        predicted_range_bps=sequence_normalizer.inverse_target(
            "future_range_bps",
            np.concatenate(all_ranges),
        ),
        predicted_volatility_bps=sequence_normalizer.inverse_target(
            "future_realized_vol_bps",
            np.concatenate(all_volatility),
        ),
        path_quantiles_log_bps=path_normalizer.inverse_path_quantiles(
            path_normalized,
        ),
        aggregate_quantiles_log_bps=(
            path_normalizer.inverse_aggregate_quantiles(
                aggregate_normalized,
            )
        ),
        mean_fusion_weights={
            name: float(weight_sums[index] / max(total, 1))
            for index, name in enumerate(timeframes)
        },
        inference_seconds=time.perf_counter() - started,
    )


def train_phase9_model(
    sequences: Phase8SequenceBuildResult,
    table: pd.DataFrame,
    train_rows: NDArray[np.int64],
    validation_rows: NDArray[np.int64] | None,
    config: Phase9Config,
    *,
    seed: int,
    forced_epochs: int | None = None,
    model_variant: Literal["direct", "recursive"] = "direct",
) -> Phase9TrainingResult:
    """Fit all scaling only on train; validation may select epochs, never scales."""

    rows = np.asarray(train_rows, dtype=np.int64)
    if len(rows) == 0:
        raise Phase9TrainingError("training rows must be nonempty")
    if forced_epochs is None and (
        validation_rows is None
        or len(validation_rows) == 0
    ):
        raise Phase9TrainingError(
            "early-stopped phase-9 training requires validation rows"
        )
    if forced_epochs is not None and not 1 <= forced_epochs <= config.max_epochs:
        raise Phase9TrainingError(
            "forced_epochs must remain inside the frozen epoch budget"
        )

    seed_everything(
        seed,
        deterministic=config.deterministic_algorithms,
    )
    device = resolve_device(config.device)
    timeframes = config.timeframes
    sequence_normalizer = fit_normalizer(
        sequences,
        table,
        rows,
        timeframes,
    )
    path_normalizer = fit_path_target_normalizer(
        table,
        rows,
    )
    all_path_targets = path_target_array(table)
    all_aggregate_targets = aggregate_target_array(table)
    cumulative_targets = table[
        "path_15m_close_return_log_bps"
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(cumulative_targets).all():
        raise Phase9TrainingError(
            "cumulative path return target is non-finite"
        )

    model_type = (
        FuturePathGRU
        if model_variant == "direct"
        else RecursiveFuturePathGRU
    )
    model = model_type(
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
    class_weights = _class_weights(
        table,
        rows,
        config.class_weighting,
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)

    use_amp = bool(
        config.mixed_precision
        and device.type == "cuda"
    )
    scaler = GradScaler(
        "cuda",
        enabled=use_amp,
    )
    max_epochs = (
        forced_epochs
        if forced_epochs is not None
        else config.max_epochs
    )
    path_scales = torch.tensor(
        path_normalizer.path_scales,
        dtype=torch.float32,
        device=device,
    )
    aggregate_scales = torch.tensor(
        path_normalizer.aggregate_scales,
        dtype=torch.float32,
        device=device,
    )
    cumulative_scale = torch.tensor(
        path_normalizer.cumulative_return_scale,
        dtype=torch.float32,
        device=device,
    )

    history: list[dict[str, float | int]] = []
    best_state: dict[str, Tensor] | None = None
    best_f1 = -math.inf
    best_epoch = 0
    stale_epochs = 0
    optimizer_steps = 0
    amp_skipped_steps = 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        samples = 0
        max_gradient_norm = 0.0
        epoch_optimizer_steps = 0
        epoch_amp_skipped_steps = 0

        for batch in _batch_indices(
            rows,
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
                    sequence_normalizer.transform(
                        name,
                        sequences.by_timeframe[name].values[batch],
                    )
                ).to(device)
                for name in timeframes
            }
            labels = torch.from_numpy(
                table.iloc[batch][
                    "target_class_id"
                ].to_numpy(dtype=np.int64)
            ).to(device)
            return_target = torch.from_numpy(
                sequence_normalizer.target(
                    "arithmetic_return_bps",
                    table.iloc[batch][
                        "arithmetic_return_bps"
                    ].to_numpy(),
                )
            ).to(device)
            range_target = torch.from_numpy(
                sequence_normalizer.target(
                    "future_range_bps",
                    table.iloc[batch][
                        "future_range_bps"
                    ].to_numpy(),
                )
            ).to(device)
            volatility_target = torch.from_numpy(
                sequence_normalizer.target(
                    "future_realized_vol_bps",
                    table.iloc[batch][
                        "future_realized_vol_bps"
                    ].to_numpy(),
                )
            ).to(device)
            path_target = torch.from_numpy(
                path_normalizer.transform_path(
                    all_path_targets[batch]
                )
            ).to(device)
            aggregate_target = torch.from_numpy(
                path_normalizer.transform_aggregate(
                    all_aggregate_targets[batch]
                )
            ).to(device)
            cumulative_target = torch.from_numpy(
                (
                    cumulative_targets[batch]
                    / path_normalizer.cumulative_return_scale
                ).astype(np.float32)
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
                    weight=class_weights,
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

                path_quantiles = output.path_quantiles.float()
                aggregate_quantiles = output.aggregate_quantiles.float()
                path_loss = pinball_loss(
                    path_quantiles,
                    path_target.float(),
                )
                aggregate_loss = pinball_loss(
                    aggregate_quantiles,
                    aggregate_target.float(),
                )

                path_median_physical = (
                    path_quantiles[..., 1]
                    * path_scales
                )
                derived_aggregate = (
                    aggregate_path_representation_torch(
                        path_median_physical,
                        clip_log_bps=(
                            config.reconstruction_clip_log_bps
                        ),
                    )
                )
                direct_aggregate = (
                    aggregate_quantiles[..., 1]
                    * aggregate_scales
                )
                consistency_loss = F.huber_loss(
                    direct_aggregate / aggregate_scales,
                    derived_aggregate / aggregate_scales,
                )
                cumulative_prediction = (
                    derived_aggregate[:, 0]
                    + derived_aggregate[:, 1]
                ) / cumulative_scale
                cumulative_loss = F.huber_loss(
                    cumulative_prediction,
                    cumulative_target.float(),
                )
                loss = (
                    config.direction_loss_weight
                    * direction_loss
                    + config.return_loss_weight
                    * return_loss
                    + config.range_loss_weight
                    * range_loss
                    + config.volatility_loss_weight
                    * volatility_loss
                    + config.path_quantile_loss_weight
                    * path_loss
                    + config.aggregate_quantile_loss_weight
                    * aggregate_loss
                    + config.cumulative_return_loss_weight
                    * cumulative_loss
                    + config.temporal_consistency_loss_weight
                    * consistency_loss
                )

            if not torch.isfinite(loss):
                raise Phase9TrainingError(
                    "phase-9 training loss became non-finite"
                )
            torch.autograd.backward(
                scaler.scale(loss)
            )
            scaler.unscale_(optimizer)
            if not _gradients_are_finite(model):
                if not use_amp:
                    raise Phase9TrainingError(
                        "phase-9 full-precision gradients became non-finite"
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                amp_skipped_steps += 1
                epoch_amp_skipped_steps += 1
                epoch_loss += (
                    float(loss.detach().cpu())
                    * len(batch)
                )
                samples += len(batch)
                continue

            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip_norm,
                    error_if_nonfinite=True,
                )
            )
            if not math.isfinite(gradient_norm):
                raise Phase9TrainingError(
                    "phase-9 gradient norm became non-finite"
                )
            max_gradient_norm = max(
                max_gradient_norm,
                gradient_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            epoch_optimizer_steps += 1
            epoch_loss += (
                float(loss.detach().cpu())
                * len(batch)
            )
            samples += len(batch)

        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": (
                epoch_loss / max(samples, 1)
            ),
            "max_preclip_gradient_norm": (
                max_gradient_norm
            ),
            "optimizer_steps": (
                epoch_optimizer_steps
            ),
            "amp_skipped_steps": (
                epoch_amp_skipped_steps
            ),
        }
        if forced_epochs is not None:
            history.append(record)
            best_state = copy.deepcopy(
                model.state_dict()
            )
            best_epoch = epoch
            continue

        assert validation_rows is not None
        validation = _predict_batches(
            model,
            sequence_normalizer,
            path_normalizer,
            sequences,
            np.asarray(
                validation_rows,
                dtype=np.int64,
            ),
            timeframes,
            batch_size=config.batch_size,
            device=device,
        )
        validation_labels = table.iloc[
            validation_rows
        ]["target_class_id"].to_numpy(
            dtype=np.int64
        )
        macro_f1 = _macro_f1(
            validation.probabilities,
            validation_labels,
        )
        record["validation_macro_f1"] = macro_f1
        history.append(record)

        if macro_f1 > best_f1 + 1e-12:
            best_f1 = macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            stale_epochs = 0
        else:
            stale_epochs += 1
            if (
                stale_epochs
                >= config.early_stopping_patience
            ):
                break

    if optimizer_steps < 1:
        raise Phase9TrainingError(
            "phase-9 training completed without a finite optimizer update"
        )
    if best_state is None or best_epoch < 1:
        raise Phase9TrainingError(
            "phase-9 training produced no valid checkpoint"
        )
    model.load_state_dict(best_state)
    return Phase9TrainingResult(
        model=model,
        sequence_normalizer=sequence_normalizer,
        path_normalizer=path_normalizer,
        history=tuple(history),
        best_epoch=best_epoch,
        parameter_count=model.parameter_count,
        device=str(device),
        mixed_precision_used=use_amp,
        optimizer_steps=optimizer_steps,
        amp_skipped_steps=amp_skipped_steps,
        fit_seconds=(
            time.perf_counter() - started
        ),
        model_variant=model_variant,
    )


def predict_phase9_model(
    result: Phase9TrainingResult,
    sequences: Phase8SequenceBuildResult,
    rows: NDArray[np.int64],
    config: Phase9Config,
) -> Phase9Prediction:
    device = next(
        result.model.parameters()
    ).device
    return _predict_batches(
        result.model,
        result.sequence_normalizer,
        result.path_normalizer,
        sequences,
        np.asarray(rows, dtype=np.int64),
        config.timeframes,
        batch_size=config.batch_size,
        device=device,
    )


def save_phase9_checkpoint(
    result: Phase9TrainingResult,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    payload = {
        "state_dict": {
            name: tensor.detach().cpu()
            for name, tensor
            in result.model.state_dict().items()
        },
        "timeframes": result.model.timeframes,
        "parameter_count": result.parameter_count,
        "best_epoch": result.best_epoch,
        "mixed_precision_used": (
            result.mixed_precision_used
        ),
        "optimizer_steps": result.optimizer_steps,
        "amp_skipped_steps": result.amp_skipped_steps,
        "model_variant": result.model_variant,
        "sequence_normalizer": (
            result.sequence_normalizer.as_record()
        ),
        "path_normalizer": (
            result.path_normalizer.as_record()
        ),
    }
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(
            payload,
            temporary,
        )
        os.replace(
            temporary,
            destination,
        )
    finally:
        temporary.unlink(
            missing_ok=True
        )


__all__ = [
    "Phase9Prediction",
    "Phase9TrainingError",
    "Phase9TrainingResult",
    "predict_phase9_model",
    "save_phase9_checkpoint",
    "train_phase9_model",
]
