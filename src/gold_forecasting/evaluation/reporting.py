"""Comparable and audit-friendly evaluation reports for the research MVP."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from numpy.typing import ArrayLike

from gold_forecasting.artifacts import write_json_atomic, write_text_atomic
from gold_forecasting.classification import CLASS_ORDER, PredictionBatch
from gold_forecasting.evaluation.metrics import (
    EvaluationError,
    HourlyMetricSummary,
    MetricSummary,
    compute_classification_metrics,
    compute_metrics_by_utc_hour,
)
from gold_forecasting.evaluation.reliability import (
    ReliabilityBucket,
    build_reliability_table,
)

ProbabilityStatus = Literal["preliminary"]
CalibrationStatus = Literal["uncalibrated"]
PROBABILITY_STATUS: ProbabilityStatus = "preliminary"
CALIBRATION_STATUS: CalibrationStatus = "uncalibrated"


class SampleAlignmentError(EvaluationError):
    """Raised when compared models do not use identical ordered samples."""


def _sample_tuple(values: object, *, model_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise SampleAlignmentError(f"sample_ids for {model_name!r} must be a collection")
    if not isinstance(values, Iterable):
        raise SampleAlignmentError(f"sample_ids for {model_name!r} must be iterable")
    raw_samples = tuple(values)
    if not raw_samples:
        raise SampleAlignmentError(f"sample_ids for {model_name!r} must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in raw_samples):
        raise SampleAlignmentError(f"sample_ids for {model_name!r} must be non-empty strings")
    samples = cast(tuple[str, ...], raw_samples)
    if len(samples) != len(set(samples)):
        raise SampleAlignmentError(f"sample_ids for {model_name!r} contain duplicates")
    return samples


def assert_identical_sample_ids(
    model_sample_ids: Mapping[str, object],
) -> tuple[str, ...]:
    """Return the shared IDs only when every model has the exact same order."""

    if not model_sample_ids:
        raise SampleAlignmentError("at least one model is required")
    canonical_name: str | None = None
    canonical: tuple[str, ...] | None = None
    for name, raw_ids in model_sample_ids.items():
        if not isinstance(name, str) or not name.strip():
            raise SampleAlignmentError("model names must be non-empty strings")
        samples = _sample_tuple(raw_ids, model_name=name)
        if canonical is None:
            canonical_name = name
            canonical = samples
        elif samples != canonical:
            same_members = len(samples) == len(canonical) and set(samples) == set(canonical)
            detail = "same records in a different order" if same_members else "different records"
            raise SampleAlignmentError(
                f"model {name!r} uses {detail} than model {canonical_name!r}"
            )
    if canonical is None:  # pragma: no cover - guarded by the non-empty mapping check
        raise SampleAlignmentError("at least one model is required")
    return canonical


def _sample_digest(sample_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sample_ids:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class ModelEvaluationInput:
    """Predictions and sample identity required for one comparable model."""

    model_name: str
    sample_ids: tuple[str, ...]
    predictions: PredictionBatch
    issued_mask: ArrayLike | None = None


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    model_name: str
    class_order: tuple[str, ...]
    probability_status: ProbabilityStatus
    calibration_status: CalibrationStatus
    metrics: MetricSummary
    per_utc_hour: tuple[HourlyMetricSummary, ...]
    reliability: tuple[ReliabilityBucket, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "class_order": list(self.class_order),
            "probability_status": self.probability_status,
            "calibration_status": self.calibration_status,
            "metrics": self.metrics.to_dict(),
            "per_utc_hour": [row.to_dict() for row in self.per_utc_hour],
            "reliability": [row.to_dict() for row in self.reliability],
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    schema_version: Literal[1]
    class_order: tuple[str, ...]
    probability_status: ProbabilityStatus
    calibration_status: CalibrationStatus
    reliability_bucket_count: int
    sample_count: int
    sample_id_digest: str
    models: tuple[ModelEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "class_order": list(self.class_order),
            "probability_status": self.probability_status,
            "calibration_status": self.calibration_status,
            "reliability_bucket_count": self.reliability_bucket_count,
            "sample_count": self.sample_count,
            "sample_id_digest": self.sample_id_digest,
            "models": [model.to_dict() for model in self.models],
        }


def evaluate_model(
    model_name: str,
    sample_ids: object,
    y_true: object,
    probabilities: ArrayLike,
    prediction_times_utc: object,
    *,
    issued_mask: ArrayLike | None = None,
    reliability_bucket_count: int = 10,
) -> ModelEvaluation:
    """Evaluate one model while retaining explicit probability caveats."""

    if not isinstance(model_name, str) or not model_name.strip():
        raise EvaluationError("model_name must be a non-empty string")
    samples = _sample_tuple(sample_ids, model_name=model_name)
    metrics = compute_classification_metrics(
        y_true,
        probabilities,
        issued_mask=issued_mask,
    )
    if len(samples) != metrics.eligible_count:
        raise SampleAlignmentError("sample_ids must align with prediction rows")
    return ModelEvaluation(
        model_name=model_name,
        class_order=tuple(CLASS_ORDER),
        probability_status=PROBABILITY_STATUS,
        calibration_status=CALIBRATION_STATUS,
        metrics=metrics,
        per_utc_hour=compute_metrics_by_utc_hour(
            y_true,
            probabilities,
            prediction_times_utc,
            issued_mask=issued_mask,
        ),
        reliability=build_reliability_table(
            y_true,
            probabilities,
            issued_mask=issued_mask,
            bucket_count=reliability_bucket_count,
        ),
    )


def build_evaluation_report(
    y_true: object,
    prediction_times_utc: object,
    models: Sequence[ModelEvaluationInput],
    *,
    reliability_bucket_count: int = 10,
) -> EvaluationReport:
    """Evaluate all models only after exact ordered sample alignment succeeds."""

    if not models:
        raise EvaluationError("at least one model is required")
    names = [model.model_name for model in models]
    if len(names) != len(set(names)):
        raise EvaluationError("model names must be unique")
    sample_ids = assert_identical_sample_ids(
        {model.model_name: model.sample_ids for model in models}
    )
    evaluations = tuple(
        evaluate_model(
            model.model_name,
            model.sample_ids,
            y_true,
            model.predictions.probabilities,
            prediction_times_utc,
            issued_mask=model.issued_mask,
            reliability_bucket_count=reliability_bucket_count,
        )
        for model in models
    )
    return EvaluationReport(
        schema_version=1,
        class_order=tuple(CLASS_ORDER),
        probability_status=PROBABILITY_STATUS,
        calibration_status=CALIBRATION_STATUS,
        reliability_bucket_count=reliability_bucket_count,
        sample_count=len(sample_ids),
        sample_id_digest=_sample_digest(sample_ids),
        models=evaluations,
    )


def render_evaluation_markdown(report: EvaluationReport) -> str:
    """Render a concise human-readable companion to the JSON audit artifact."""

    lines = [
        "# MVP evaluation report",
        "",
        "> Model probabilities are preliminary and uncalibrated.",
        "",
        f"- Class order: `{', '.join(report.class_order)}`",
        f"- Compared samples: {report.sample_count}",
        f"- Reliability buckets: {report.reliability_bucket_count}",
        f"- Sample ID digest: `{report.sample_id_digest}`",
        "",
        "| Model | Coverage | Accuracy | Balanced accuracy | Macro-F1 | Log loss | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in report.models:
        metrics = model.metrics
        values = (
            metrics.coverage,
            metrics.accuracy,
            metrics.balanced_accuracy,
            metrics.macro_f1,
            metrics.log_loss,
            metrics.multiclass_brier,
        )
        rendered = ["n/a" if value is None else f"{value:.6f}" for value in values]
        lines.append(f"| {model.model_name} | {' | '.join(rendered)} |")
    lines.append("")

    for model in report.models:
        lines.extend(
            [
                f"## {model.model_name} diagnostics",
                "",
                "### Metrics per UTC hour",
                "",
                "| UTC hour | Eligible | Issued | Coverage | Accuracy | Macro-F1 |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in model.per_utc_hour:
            metrics = row.metrics
            accuracy = "n/a" if metrics.accuracy is None else f"{metrics.accuracy:.6f}"
            macro_f1 = "n/a" if metrics.macro_f1 is None else f"{metrics.macro_f1:.6f}"
            lines.append(
                f"| {row.hour_utc:02d} | {metrics.eligible_count} | {metrics.issued_count} "
                f"| {metrics.coverage:.6f} | {accuracy} | {macro_f1} |"
            )
        lines.extend(
            [
                "",
                "### Confidence reliability",
                "",
                "| Bucket | Samples | Share issued | Mean confidence | Accuracy | Gap |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for bucket in model.reliability:
            close = "]" if bucket.upper_inclusive else ")"
            interval = f"[{bucket.lower_bound:.1f}, {bucket.upper_bound:.1f}{close}"
            mean_confidence = (
                "n/a" if bucket.mean_confidence is None else f"{bucket.mean_confidence:.6f}"
            )
            empirical_accuracy = (
                "n/a"
                if bucket.empirical_accuracy is None
                else f"{bucket.empirical_accuracy:.6f}"
            )
            calibration_gap = (
                "n/a" if bucket.calibration_gap is None else f"{bucket.calibration_gap:.6f}"
            )
            lines.append(
                f"| {interval} | {bucket.sample_count} | {bucket.fraction_of_issued:.6f} "
                f"| {mean_confidence} | {empirical_accuracy} | {calibration_gap} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_evaluation_report(
    report: EvaluationReport,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Atomically persist machine- and human-readable evaluation artifacts."""

    write_json_atomic(json_path, report.to_dict())
    write_text_atomic(markdown_path, render_evaluation_markdown(report))


__all__ = [
    "CALIBRATION_STATUS",
    "PROBABILITY_STATUS",
    "CalibrationStatus",
    "EvaluationReport",
    "ModelEvaluation",
    "ModelEvaluationInput",
    "ProbabilityStatus",
    "SampleAlignmentError",
    "assert_identical_sample_ids",
    "build_evaluation_report",
    "evaluate_model",
    "render_evaluation_markdown",
    "write_evaluation_report",
]
