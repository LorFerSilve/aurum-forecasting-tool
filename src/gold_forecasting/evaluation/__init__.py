"""Evaluation metrics, reliability diagnostics, and comparable reports."""

from gold_forecasting.evaluation.metrics import (
    EvaluationError,
    HourlyMetricSummary,
    MetricSummary,
    compute_classification_metrics,
    compute_metrics_by_utc_hour,
)
from gold_forecasting.evaluation.reliability import (
    CONFIDENCE_BUCKET_COUNT,
    ReliabilityBucket,
    build_reliability_table,
)
from gold_forecasting.evaluation.reporting import (
    CALIBRATION_STATUS,
    PROBABILITY_STATUS,
    EvaluationReport,
    ModelEvaluation,
    ModelEvaluationInput,
    SampleAlignmentError,
    assert_identical_sample_ids,
    build_evaluation_report,
    evaluate_model,
    render_evaluation_markdown,
    write_evaluation_report,
)

__all__ = [
    "CALIBRATION_STATUS",
    "CONFIDENCE_BUCKET_COUNT",
    "PROBABILITY_STATUS",
    "EvaluationError",
    "EvaluationReport",
    "HourlyMetricSummary",
    "MetricSummary",
    "ModelEvaluation",
    "ModelEvaluationInput",
    "ReliabilityBucket",
    "SampleAlignmentError",
    "assert_identical_sample_ids",
    "build_evaluation_report",
    "build_reliability_table",
    "compute_classification_metrics",
    "compute_metrics_by_utc_hour",
    "evaluate_model",
    "render_evaluation_markdown",
    "write_evaluation_report",
]
