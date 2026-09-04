from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gold_forecasting.classification import CLASS_ORDER, PredictionBatch
from gold_forecasting.evaluation import (
    EvaluationError,
    ModelEvaluationInput,
    SampleAlignmentError,
    assert_identical_sample_ids,
    build_evaluation_report,
    build_reliability_table,
    compute_classification_metrics,
    compute_metrics_by_utc_hour,
    render_evaluation_markdown,
    write_evaluation_report,
)


def _fixture() -> tuple[tuple[str, ...], np.ndarray, pd.DatetimeIndex]:
    labels = ("down", "neutral", "up", "up")
    probabilities = np.asarray(
        [
            [0.80, 0.10, 0.10],
            [0.20, 0.60, 0.20],
            [0.10, 0.20, 0.70],
            [0.60, 0.20, 0.20],
        ],
        dtype=np.float64,
    )
    times = pd.to_datetime(
        [
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:03:00Z",
            "2024-01-01T01:00:00Z",
            "2024-01-01T01:03:00Z",
        ],
        utc=True,
    )
    return labels, probabilities, times


def test_metrics_use_fixed_class_order_and_multiclass_brier() -> None:
    labels, probabilities, _ = _fixture()

    result = compute_classification_metrics(labels, probabilities)

    assert result.eligible_count == 4
    assert result.issued_count == 4
    assert result.coverage == 1.0
    assert result.accuracy == 0.75
    assert result.balanced_accuracy == pytest.approx(5 / 6)
    assert result.macro_f1 == pytest.approx((2 / 3 + 1.0 + 2 / 3) / 3)
    assert result.confusion_matrix == ((1, 0, 0), (0, 1, 0), (1, 0, 1))
    expected_one_hot = np.eye(3)[[0, 1, 2, 2]]
    expected_brier = np.mean(np.sum((probabilities - expected_one_hot) ** 2, axis=1))
    assert result.multiclass_brier == pytest.approx(expected_brier)
    assert result.log_loss is not None and result.log_loss > 0.0


def test_selective_metrics_report_coverage_and_support_no_signals() -> None:
    labels, probabilities, _ = _fixture()

    selected = compute_classification_metrics(
        labels,
        probabilities,
        issued_mask=np.asarray([True, False, True, False]),
    )
    empty = compute_classification_metrics(
        labels,
        probabilities,
        issued_mask=np.zeros(4, dtype=bool),
    )

    assert selected.coverage == 0.5
    assert selected.accuracy == 1.0
    assert selected.confusion_matrix == ((1, 0, 0), (0, 0, 0), (0, 0, 1))
    assert empty.coverage == 0.0
    assert empty.accuracy is None
    assert empty.log_loss is None
    assert empty.confusion_matrix == ((0, 0, 0),) * 3


def test_issued_mask_must_be_boolean_and_aligned() -> None:
    labels, probabilities, _ = _fixture()

    with pytest.raises(EvaluationError, match="booleans"):
        compute_classification_metrics(labels, probabilities, issued_mask=[1, 0, 1, 0])
    with pytest.raises(EvaluationError, match="align"):
        compute_classification_metrics(labels, probabilities, issued_mask=[True])


def test_hourly_metrics_include_all_utc_hours_and_convert_offsets() -> None:
    labels, probabilities, times = _fixture()
    offset_times = [timestamp.tz_convert("Europe/Brussels") for timestamp in times]

    rows = compute_metrics_by_utc_hour(labels, probabilities, offset_times)

    assert len(rows) == 24
    assert [row.hour_utc for row in rows] == list(range(24))
    assert rows[0].metrics.eligible_count == 2
    assert rows[0].metrics.accuracy == 1.0
    assert rows[1].metrics.eligible_count == 2
    assert rows[1].metrics.accuracy == 0.5
    assert rows[2].metrics.eligible_count == 0
    assert rows[2].metrics.accuracy is None


def test_reliability_has_ten_buckets_and_includes_exact_one() -> None:
    labels = ("down", "neutral", "up")
    probabilities = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.05, 0.90, 0.05],
            [0.15, 0.15, 0.70],
        ]
    )

    buckets = build_reliability_table(labels, probabilities)

    assert len(buckets) == 10
    assert sum(bucket.sample_count for bucket in buckets) == 3
    assert buckets[9].upper_inclusive is True
    assert buckets[9].sample_count == 2
    assert buckets[9].mean_confidence == pytest.approx(0.95)
    assert buckets[7].sample_count == 1
    assert buckets[7].lower_bound == pytest.approx(0.7)
    assert buckets[7].upper_bound == pytest.approx(0.8)


def test_reliability_bucket_count_is_configurable_and_validated() -> None:
    labels, probabilities, _ = _fixture()

    buckets = build_reliability_table(labels, probabilities, bucket_count=4)

    assert len(buckets) == 4
    assert buckets[-1].upper_inclusive is True
    assert sum(bucket.sample_count for bucket in buckets) == len(labels)
    with pytest.raises(EvaluationError, match=r"\[2, 100\]"):
        build_reliability_table(labels, probabilities, bucket_count=1)


def test_sample_alignment_rejects_reordering_different_rows_and_duplicates() -> None:
    assert assert_identical_sample_ids(
        {"baseline": ("a", "b"), "model": ("a", "b")}
    ) == ("a", "b")
    with pytest.raises(SampleAlignmentError, match="different order"):
        assert_identical_sample_ids({"baseline": ("a", "b"), "model": ("b", "a")})
    with pytest.raises(SampleAlignmentError, match="different records"):
        assert_identical_sample_ids({"baseline": ("a", "b"), "model": ("a", "c")})
    with pytest.raises(SampleAlignmentError, match="duplicates"):
        assert_identical_sample_ids({"model": ("a", "a")})


def test_report_is_comparable_and_marks_probabilities_uncalibrated(tmp_path: Path) -> None:
    labels, probabilities, times = _fixture()
    sample_ids = tuple(f"sample-{index}" for index in range(len(labels)))
    batch = PredictionBatch.from_probabilities(probabilities)
    report = build_evaluation_report(
        labels,
        times,
        [
            ModelEvaluationInput("baseline", sample_ids, batch),
            ModelEvaluationInput("logistic", sample_ids, batch),
        ],
        reliability_bucket_count=5,
    )

    assert report.class_order == CLASS_ORDER
    assert report.probability_status == "preliminary"
    assert report.calibration_status == "uncalibrated"
    assert report.reliability_bucket_count == 5
    assert report.sample_count == 4
    assert report.sample_id_digest.startswith("sha256:")
    assert all(model.probability_status == "preliminary" for model in report.models)
    assert all(model.calibration_status == "uncalibrated" for model in report.models)
    assert all(model.class_order == CLASS_ORDER for model in report.models)
    markdown = render_evaluation_markdown(report)
    assert "preliminary and uncalibrated" in markdown
    assert "Metrics per UTC hour" in markdown
    assert "[0.8, 1.0]" in markdown

    json_path = tmp_path / "evaluation.json"
    markdown_path = tmp_path / "evaluation.md"
    write_evaluation_report(report, json_path=json_path, markdown_path=markdown_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["class_order"] == ["down", "neutral", "up"]
    assert payload["reliability_bucket_count"] == 5
    assert payload["models"][0]["metrics"]["confusion_matrix"] == [
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
    ]
    assert markdown_path.read_text(encoding="utf-8") == markdown


def test_report_rejects_same_samples_in_a_different_order() -> None:
    labels, probabilities, times = _fixture()
    batch = PredictionBatch.from_probabilities(probabilities)

    with pytest.raises(SampleAlignmentError, match="different order"):
        build_evaluation_report(
            labels,
            times,
            [
                ModelEvaluationInput("first", ("a", "b", "c", "d"), batch),
                ModelEvaluationInput("second", ("a", "c", "b", "d"), batch),
            ],
        )
