from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from gold_forecasting.evaluation.walk_forward import (
    InnerFold,
    TimeBlock,
    WalkForwardError,
    WalkForwardFold,
    make_walk_forward_folds,
    required_gap_minutes,
    select_block,
    summarize_fold_metrics,
    validate_development_frame,
)


def _block(start: str, end: str) -> TimeBlock:
    return TimeBlock(pd.Timestamp(start), pd.Timestamp(end))


def _frame(predictions: list[str], durations: list[int] | None = None) -> pd.DataFrame:
    times = pd.to_datetime(predictions, utc=True)
    lengths = durations if durations is not None else [16] * len(times)
    return pd.DataFrame(
        {
            "sample_id": [f"row_{index}" for index in range(len(times))],
            "prediction_time_utc": times,
            "entry_time_utc": times + pd.Timedelta(minutes=1),
            "label_end_time_utc": times + pd.to_timedelta(lengths, unit="min"),
            "feature_available_at_utc": times,
            "feature_window_start_utc": times - pd.Timedelta(minutes=60),
        }
    )


def test_frozen_schedule_reserves_calibration_and_expands_training() -> None:
    folds = make_walk_forward_folds()
    assert tuple(fold.name for fold in folds) == ("test_2022", "test_2023", "test_2024")
    for year, fold in zip((2022, 2023, 2024), folds, strict=True):
        assert fold.train.start == pd.Timestamp("2020-01-01T00:00:00Z")
        assert fold.train.end == pd.Timestamp(f"{year - 1}-10-01T00:00:00Z")
        assert fold.train.end == fold.calibration.start
        assert fold.calibration.end == fold.test.start
        assert fold.test.end == pd.Timestamp(f"{year + 1}-01-01T00:00:00Z")
        first, second = fold.inner_folds
        assert first.validation.start == pd.Timestamp(f"{year - 1}-04-01T00:00:00Z")
        assert first.validation.end == second.validation.start
        assert second.validation.end == fold.calibration.start
        for inner in fold.inner_folds:
            assert inner.train.start == fold.train.start
            assert inner.train.end == inner.validation.start
            assert inner.validation.end <= fold.train.end
        assert fold.as_record()["calibration_usage"] == "reserved_not_used_for_fit_or_selection"
    assert folds[0].test.end == folds[1].test.start
    with pytest.raises(FrozenInstanceError):
        folds[0].name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("years", [(), (2024, 2023), (2023, 2023), (2025,), (2020,), (True,)])
def test_invalid_schedules_are_rejected(years: tuple[int, ...]) -> None:
    with pytest.raises(WalkForwardError):
        make_walk_forward_folds(years)


def test_overlapping_blocks_and_unordered_inner_folds_are_rejected() -> None:
    with pytest.raises(WalkForwardError, match="overlap"):
        InnerFold(
            "invalid",
            _block("2020-01-01T00:00:00Z", "2021-06-01T00:00:00Z"),
            _block("2021-04-01T00:00:00Z", "2021-07-01T00:00:00Z"),
        )
    valid = make_walk_forward_folds()[0]
    with pytest.raises(WalkForwardError, match="ordered"):
        WalkForwardFold(
            valid.name, valid.train, valid.calibration, valid.test, valid.inner_folds[::-1]
        )


def test_train_gap_is_common_to_short_and_long_horizons() -> None:
    frame = _frame(
        [
            "2023-03-31T23:28:00Z",
            "2023-03-31T23:29:00Z",
            "2023-03-31T23:30:00Z",
            "2023-03-31T23:31:00Z",
        ],
        [4, 4, 16, 31],
    )
    original = frame.copy(deep=True)
    block = _block("2023-01-01T00:00:00Z", "2023-04-01T00:00:00Z")
    result = select_block(frame, block, gap_minutes=required_gap_minutes((3, 15, 30)))
    assert result["sample_id"].tolist() == ["row_0"]
    pd.testing.assert_frame_equal(frame, original)
    result.loc[0, "sample_id"] = "modified_copy"
    assert frame.loc[0, "sample_id"] == "row_0"


def test_test_selection_has_no_train_gap_but_requires_strictly_mature_labels() -> None:
    frame = _frame(
        [
            "2023-01-01T00:00:00Z",
            "2023-03-31T23:44:00Z",
            "2023-03-31T23:45:00Z",
            "2023-03-31T23:55:00Z",
            "2023-03-31T23:56:00Z",
            "2023-04-01T00:00:00Z",
        ],
        [16, 16, 16, 4, 4, 16],
    )
    result = select_block(
        frame,
        _block("2023-01-01T00:00:00Z", "2023-04-01T00:00:00Z"),
        purge=False,
    )
    assert result["sample_id"].tolist() == ["row_0", "row_3"]


def test_label_crossing_is_removed_even_when_prediction_precedes_train_gap() -> None:
    frame = _frame(["2023-03-31T22:00:00Z", "2023-03-31T22:03:00Z"], [121, 16])
    result = select_block(frame, _block("2023-01-01T00:00:00Z", "2023-04-01T00:00:00Z"))
    assert result["sample_id"].tolist() == ["row_1"]


def test_holdout_row_outside_selected_block_is_still_rejected() -> None:
    frame = _frame(["2023-01-02T00:00:00Z", "2025-01-02T00:00:00Z"])
    with pytest.raises(WalkForwardError, match="reserved holdout"):
        select_block(frame, make_walk_forward_folds()[0].train)


@pytest.mark.parametrize(
    "column",
    ["label_end_time_utc", "feature_available_at_utc", "feature_window_start_utc"],
)
def test_holdout_guard_covers_labels_and_features_not_only_predictions(column: str) -> None:
    frame = _frame(["2024-12-31T12:00:00Z"])
    frame[column] = pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True)
    with pytest.raises(WalkForwardError, match="reserved holdout"):
        validate_development_frame(frame)


def test_guard_cannot_be_disabled_or_weakened_but_can_be_stricter() -> None:
    frame = _frame(["2024-01-02T00:00:00Z"])
    with pytest.raises(WalkForwardError, match="cannot expose"):
        validate_development_frame(frame, guard_start="2026-01-01T00:00:00Z")
    with pytest.raises(WalkForwardError, match="reserved holdout"):
        validate_development_frame(frame, guard_start="2024-01-01T00:00:00Z")
    with pytest.raises(WalkForwardError, match="block would expose"):
        select_block(frame, _block("2024-01-01T00:00:00Z", "2026-01-01T00:00:00Z"))


def test_ingestion_timestamp_is_not_confused_with_observation_time() -> None:
    frame = _frame(["2024-01-02T00:00:00Z"])
    frame["ingested_at_utc"] = pd.to_datetime(["2026-09-05T00:00:00Z"], utc=True)
    validate_development_frame(frame)


@pytest.mark.parametrize("timezone", [None, "Europe/Brussels"])
def test_non_utc_frame_is_rejected(timezone: str | None) -> None:
    frame = _frame(["2023-01-02T00:00:00Z"])
    frame["prediction_time_utc"] = frame["prediction_time_utc"].dt.tz_convert(timezone)
    with pytest.raises(WalkForwardError, match="UTC"):
        validate_development_frame(frame)


@pytest.mark.parametrize(
    "start,end",
    [
        ("2023-01-01", "2024-01-01"),
        ("2023-01-01T00:00:00+01:00", "2024-01-01T00:00:00+01:00"),
        ("2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    ],
)
def test_time_block_requires_utc_and_positive_duration(start: str, end: str) -> None:
    with pytest.raises(WalkForwardError):
        _block(start, end)


def test_missing_timestamps_and_future_features_are_rejected() -> None:
    frame = _frame(["2023-01-02T00:00:00Z"])
    with pytest.raises(WalkForwardError, match="missing evaluation"):
        validate_development_frame(frame.drop(columns="label_end_time_utc"))
    frame.loc[0, "feature_available_at_utc"] = pd.NaT
    with pytest.raises(WalkForwardError, match="missing timestamps"):
        validate_development_frame(frame)
    frame.loc[0, "feature_available_at_utc"] = pd.Timestamp("2023-01-02T00:01:00Z")
    with pytest.raises(WalkForwardError, match="must not follow"):
        validate_development_frame(frame)


def test_duplicate_columns_are_rejected_explicitly() -> None:
    frame = _frame(["2023-01-02T00:00:00Z"])
    malformed = pd.concat([frame, frame[["prediction_time_utc"]]], axis=1)
    with pytest.raises(WalkForwardError, match="duplicate columns"):
        validate_development_frame(malformed)


def test_label_and_entry_order_is_strict() -> None:
    frame = _frame(["2023-01-02T00:00:00Z"])
    frame["entry_time_utc"] = frame["prediction_time_utc"]
    with pytest.raises(WalkForwardError, match="entry must follow"):
        validate_development_frame(frame)
    frame = _frame(["2023-01-02T00:00:00Z"])
    frame["label_end_time_utc"] = frame["prediction_time_utc"]
    with pytest.raises(WalkForwardError, match="labels must end"):
        validate_development_frame(frame)


def test_empty_utc_frame_is_valid_and_returns_empty_copy() -> None:
    frame = _frame([])
    result = select_block(frame, make_walk_forward_folds()[0].train)
    assert result.empty
    assert result is not frame


def test_required_gap_includes_latency_and_rejects_invalid_arguments() -> None:
    assert required_gap_minutes((3, 6, 9, 12, 15, 30)) == 31
    assert required_gap_minutes((3, 60, 180), latency_minutes=2) == 182
    for horizons in ((), (0,), (-3,), (True,)):
        with pytest.raises(WalkForwardError, match="positive integers"):
            required_gap_minutes(horizons)
    with pytest.raises(WalkForwardError, match="nonnegative integer"):
        required_gap_minutes((3,), latency_minutes=-1)
    with pytest.raises(WalkForwardError, match="nonnegative integer"):
        select_block(_frame([]), make_walk_forward_folds()[0].train, gap_minutes=-1)


def test_fold_summary_has_explicit_direction_and_equal_fold_weights() -> None:
    summary = summarize_fold_metrics(
        [
            {"return_bps": 12.0, "loss": 2.0},
            {"return_bps": -6.0, "loss": 1.0},
            {"return_bps": 3.0, "loss": 3.0},
        ],
        higher_is_better={"return_bps": True, "loss": False},
    )
    assert summary["return_bps"] == {
        "fold_count": 3,
        "observed_fold_count": 3,
        "missing_fold_count": 0,
        "higher_is_better": True,
        "mean": 3.0,
        "median": 3.0,
        "worst": -6.0,
    }
    assert summary["loss"]["worst"] == 3.0
    assert summary["loss"]["mean"] == 2.0


def test_missing_fold_metrics_do_not_hide_unknown_worst_case() -> None:
    summary = summarize_fold_metrics(
        [{"precision": 0.7}, {"precision": None}], higher_is_better={"precision": True}
    )["precision"]
    assert summary["mean"] == 0.7
    assert summary["worst"] is None
    assert summary["missing_fold_count"] == 1
    empty = summarize_fold_metrics([{"value": None}], higher_is_better={"value": False})
    assert empty["value"]["mean"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_invalid_metric_values_are_rejected(value: float) -> None:
    with pytest.raises(WalkForwardError, match="finite numeric"):
        summarize_fold_metrics([{"loss": value}], higher_is_better={"loss": False})


def test_metric_summary_requires_declared_fields_and_nonempty_folds() -> None:
    with pytest.raises(WalkForwardError, match="missing metric"):
        summarize_fold_metrics([{"loss": 1.0}, {}], higher_is_better={"loss": False})
    with pytest.raises(WalkForwardError, match="must not be empty"):
        summarize_fold_metrics([], higher_is_better={"loss": False})
