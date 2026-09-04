from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from gold_forecasting.inference import PredictionRecord


def _valid_prediction(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "prediction_time_utc": datetime(2024, 1, 2, 12, 0, tzinfo=UTC),
        "instrument": "XAU_USD",
        "horizon_minutes": 15,
        "predicted_class": "up",
        "p_down": 0.18,
        "p_neutral": 0.27,
        "p_up": 0.55,
        "model_version": "v0.1.0",
        "data_version": "sha256:abc",
        "calibration_status": "preliminary",
    }
    values.update(overrides)
    return values


def test_prediction_record_has_exact_roadmap_fields_and_is_immutable() -> None:
    record = PredictionRecord.model_validate(_valid_prediction())

    assert tuple(PredictionRecord.model_fields) == (
        "prediction_time_utc",
        "instrument",
        "horizon_minutes",
        "predicted_class",
        "p_down",
        "p_neutral",
        "p_up",
        "model_version",
        "data_version",
        "calibration_status",
    )
    assert record.calibration_status == "preliminary"
    assert record.model_dump(mode="json")["prediction_time_utc"] == "2024-01-02T12:00:00Z"
    with pytest.raises(ValidationError, match="frozen"):
        record.p_up = 0.9


def test_prediction_time_is_normalized_to_utc() -> None:
    offset = timezone(timedelta(hours=2))

    record = PredictionRecord.model_validate(
        _valid_prediction(prediction_time_utc=datetime(2024, 1, 2, 14, 0, tzinfo=offset))
    )

    assert record.prediction_time_utc == datetime(2024, 1, 2, 12, 0, tzinfo=UTC)
    assert record.prediction_time_utc.tzinfo is UTC


def test_probability_ties_follow_fixed_down_neutral_up_order() -> None:
    record = PredictionRecord.model_validate(
        _valid_prediction(predicted_class="down", p_down=0.4, p_neutral=0.4, p_up=0.2)
    )

    assert record.predicted_class == "down"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"prediction_time_utc": datetime(2024, 1, 2, 12, 0)}, "timezone-aware"),
        ({"p_down": 0.2}, "sum to one"),
        ({"p_up": 1.1, "p_neutral": -0.1}, "greater than or equal to 0"),
        ({"p_up": float("nan")}, "finite number"),
        ({"predicted_class": "neutral"}, "probability argmax"),
        ({"calibration_status": "calibrated"}, "preliminary"),
        ({"horizon_minutes": 0}, "greater than 0"),
        ({"instrument": ""}, "at least 1 character"),
    ],
)
def test_invalid_prediction_records_are_rejected(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        PredictionRecord.model_validate(_valid_prediction(**overrides))


def test_prediction_record_forbids_fields_outside_the_public_contract() -> None:
    values = _valid_prediction()
    values["expected_return_bps"] = 4.2

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PredictionRecord.model_validate(values)
