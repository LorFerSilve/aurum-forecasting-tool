"""Serializable contract for one three-class MVP prediction."""

from __future__ import annotations

from datetime import UTC, datetime
from math import isclose
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gold_forecasting.classification import CLASS_ORDER, ClassLabel


class PredictionRecord(BaseModel):
    """One immutable, audit-ready prediction in the roadmap's public schema.

    Probabilities always use the project's fixed ``down/neutral/up`` order.
    A tie is resolved by that same order, matching ``numpy.argmax`` in the
    model layer.  Timezone-aware offsets are accepted at the boundary and
    normalized to UTC so serialized records cannot silently mix timezones.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )

    prediction_time_utc: datetime
    instrument: str = Field(min_length=1)
    horizon_minutes: int = Field(gt=0)
    predicted_class: ClassLabel
    p_down: float = Field(ge=0.0, le=1.0)
    p_neutral: float = Field(ge=0.0, le=1.0)
    p_up: float = Field(ge=0.0, le=1.0)
    model_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)
    calibration_status: Literal["preliminary"] = "preliminary"

    @field_validator("prediction_time_utc")
    @classmethod
    def normalize_prediction_time_to_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("prediction_time_utc must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def probabilities_match_class_contract(self) -> Self:
        probabilities = (self.p_down, self.p_neutral, self.p_up)
        if not isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("p_down, p_neutral and p_up must sum to one")
        winning_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        expected_class = CLASS_ORDER[winning_index]
        if self.predicted_class != expected_class:
            raise ValueError(
                "predicted_class must equal the probability argmax "
                f"({expected_class!r}) in {CLASS_ORDER} order"
            )
        return self


__all__ = ["PredictionRecord"]
