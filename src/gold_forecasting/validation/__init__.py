"""Validation contracts for curated market data."""

from gold_forecasting.validation.candles import (
    CANDLE_KEY_COLUMNS,
    REQUIRED_CANDLE_COLUMNS,
    CandleGap,
    CandleValidationError,
    CandleValidationResult,
    GapReport,
    detect_gaps,
    validate_candles,
)

__all__ = [
    "CANDLE_KEY_COLUMNS",
    "REQUIRED_CANDLE_COLUMNS",
    "CandleGap",
    "CandleValidationError",
    "CandleValidationResult",
    "GapReport",
    "detect_gaps",
    "validate_candles",
]
