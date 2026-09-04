"""Process-level runtime helpers for reproducible research runs."""

from __future__ import annotations

import logging
import os
import random
from datetime import UTC, datetime
from typing import TextIO

import numpy as np

_MAX_NUMPY_SEED = 2**32 - 1
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def utc_now() -> datetime:
    """Return an aware timestamp in UTC.

    Keeping timestamp creation behind one small function also gives callers a
    single, unambiguous clock contract: persisted timestamps are always UTC.
    """

    return datetime.now(UTC)


def format_utc(timestamp: datetime) -> str:
    """Serialize an aware timestamp as an RFC 3339 UTC value."""

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return timestamp.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class UTCFormatter(logging.Formatter):
    """Logging formatter whose ``asctime`` is an RFC 3339 UTC timestamp."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC)
        if datefmt is not None:
            return timestamp.strftime(datefmt)
        return format_utc(timestamp)


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure the process root logger with one UTC-aware stream handler.

    Repeated calls replace the previous configuration instead of accumulating
    duplicate handlers.  The returned root logger is convenient for tests and
    small command-line entrypoints; application modules should still obtain
    their own logger with :func:`logging.getLogger`.
    """

    handler = logging.StreamHandler(stream)
    handler.setFormatter(UTCFormatter(_LOG_FORMAT))

    root_logger = logging.getLogger()
    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)
        existing_handler.close()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    return root_logger


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy's process-global pseudo-random generators.

    ``PYTHONHASHSEED`` is also exported for child Python processes.  Python's
    hash randomization for the already-running interpreter is chosen at process
    startup and therefore cannot be changed retroactively.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= _MAX_NUMPY_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_NUMPY_SEED}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
