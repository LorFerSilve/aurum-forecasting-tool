"""Bounded, polite HTTP request primitives for market-data adapters.

The requester is intentionally provider-neutral.  It rate-limits every
attempt, retries only transient failures, honours a numeric ``Retry-After``
within the configured ceiling, and never retries forever.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

import httpx

_DEFAULT_RETRYABLE_STATUSES: Final = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class HttpRequestPolicy:
    """Network safety limits shared by public and future broker adapters."""

    max_attempts: int = 3
    min_interval_seconds: float = 0.25
    initial_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 4.0
    retryable_statuses: frozenset[int] = _DEFAULT_RETRYABLE_STATUSES

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise ValueError("max_attempts must be an integer")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        for name in (
            "min_interval_seconds",
            "initial_backoff_seconds",
            "max_backoff_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.initial_backoff_seconds > self.max_backoff_seconds:
            raise ValueError("initial_backoff_seconds must not exceed max_backoff_seconds")
        if any(status < 400 or status > 599 for status in self.retryable_statuses):
            raise ValueError("retryable_statuses must contain HTTP error status codes")


class BoundedHttpRequester:
    """Apply one request policy while preserving the caller-owned client."""

    def __init__(
        self,
        policy: HttpRequestPolicy | None = None,
        *,
        sleeper: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.policy = policy or HttpRequestPolicy()
        self._sleep = sleeper or time.sleep
        self._monotonic = monotonic or time.monotonic
        self._last_attempt_started: float | None = None

    def request(
        self,
        client: httpx.Client,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        data: Mapping[str, str] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> httpx.Response:
        """Issue a bounded request and return the final response.

        Transport errors and the configured transient statuses are retried.
        A final HTTP error remains an ``httpx.HTTPStatusError`` so an adapter
        can translate it into its own domain error.
        """

        last_transport_error: httpx.TransportError | None = None
        for attempt in range(1, self.policy.max_attempts + 1):
            self._pace()
            try:
                if timeout is None:
                    # Omission preserves the caller's client timeout; passing
                    # None to httpx would silently disable its time limit.
                    response = client.request(method, url, headers=headers, data=data)
                else:
                    response = client.request(
                        method, url, headers=headers, data=data, timeout=timeout
                    )
            except httpx.TransportError as exc:
                last_transport_error = exc
                if attempt == self.policy.max_attempts:
                    raise
                self._sleep(self._backoff_seconds(attempt, retry_after=None))
                continue

            if response.status_code not in self.policy.retryable_statuses:
                response.raise_for_status()
                return response
            if attempt == self.policy.max_attempts:
                response.raise_for_status()
            retry_after = _numeric_retry_after(response.headers.get("Retry-After"))
            response.close()
            self._sleep(self._backoff_seconds(attempt, retry_after=retry_after))

        if last_transport_error is not None:  # pragma: no cover - loop is exhaustive
            raise last_transport_error
        raise RuntimeError("bounded HTTP request loop ended without a response")  # pragma: no cover

    def _pace(self) -> None:
        now = self._monotonic()
        if self._last_attempt_started is not None:
            remaining = self.policy.min_interval_seconds - (now - self._last_attempt_started)
            if remaining > 0:
                self._sleep(remaining)
                now = self._monotonic()
        self._last_attempt_started = now

    def _backoff_seconds(self, attempt: int, *, retry_after: float | None) -> float:
        exponential = self.policy.initial_backoff_seconds
        for _ in range(attempt - 1):
            exponential = min(exponential * 2.0, self.policy.max_backoff_seconds)
            if exponential == 0 or exponential >= self.policy.max_backoff_seconds:
                break
        requested = exponential if retry_after is None else max(exponential, retry_after)
        return min(requested, self.policy.max_backoff_seconds)


def _numeric_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value.strip())
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return max(0.0, parsed)


__all__ = ["BoundedHttpRequester", "HttpRequestPolicy"]
