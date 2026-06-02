"""Shared transient-failure retry layer (Phase 4).

A single place all network callers (LLM in `llm.py`, web search in `retrieval.py`) route
through, so the retry policy lives in one spot rather than being copy-pasted. This is
application-level resilience *on top of* whatever the SDKs do internally: the goal is
that a transient blip never throws away the work the agent already completed.

Policy:
  * Retry ONLY transient errors — HTTP 429/408/5xx, timeouts, and connection errors.
  * Fail fast (no retry) on permanent errors — 400 (bad request), 401/403 (auth),
    404, 422 — because retrying them just wastes time and quota.
  * Exponential backoff with jitter, capped at `RETRY_MAX_ATTEMPTS` total attempts.
  * If the error carries a server hint (a `Retry-After` header or a `retry_after`
    attribute — Groq and Gemini both provide one), honour that delay instead.

Errors are classified by duck-typing (status code + exception type name) rather than by
importing each SDK's exception classes, so the layer stays provider-agnostic.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

from . import config

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Status codes worth retrying (server-side / rate limiting) vs. never retrying (the
# request itself is wrong and will keep being wrong).
_TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}
_PERMANENT_STATUS = {400, 401, 403, 404, 405, 406, 409, 422}

# Substrings in an exception's class name that indicate a transient network problem,
# for errors that don't expose a status code (e.g. raw timeouts / connection resets).
_TRANSIENT_NAME_HINTS = (
    "timeout",
    "connection",
    "serviceunavailable",
    "apiconnection",
    "remotedisconnected",
)


def _status_code(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from an exception."""
    code = getattr(exc, "status_code", None)
    if code is None:
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def is_transient(exc: BaseException) -> bool:
    """Classify an exception as a transient (retryable) failure."""
    code = _status_code(exc)
    if code in _PERMANENT_STATUS:
        return False
    if code in _TRANSIENT_STATUS:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__.lower()
    return any(hint in name for hint in _TRANSIENT_NAME_HINTS)


def _server_retry_hint(exc: BaseException) -> float | None:
    """Return a server-suggested delay in seconds, if the error carries one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            raw = headers.get("retry-after") or headers.get("Retry-After")
        except AttributeError:
            raw = None
        if raw:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
    hint = getattr(exc, "retry_after", None)
    if isinstance(hint, (int, float)):
        return float(hint)
    return None


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff (1s, 2s, 4s, ...) capped, with up to 25% added jitter."""
    base = min(config.RETRY_MAX_DELAY, config.RETRY_BASE_DELAY * (2 ** (attempt - 1)))
    return base + random.uniform(0, base * 0.25)


def call_with_retries(
    func: Callable[[], T],
    *,
    description: str,
    max_attempts: int | None = None,
) -> T:
    """Call `func()` with retries on transient failures.

    Permanent errors propagate immediately (fail fast). Transient errors are retried
    with backoff up to `max_attempts`; the final failure is re-raised so the caller can
    decide how to degrade (e.g. skip a sub-question).
    """
    attempts = max_attempts or config.RETRY_MAX_ATTEMPTS
    attempt = 0
    while True:
        attempt += 1
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - classified below, re-raised if not transient
            if not is_transient(exc):
                logger.error("%s: permanent error, not retrying: %r", description, exc)
                raise
            if attempt >= attempts:
                logger.error(
                    "%s: still failing after %d attempts, giving up: %r",
                    description, attempt, exc,
                )
                raise
            delay = _server_retry_hint(exc)
            source = "server hint" if delay is not None else "backoff"
            if delay is None:
                delay = _backoff_delay(attempt)
            logger.warning(
                "%s: transient error on attempt %d/%d (%r); retrying in %.1fs (%s)",
                description, attempt, attempts, exc, delay, source,
            )
            time.sleep(delay)
