"""Retry + rate-limit handling shared across all model adapters.

Exponential backoff with jitter. Respects provider-specific rate-limit headers
where available. Logs every retry.
"""

from __future__ import annotations
import logging
import random
import time
from functools import wraps
from typing import Callable, TypeVar

T = TypeVar("T")
log = logging.getLogger("oa_bench.retry")


# Provider-specific transient error fingerprints
_TRANSIENT_PATTERNS = (
    "rate limit", "rate_limit", "ratelimit", "429",
    "overloaded", "overload", "529",
    "service_unavailable", "503", "service unavailable",
    "internal_server", "500", "internal server",
    "timeout", "timed out", "connection reset", "connection error",
    "bad_gateway", "502",
    "gateway_timeout", "504",
    "anthropic_api_error_overloaded",
)


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    for p in _TRANSIENT_PATTERNS:
        if p in name or p in msg:
            return True
    # Status code attributes
    code = getattr(exc, "status_code", None) or getattr(exc, "status", None) or getattr(exc, "code", None)
    if isinstance(code, int) and code in (408, 425, 429, 500, 502, 503, 504, 529):
        return True
    return False


def with_retry(
    *,
    max_retries: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.25,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that retries the wrapped call on transient errors with exponential backoff."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = base_delay
            while True:
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt > max_retries or not _is_transient(e):
                        raise
                    # If the exception exposes a Retry-After hint, honor it
                    retry_after = getattr(e, "retry_after", None)
                    if isinstance(retry_after, (int, float)) and retry_after > 0:
                        wait = min(float(retry_after), max_delay)
                    else:
                        # Exponential backoff with jitter
                        wait = min(delay, max_delay)
                        wait += random.uniform(0, wait * jitter)
                        delay *= backoff_factor
                    log.warning(
                        "retry %d/%d after %.1fs  %s: %s",
                        attempt, max_retries, wait, type(e).__name__, e,
                    )
                    time.sleep(wait)
        return wrapper

    return decorator
