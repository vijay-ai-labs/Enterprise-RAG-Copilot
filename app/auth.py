"""
API key authentication + sliding-window rate limiter.

Auth:  X-API-Key header (only enforced when API_KEY env var is set).
Rate:  Per-IP sliding window; disabled when RATE_LIMIT_PER_MINUTE=0.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(x_api_key: Optional[str] = Security(_api_key_header)) -> None:
    """FastAPI dependency — raises 401 when API_KEY is set and header is wrong/missing."""
    if not settings.api_key:
        return  # Auth disabled; dev/open mode
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


class SlidingWindowRateLimiter:
    """Simple in-memory per-key sliding window rate limiter."""

    def __init__(self, max_requests: int, window_seconds: int = 60) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._buckets: dict = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        if self._max <= 0:
            return True
        now = time.time()
        cutoff = now - self._window
        with self._lock:
            self._buckets[key] = [t for t in self._buckets[key] if t > cutoff]
            if len(self._buckets[key]) >= self._max:
                return False
            self._buckets[key].append(now)
            return True

    def cleanup(self) -> None:
        now = time.time()
        cutoff = now - self._window
        with self._lock:
            stale = [k for k, ts in self._buckets.items() if not any(t > cutoff for t in ts)]
            for k in stale:
                del self._buckets[k]


rate_limiter = SlidingWindowRateLimiter(
    max_requests=settings.rate_limit_per_minute,
    window_seconds=60,
)
