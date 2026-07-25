#!/usr/bin/env python3
"""In-process rate limiting for the paid AI overview path.

Deliberately dependency-free and per-process: the service runs as a single
instance today. If it is ever scaled out, move this to a shared store — each
worker would otherwise permit the full quota on its own.
"""

import threading
import time
from collections import defaultdict, deque

from fastapi import Request


class SlidingWindowRateLimiter:
    """Allow at most `limit` events per `window_seconds` per key."""

    def __init__(self, limit: int, window_seconds: float = 60.0):
        self._limit = limit
        self._window = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._limit > 0

    def allow(self, key: str) -> bool:
        """Record an event for `key`, returning False if it exceeds the quota."""
        if not self.enabled:
            return True

        now = time.monotonic()
        cutoff = now - self._window

        with self._lock:
            events = self._events[key]
            while events and events[0] < cutoff:
                events.popleft()

            if len(events) >= self._limit:
                return False

            events.append(now)

            # Keep the map from growing without bound as clients come and go.
            if len(self._events) > 10_000:
                for stale_key in [k for k, v in self._events.items() if not v]:
                    del self._events[stale_key]

            return True

    def retry_after(self, key: str) -> int:
        """Seconds until the oldest event in the window expires."""
        with self._lock:
            events = self._events.get(key)
            if not events:
                return 0
            return max(1, int(self._window - (time.monotonic() - events[0])) + 1)


def client_key(request: Request) -> str:
    """Identify the caller for rate limiting purposes."""
    # Render terminates TLS upstream, so the direct peer is the proxy.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
