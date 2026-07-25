#!/usr/bin/env python3
"""A small thread-safe TTL cache.

`functools.lru_cache` covers the embedding cache, but search responses need
entries to expire so the API reflects reseeded data within a bounded delay.
"""

import threading
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    """LRU cache whose entries expire after a fixed time-to-live."""

    def __init__(self, maxsize: int, ttl_seconds: float):
        self._maxsize = max(0, maxsize)
        self._ttl = ttl_seconds
        self._entries: OrderedDict[Any, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self._maxsize > 0 and self._ttl > 0

    def get(self, key: Any) -> Any | None:
        """Return the cached value, or None if absent or expired."""
        if not self.enabled:
            return None

        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None

            expires_at, value = entry
            if expires_at < time.monotonic():
                del self._entries[key]
                self.misses += 1
                return None

            self._entries.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: Any, value: Any) -> None:
        if not self.enabled:
            return

        with self._lock:
            self._entries[key] = (time.monotonic() + self._ttl, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def info(self) -> dict:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "size": len(self._entries),
                "maxsize": self._maxsize,
                "ttl_seconds": self._ttl,
            }
