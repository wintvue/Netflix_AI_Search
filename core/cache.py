#!/usr/bin/env python3
"""A small TTL cache.

`functools.lru_cache` covers the embedding cache, but search responses need
entries to expire so the API reflects reseeded data within a bounded delay.

No lock. Every mutation below is a single OrderedDict operation, which the GIL
makes atomic, and the multi-step paths (expire-then-delete, insert-then-evict)
are written to tolerate a concurrent thread having already done the step. The
worst a race costs is a redundant miss or a slightly early eviction.
"""

import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    """LRU cache whose entries expire after a fixed time-to-live."""

    def __init__(self, maxsize: int, ttl_seconds: float):
        self._maxsize = max(0, maxsize)
        self._ttl = ttl_seconds
        self._entries: OrderedDict[Any, tuple[float, Any]] = OrderedDict()
        # Plain ints: `+= 1` can lose an update under concurrency, which is
        # acceptable for hit-rate reporting and not worth serializing lookups.
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self._maxsize > 0 and self._ttl > 0

    def get(self, key: Any) -> Any | None:
        """Return the cached value, or None if absent or expired."""
        if not self.enabled:
            return None

        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None

        expires_at, value = entry
        if expires_at < time.monotonic():
            # pop, not del: another thread may have expired or evicted it first.
            self._entries.pop(key, None)
            self.misses += 1
            return None

        self._touch(key)
        self.hits += 1
        return value

    def set(self, key: Any, value: Any) -> None:
        if not self.enabled:
            return

        self._entries[key] = (time.monotonic() + self._ttl, value)
        self._touch(key)
        while len(self._entries) > self._maxsize:
            try:
                self._entries.popitem(last=False)
            except KeyError:  # another thread drained it first
                break

    def _touch(self, key: Any) -> None:
        """Mark `key` most-recently-used, tolerating a concurrent eviction."""
        try:
            self._entries.move_to_end(key)
        except KeyError:
            pass

    def clear(self) -> None:
        self._entries.clear()

    def info(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self._entries),
            "maxsize": self._maxsize,
            "ttl_seconds": self._ttl,
        }
