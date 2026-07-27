"""Tests for the TTL cache backing search responses."""

import threading
import time

from core.cache import TTLCache


class TestTTLCache:
    def test_get_returns_what_was_set(self):
        cache = TTLCache(maxsize=4, ttl_seconds=60)
        cache.set("k", {"v": 1})
        assert cache.get("k") == {"v": 1}

    def test_miss_returns_none(self):
        assert TTLCache(maxsize=4, ttl_seconds=60).get("absent") is None

    def test_entries_expire(self):
        """Entries must expire so reseeded data appears within a bounded delay."""
        cache = TTLCache(maxsize=4, ttl_seconds=0.05)
        cache.set("k", "v")
        assert cache.get("k") == "v"
        time.sleep(0.08)
        assert cache.get("k") is None

    def test_evicts_least_recently_used(self):
        cache = TTLCache(maxsize=2, ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # refresh a, making b the LRU
        cache.set("c", 3)

        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3

    def test_disabled_when_size_is_zero(self):
        cache = TTLCache(maxsize=0, ttl_seconds=60)
        cache.set("k", "v")
        assert not cache.enabled
        assert cache.get("k") is None

    def test_disabled_when_ttl_is_zero(self):
        cache = TTLCache(maxsize=10, ttl_seconds=0)
        cache.set("k", "v")
        assert not cache.enabled
        assert cache.get("k") is None

    def test_clear_empties_the_cache(self):
        cache = TTLCache(maxsize=4, ttl_seconds=60)
        cache.set("k", "v")
        cache.clear()
        assert cache.get("k") is None

    def test_info_tracks_hits_and_misses(self):
        cache = TTLCache(maxsize=4, ttl_seconds=60)
        cache.set("k", "v")
        cache.get("k")
        cache.get("nope")

        info = cache.info()
        assert info["hits"] == 1
        assert info["misses"] == 1
        assert info["size"] == 1

    def test_is_thread_safe(self):
        """The cache is shared across the request threadpool."""
        cache = TTLCache(maxsize=100, ttl_seconds=60)
        errors = []

        def hammer(n):
            try:
                for i in range(200):
                    cache.set(f"{n}-{i}", i)
                    cache.get(f"{n}-{i}")
            except Exception as e:  # pragma: no cover - only on a locking bug
                errors.append(e)

        threads = [threading.Thread(target=hammer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
