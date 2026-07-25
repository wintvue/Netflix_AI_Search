"""Tests for the TTL cache and the AI overview rate limiter."""

import time

import pytest

from api.ratelimit import SlidingWindowRateLimiter, client_key
from core.cache import TTLCache


class TestTTLCache:
    def test_get_returns_what_was_set(self):
        cache = TTLCache(maxsize=4, ttl_seconds=60)
        cache.set("k", {"v": 1})
        assert cache.get("k") == {"v": 1}

    def test_miss_returns_none(self):
        assert TTLCache(maxsize=4, ttl_seconds=60).get("absent") is None

    def test_entries_expire(self):
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


class TestSlidingWindowRateLimiter:
    def test_allows_up_to_the_limit(self):
        limiter = SlidingWindowRateLimiter(limit=3, window_seconds=60)
        assert [limiter.allow("ip") for _ in range(3)] == [True, True, True]

    def test_blocks_beyond_the_limit(self):
        limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60)
        limiter.allow("ip")
        limiter.allow("ip")
        assert limiter.allow("ip") is False

    def test_keys_are_independent(self):
        limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)
        assert limiter.allow("a") is True
        assert limiter.allow("b") is True
        assert limiter.allow("a") is False

    def test_window_slides(self):
        limiter = SlidingWindowRateLimiter(limit=1, window_seconds=0.05)
        assert limiter.allow("ip") is True
        assert limiter.allow("ip") is False
        time.sleep(0.08)
        assert limiter.allow("ip") is True

    def test_zero_limit_disables_limiting(self):
        limiter = SlidingWindowRateLimiter(limit=0)
        assert not limiter.enabled
        assert all(limiter.allow("ip") for _ in range(100))

    def test_retry_after_is_positive_once_blocked(self):
        limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)
        limiter.allow("ip")
        assert limiter.retry_after("ip") > 0

    def test_retry_after_is_zero_for_an_unknown_key(self):
        assert SlidingWindowRateLimiter(limit=1).retry_after("unseen") == 0


class FakeRequest:
    def __init__(self, headers=None, host=None):
        self.headers = headers or {}
        self.client = type("C", (), {"host": host})() if host else None


class TestClientKey:
    def test_prefers_the_forwarded_header(self):
        request = FakeRequest(
            headers={"x-forwarded-for": "203.0.113.7, 10.0.0.1"}, host="10.0.0.1"
        )
        assert client_key(request) == "203.0.113.7"

    def test_falls_back_to_the_peer_address(self):
        assert client_key(FakeRequest(host="198.51.100.4")) == "198.51.100.4"

    def test_unknown_when_there_is_no_client(self):
        assert client_key(FakeRequest()) == "unknown"


@pytest.mark.parametrize("workers", [4])
def test_ttl_cache_is_thread_safe(workers):
    """The cache is shared across the request threadpool."""
    import threading

    cache = TTLCache(maxsize=100, ttl_seconds=60)
    errors = []

    def hammer(n):
        try:
            for i in range(200):
                cache.set(f"{n}-{i}", i)
                cache.get(f"{n}-{i}")
        except Exception as e:  # pragma: no cover - only on a locking bug
            errors.append(e)

    threads = [threading.Thread(target=hammer, args=(n,)) for n in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
