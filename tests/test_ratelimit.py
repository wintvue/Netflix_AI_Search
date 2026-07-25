"""Tests for the AI overview rate limiter."""

import time

from api.ratelimit import SlidingWindowRateLimiter, client_key


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

    def test_is_thread_safe(self):
        """The limiter is shared across the request threadpool."""
        import threading

        limiter = SlidingWindowRateLimiter(limit=1000, window_seconds=60)
        errors = []

        def hammer():
            try:
                for _ in range(200):
                    limiter.allow("shared")
            except Exception as e:  # pragma: no cover - only on a locking bug
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


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
