"""Endpoint tests. The search layer is mocked; no database or network."""

import pytest
from fastapi.testclient import TestClient

import api.main as main


def fake_response(query="q", count=2):
    return {
        "query": query,
        "config": {
            "alpha": 0.8,
            "rrf_k": 60,
            "vector_candidates": 150,
            "bm25_candidates": 150,
            "rerank_candidates": 100,
            "reranked": True,
        },
        "timings": {
            "encode_ms": 1.0,
            "retrieval_ms": 2.0,
            "fusion_ms": 0.5,
            "rerank_ms": 3.0,
            "total_ms": 6.5,
        },
        "retrieval": {"vector": 150, "bm25": 40, "fused": 170},
        "count": count,
        "results": [
            {"id": i, "title": f"Movie {i}", "rrf_score": 0.01, "rerank_score": 0.9}
            for i in range(1, count + 1)
        ],
    }


@pytest.fixture
def make_client(monkeypatch):
    """Build a TestClient with the search layer and startup stubbed out."""

    def factory(raise_server_exceptions: bool = True):
        # Keep startup off the network and off the database.
        monkeypatch.setattr(main, "get_huggingface_client", lambda: object())
        monkeypatch.setattr(main, "create_db_pool", lambda: object())
        monkeypatch.setattr(main, "close_pool", lambda: None)
        monkeypatch.setattr(
            main, "hybrid_search", lambda q, k, alpha: fake_response(q, k)
        )
        monkeypatch.setattr(
            main, "search_movies", lambda q, k: fake_response(q, k)["results"]
        )
        monkeypatch.setattr(
            main, "semantic_search", lambda q, k: fake_response(q, k)["results"]
        )
        return TestClient(main.app, raise_server_exceptions=raise_server_exceptions)

    return factory


@pytest.fixture
def client(make_client):
    with make_client() as c:
        yield c


@pytest.fixture
def lenient_client(make_client):
    """Client that returns the 500 response instead of re-raising it.

    Starlette's ServerErrorMiddleware returns the handler's response *and*
    re-raises so the ASGI server can log it; TestClient surfaces that re-raise
    unless told not to.
    """
    with make_client(raise_server_exceptions=False) as c:
        yield c


class TestResponseContract:
    def test_returns_the_documented_shape(self, client):
        body = client.get("/search", params={"q": "sci-fi", "k": 3}).json()

        assert body["query"] == "sci-fi"
        assert body["count"] == 3
        assert len(body["results"]) == 3
        assert body["timings"]["rerank_ms"] == 3.0
        assert body["ai_overview"] is None

    def test_reports_whether_reranking_actually_ran(self, client, monkeypatch):
        """config.reranked keeps the response honest when the reranker is down."""
        degraded = fake_response("q", 2)
        degraded["config"]["reranked"] = False
        for r in degraded["results"]:
            r.pop("rerank_score")
        monkeypatch.setattr(main, "hybrid_search", lambda q, k, alpha: degraded)

        body = client.get("/search", params={"q": "x"}).json()

        assert body["config"]["reranked"] is False
        assert body["results"][0]["rerank_score"] is None

    def test_keyword_search_returns_the_same_result_shape(self, client):
        body = client.get("/search/keyword", params={"query": "matrix", "k": 2}).json()

        assert body["count"] == 2
        assert body["results"][0]["id"] == 1
        assert "title" in body["results"][0]

    def test_openapi_documents_response_schemas(self, client):
        spec = client.get("/openapi.json").json()
        schema = spec["paths"]["/search"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        assert "SearchResponse" in schema["$ref"]


class TestValidation:
    def test_k_upper_bound_is_enforced(self, client):
        assert client.get("/search", params={"q": "x", "k": 10_000}).status_code == 422

    def test_k_lower_bound_is_enforced(self, client):
        assert client.get("/search", params={"q": "x", "k": 0}).status_code == 422

    def test_max_k_is_accepted(self, client):
        r = client.get("/search", params={"q": "x", "k": main.MAX_TOP_K})
        assert r.status_code == 200
        assert r.json()["count"] == main.MAX_TOP_K

    def test_empty_query_rejected(self, client):
        assert client.get("/search", params={"q": ""}).status_code == 422

    def test_alpha_bounds_enforced(self, client):
        assert client.get("/search", params={"q": "x", "alpha": 1.5}).status_code == 422
        assert client.get("/search", params={"q": "x", "alpha": -0.1}).status_code == 422
        assert client.get("/search", params={"q": "x", "alpha": 0.0}).status_code == 200


class TestStreaming:
    def test_stream_without_ai_overview_is_rejected(self, client):
        """Previously this silently returned a plain JSON body."""
        r = client.get("/search", params={"q": "x", "stream": True})
        assert r.status_code == 422
        assert "ai_overview" in r.text

    def test_sse_event_sequence(self, client, monkeypatch):
        monkeypatch.setattr(
            main,
            "generate_ai_overview",
            lambda q, results: {
                "overview": "summary",
                "movie_explanations": [],
                "ai_metadata": {
                    "model": "test",
                    "generation_time_ms": 1.0,
                    "status": "success",
                },
            },
        )

        r = client.get(
            "/search", params={"q": "x", "ai_overview": True, "stream": True}
        )

        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [
            line.split(": ", 1)[1]
            for line in r.text.splitlines()
            if line.startswith("event: ")
        ]
        assert events == ["results", "overview", "done"]
        # Results must precede the overview: that is the point of streaming.
        assert r.text.index("event: results") < r.text.index("event: overview")


class TestRequestIds:
    def test_request_id_header_is_returned(self, client):
        assert client.get("/search", params={"q": "x"}).headers["X-Request-ID"]

    def test_supplied_request_id_is_echoed(self, client):
        r = client.get("/search", params={"q": "x"}, headers={"X-Request-ID": "abc123"})
        assert r.headers["X-Request-ID"] == "abc123"


class TestAIOverviewRateLimit:
    def test_limit_is_enforced(self, client, monkeypatch):
        monkeypatch.setattr(
            main,
            "generate_ai_overview",
            lambda q, results: {
                "overview": "s",
                "movie_explanations": [],
                "ai_metadata": {
                    "model": "t",
                    "generation_time_ms": 1.0,
                    "status": "success",
                },
            },
        )
        limiter = main.SlidingWindowRateLimiter(limit=2, window_seconds=60)
        monkeypatch.setattr(main, "_ai_overview_limiter", limiter)

        params = {"q": "x", "ai_overview": True}
        assert client.get("/search", params=params).status_code == 200
        assert client.get("/search", params=params).status_code == 200

        blocked = client.get("/search", params=params)
        assert blocked.status_code == 429
        assert blocked.json()["error"] == "rate_limited"
        assert int(blocked.headers["Retry-After"]) > 0

    def test_search_without_ai_overview_is_never_limited(self, client, monkeypatch):
        monkeypatch.setattr(
            main, "_ai_overview_limiter", main.SlidingWindowRateLimiter(limit=1)
        )

        for _ in range(5):
            assert client.get("/search", params={"q": "x"}).status_code == 200


class TestErrorMapping:
    def test_embedding_failure_maps_to_503(self, client, monkeypatch):
        def boom(q, k, alpha):
            raise main.EmbeddingError("hf is down")

        monkeypatch.setattr(main, "hybrid_search", boom)

        r = client.get("/search", params={"q": "x"})

        assert r.status_code == 503
        assert r.json()["error"] == "embedding_unavailable"
        # Internals must not leak into the response body.
        assert "hf is down" not in r.text

    def test_configuration_failure_maps_to_503(self, client, monkeypatch):
        def boom(q, k, alpha):
            raise main.ConfigurationError("HF_TOKEN missing")

        monkeypatch.setattr(main, "hybrid_search", boom)

        r = client.get("/search", params={"q": "x"})

        assert r.status_code == 503
        assert r.json()["error"] == "service_misconfigured"
        assert "HF_TOKEN" not in r.text

    def test_pool_timeout_maps_to_504(self, client, monkeypatch):
        def boom(q, k, alpha):
            raise TimeoutError("no pool slot")

        monkeypatch.setattr(main, "hybrid_search", boom)

        r = client.get("/search", params={"q": "x"})

        assert r.status_code == 504
        assert r.json()["error"] == "timeout"

    def test_unexpected_error_maps_to_500_without_a_traceback(
        self, lenient_client, monkeypatch
    ):
        def boom(q, k, alpha):
            raise ValueError("some internal detail")

        monkeypatch.setattr(main, "hybrid_search", boom)

        r = lenient_client.get("/search", params={"q": "x"})

        assert r.status_code == 500
        assert r.json()["error"] == "internal_error"
        assert "some internal detail" not in r.text
        assert "Traceback" not in r.text

    def test_errors_carry_the_request_id(self, client, monkeypatch):
        def boom(q, k, alpha):
            raise main.EmbeddingError("down")

        monkeypatch.setattr(main, "hybrid_search", boom)

        r = client.get("/search", params={"q": "x"}, headers={"X-Request-ID": "rid-1"})

        assert r.json()["request_id"] == "rid-1"


class TestProbes:
    def test_health_is_cheap_and_always_ok(self, client):
        assert client.get("/health").json() == {"status": "healthy"}

    def test_ready_reports_not_ready_when_the_database_is_down(
        self, client, monkeypatch
    ):
        monkeypatch.setattr(main, "ping", lambda: False)

        r = client.get("/ready")

        assert r.status_code == 503
        assert r.json()["status"] == "not_ready"
        assert r.json()["database"] is False

    def test_ready_reports_not_ready_when_the_client_failed_to_load(
        self, client, monkeypatch
    ):
        monkeypatch.setattr(main, "ping", lambda: True)
        monkeypatch.setattr(main, "is_client_loaded", lambda: False)

        r = client.get("/ready")

        assert r.status_code == 503
        assert r.json()["embedding_client"] is False

    def test_ready_is_ok_when_dependencies_answer(self, client, monkeypatch):
        monkeypatch.setattr(main, "ping", lambda: True)
        monkeypatch.setattr(main, "is_client_loaded", lambda: True)

        r = client.get("/ready")

        assert r.status_code == 200
        assert r.json()["status"] == "ready"

    def test_ready_reports_real_startup_state(self, client, monkeypatch):
        monkeypatch.setattr(main, "ping", lambda: True)
        monkeypatch.setattr(main, "is_client_loaded", lambda: True)

        checks = client.get("/ready").json()["checks"]

        # Previously /ready read app.state.model_load_times, which nothing set.
        assert checks["database"] == "ok"
        assert checks["embedding_client"] == "ok"


class TestCorsOrigins:
    def test_localhost_is_always_allowed(self, monkeypatch):
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        assert "http://localhost:3000" in main.get_cors_origins()

    def test_extra_origins_are_appended(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "https://a.com, https://b.com")
        origins = main.get_cors_origins()
        assert "https://a.com" in origins and "https://b.com" in origins

    def test_empty_entries_are_dropped(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "https://a.com,, ,")
        assert main.get_cors_origins().count("") == 0
