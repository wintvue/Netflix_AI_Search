"""Shared test setup.

Tests must behave the same on a developer machine with a populated `.env` as
they do in CI with none, so the suite runs with dotenv loading disabled and with
every variable the app reads cleared from the environment. Tests that care about
a variable set it explicitly.
"""

import dotenv
import pytest

APP_ENV_VARS = (
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "DB_POOL_MIN",
    "DB_POOL_MAX",
    "DB_CONNECT_TIMEOUT",
    "HOST",
    "PORT",
    "RENDER",
    "DYNO",
    "K_SERVICE",
    "WEBSITE_INSTANCE_ID",
    "HF_TOKEN",
    "HF_ROUTER_URL",
    "HF_TIMEOUT_SECONDS",
    "EMBED_MODEL_NAME",
    "EMBED_CACHE_SIZE",
    "RERANK_BACKEND",
    "RERANK_ENABLED",
    "RERANK_MODEL_NAME",
    "RERANK_LOCAL_MODEL_NAME",
    "RERANK_BATCH_SIZE",
    "RERANK_CANDIDATES",
    "VECTOR_CANDIDATES",
    "BM25_CANDIDATES",
    "DEFAULT_TOP_K",
    "MAX_TOP_K",
    "RRF_K",
    "HYBRID_ALPHA",
    "SEARCH_CACHE_SIZE",
    "SEARCH_CACHE_TTL_SECONDS",
    "OLLAMA_HOST",
    "OLLAMA_API_KEY",
    "OLLAMA_MODEL",
    "OLLAMA_TIMEOUT_SECONDS",
    "AI_OVERVIEW_MAX_MOVIES",
    "AI_OVERVIEW_RATE_LIMIT",
    "CORS_ORIGINS",
    "LOG_LEVEL",
)


@pytest.fixture(autouse=True, scope="session")
def _no_dotenv():
    """Stop `core.config` from reading the developer's .env on import or reload."""
    original = dotenv.load_dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False
    try:
        yield
    finally:
        dotenv.load_dotenv = original


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in APP_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
