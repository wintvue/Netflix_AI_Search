#!/usr/bin/env python3
"""Configuration and environment variables."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger."""
    return logging.getLogger(name)


logger = get_logger(__name__)


# ==============================================================================
# Environment helpers
# ==============================================================================

# Managed platforms inject PORT (and sometimes HOST) for the *web* listener.
# If we let those leak into the database settings the app tries to connect to
# Postgres on the web port. Detect the platform so the legacy fallbacks below
# are only honoured on a developer machine.
_PLATFORM_MARKERS = ("RENDER", "DYNO", "K_SERVICE", "WEBSITE_INSTANCE_ID")


def _on_managed_platform() -> bool:
    """True when running on a PaaS that injects its own HOST/PORT variables."""
    return any(os.getenv(marker) for marker in _PLATFORM_MARKERS)


def _env(name: str, default: str | None = None) -> str | None:
    """Read an env var, treating an empty string as unset."""
    value = os.getenv(name)
    return value if value else default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer, falling back to %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number, falling back to %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _legacy_env(name: str, legacy_name: str, default: str | None = None) -> str | None:
    """
    Read `name`, falling back to a legacy variable on developer machines only.

    The legacy names (HOST, PORT) collide with the variables PaaS providers set
    for the web listener, so the fallback is disabled there.
    """
    value = _env(name)
    if value is not None:
        return value

    legacy = _env(legacy_name)
    if legacy is None:
        return default

    if _on_managed_platform():
        logger.warning(
            "Ignoring %s=%r as a source for %s: on a managed platform that "
            "variable belongs to the web listener. Set %s explicitly.",
            legacy_name,
            legacy,
            name,
            name,
        )
        return default

    logger.warning(
        "%s is unset; falling back to deprecated %s. Rename it to %s.",
        name,
        legacy_name,
        name,
    )
    return legacy


# ==============================================================================
# Database configuration
# ==============================================================================

DB_USER = _env("DB_USER")
DB_PASSWORD = _env("DB_PASSWORD")
DB_NAME = _env("DB_NAME")
DB_HOST = _legacy_env("DB_HOST", "HOST")

_db_port_raw = _legacy_env("DB_PORT", "PORT", default="5432")
try:
    DB_PORT = int(_db_port_raw)
except (TypeError, ValueError):
    logger.warning("DB_PORT=%r is not an integer, falling back to 5432", _db_port_raw)
    DB_PORT = 5432

DB_POOL_MIN = _env_int("DB_POOL_MIN", 1)
DB_POOL_MAX = _env_int("DB_POOL_MAX", 10)
DB_CONNECT_TIMEOUT = _env_int("DB_CONNECT_TIMEOUT", 5)


def missing_db_settings() -> list[str]:
    """Names of required database settings that are unset."""
    required = {
        "DB_USER": DB_USER,
        "DB_PASSWORD": DB_PASSWORD,
        "DB_NAME": DB_NAME,
        "DB_HOST": DB_HOST,
    }
    return sorted(name for name, value in required.items() if not value)


# ==============================================================================
# Model configuration
# ==============================================================================

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Reranking backend: "hf" calls the hosted Inference API, "local" loads a
# cross-encoder into this process, "none" disables the stage.
#
# The originally-configured cross-encoder/ms-marco-MiniLM-L-6-v2 has been
# renamed upstream (it now redirects to .../ms-marco-MiniLM-L6-v2) and, more
# importantly, no inference provider serves it -- its inferenceProviderMapping
# is empty, so it can only run locally. BAAI/bge-reranker-base is a comparable
# cross-encoder that *is* live on hf-inference, exposed as text-classification
# over sentence pairs, so it is the default for the hosted path.
RERANK_BACKEND = (_env("RERANK_BACKEND", "hf") or "hf").lower()
RERANK_MODEL_NAME = _env("RERANK_MODEL_NAME", "BAAI/bge-reranker-base")
RERANK_LOCAL_MODEL_NAME = _env(
    "RERANK_LOCAL_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L6-v2"
)
RERANK_BATCH_SIZE = _env_int("RERANK_BATCH_SIZE", 32)

HF_TOKEN = _env("HF_TOKEN")
HF_ROUTER_URL = _env("HF_ROUTER_URL", "https://router.huggingface.co/hf-inference")

# Every embedding/rerank call is a network round trip; without a timeout a slow
# upstream pins a worker thread for as long as the socket stays open.
HF_TIMEOUT_SECONDS = _env_float("HF_TIMEOUT_SECONDS", 10.0)

# Queries repeat heavily in a search workload, and the embedding is on the
# critical path before the database is touched.
EMBED_CACHE_SIZE = _env_int("EMBED_CACHE_SIZE", 512)

# Reranking is the most expensive stage; keep it switchable without a redeploy.
# It is best-effort: a failure logs and falls back to the RRF order rather than
# failing the search, so defaulting it on cannot take the endpoint down.
RERANK_ENABLED = _env_bool("RERANK_ENABLED", True) and RERANK_BACKEND != "none"


# ==============================================================================
# Search configuration
# ==============================================================================

# Default page size. Callers that want more must ask for it explicitly.
DEFAULT_TOP_K = _env_int("DEFAULT_TOP_K", 10)

# Largest `k` the API will serve. The candidate pools below must be at least
# this large or the pipeline silently truncates the response.
MAX_TOP_K = _env_int("MAX_TOP_K", 100)

# Candidate pool sizes for each retrieval method.
VECTOR_CANDIDATES = _env_int("VECTOR_CANDIDATES", 150)
BM25_CANDIDATES = _env_int("BM25_CANDIDATES", 150)

# Number of fused candidates handed to the reranker.
RERANK_CANDIDATES = _env_int("RERANK_CANDIDATES", 100)

# Reciprocal Rank Fusion (RRF) parameters.
# RRF formula: score = Σ 1/(k + rank)
# k=60 is the standard constant (from Microsoft's original RRF paper)
RRF_K = _env_int("RRF_K", 60)

# Alpha controls the blend between semantic and keyword search
# alpha=1.0 -> pure vector, alpha=0.0 -> pure BM25
# 0.5-0.7 is typical for balanced hybrid search
HYBRID_ALPHA = _env_float("HYBRID_ALPHA", 0.8)

# Cache of complete search responses, keyed on (query, k, alpha).
SEARCH_CACHE_SIZE = _env_int("SEARCH_CACHE_SIZE", 256)
SEARCH_CACHE_TTL_SECONDS = _env_float("SEARCH_CACHE_TTL_SECONDS", 60.0)


def candidate_pool_problems() -> list[str]:
    """
    Descriptions of any violated candidate-pool invariants.

    The pipeline narrows at every stage, so the pools must satisfy
    VECTOR_CANDIDATES / BM25_CANDIDATES >= RERANK_CANDIDATES >= MAX_TOP_K.
    Otherwise a caller asking for `k` results quietly receives fewer.
    """
    problems = []
    if RERANK_CANDIDATES < MAX_TOP_K:
        problems.append(
            f"RERANK_CANDIDATES ({RERANK_CANDIDATES}) < MAX_TOP_K ({MAX_TOP_K}): "
            f"requests for more than {RERANK_CANDIDATES} results will be truncated"
        )
    if VECTOR_CANDIDATES < RERANK_CANDIDATES and BM25_CANDIDATES < RERANK_CANDIDATES:
        problems.append(
            f"VECTOR_CANDIDATES ({VECTOR_CANDIDATES}) and BM25_CANDIDATES "
            f"({BM25_CANDIDATES}) are both below RERANK_CANDIDATES "
            f"({RERANK_CANDIDATES}): the reranker will never see a full pool"
        )
    return problems


# ==============================================================================
# AI overview configuration
# ==============================================================================

OLLAMA_MODEL = _env("OLLAMA_MODEL", "ministral-3:8b-cloud")
OLLAMA_HOST = _env("OLLAMA_HOST", "https://ollama.com")
OLLAMA_API_KEY = _env("OLLAMA_API_KEY")
OLLAMA_KEEP_ALIVE = _env("OLLAMA_KEEP_ALIVE", "10m")
OLLAMA_TIMEOUT_SECONDS = _env_float("OLLAMA_TIMEOUT_SECONDS", 60.0)

# Every movie in the prompt costs input tokens. Users read the top of an
# overview, so summarising the whole result page is spend without a reader.
AI_OVERVIEW_MAX_MOVIES = _env_int("AI_OVERVIEW_MAX_MOVIES", 5)

# Requests per minute per client IP for the (paid) AI overview path.
AI_OVERVIEW_RATE_LIMIT = _env_int("AI_OVERVIEW_RATE_LIMIT", 10)


# ==============================================================================
# Logging
# ==============================================================================

LOG_LEVEL = (_env("LOG_LEVEL", "INFO") or "INFO").upper()

_logging_configured = False


def setup_logging(level: str | None = None) -> None:
    """
    Configure root logging. Idempotent, so entry points can call it freely.

    Kept out of import time so that importing config has no side effects.
    """
    global _logging_configured
    if _logging_configured:
        return

    logging.basicConfig(
        level=getattr(logging, level or LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _logging_configured = True
