#!/usr/bin/env python3
"""Configuration and environment variables."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

HF_TOKEN = _env("HF_TOKEN")


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

# Number of fused candidates kept for the final ranking.
RERANK_CANDIDATES = _env_int("RERANK_CANDIDATES", 100)

# Reciprocal Rank Fusion (RRF) parameters.
# RRF formula: score = Σ 1/(k + rank)
# k=60 is the standard constant (from Microsoft's original RRF paper)
RRF_K = _env_int("RRF_K", 60)

# Alpha controls the blend between semantic and keyword search
# alpha=1.0 -> pure vector, alpha=0.0 -> pure BM25
# 0.5-0.7 is typical for balanced hybrid search
HYBRID_ALPHA = _env_float("HYBRID_ALPHA", 0.8)


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
            f"({RERANK_CANDIDATES}): the final ranking will never see a full pool"
        )
    return problems
