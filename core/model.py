#!/usr/bin/env python3
"""Embedding model client backed by the Hugging Face Inference API."""

import threading
import time

import numpy as np
from huggingface_hub import InferenceClient

from core.config import (
    EMBED_MODEL_NAME,
    HF_TIMEOUT_SECONDS,
    HF_TOKEN,
    get_logger,
)
from core.errors import ConfigurationError, EmbeddingError

logger = get_logger(__name__)

# Singleton instance, guarded because worker threads share it.
_client: InferenceClient | None = None
_client_lock = threading.Lock()

# One retry only: a longer chain would outlive the caller's own timeout budget.
_MAX_ATTEMPTS = 2
_RETRY_BACKOFF_SECONDS = 0.25


def get_huggingface_client() -> InferenceClient:
    """Get the Hugging Face client (constructed once, on first use)."""
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is None:
            if not HF_TOKEN:
                raise ConfigurationError(
                    "HF_TOKEN is not set; the Hugging Face Inference API cannot "
                    "be reached. Add it to your .env (see .env.example)."
                )
            logger.info("Loading Hugging Face client")
            _client = InferenceClient(
                provider="auto",
                api_key=HF_TOKEN,
                timeout=HF_TIMEOUT_SECONDS,
            )
            logger.info("Hugging Face client loaded successfully")
    return _client


def is_client_loaded() -> bool:
    """Whether the Hugging Face client has been constructed."""
    return _client is not None


def _with_retry(operation: str, fn):
    """Run `fn`, retrying once on failure before giving up."""
    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - upstream raises many types
            last_error = e
            if attempt < _MAX_ATTEMPTS:
                logger.warning(
                    "%s failed (attempt %d/%d): %s",
                    operation,
                    attempt,
                    _MAX_ATTEMPTS,
                    e,
                )
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
    raise last_error  # type: ignore[misc]


def encode_query(query: str) -> np.ndarray:
    """
    Encode a query string into an embedding vector.

    Resolves the client through get_huggingface_client() rather than reading
    the module global: the global is only populated by the API's lifespan hook,
    so every other entry point (CLI, scripts, __main__ blocks) would otherwise
    dereference None.

    Raises:
        EmbeddingError: the upstream API could not be reached.
        ConfigurationError: HF_TOKEN is unset.
    """
    if not query.strip():
        raise EmbeddingError("Cannot embed an empty query")

    client = get_huggingface_client()

    def call() -> np.ndarray:
        return client.feature_extraction(
            query,
            model=EMBED_MODEL_NAME,
            normalize=True,
        )

    try:
        raw = _with_retry("Query embedding", call)
    except Exception as e:
        raise EmbeddingError(f"Failed to embed query: {e}") from e

    return np.asarray(raw, dtype=np.float32)
