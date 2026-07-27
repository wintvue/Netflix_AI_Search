#!/usr/bin/env python3
"""Embedding model client backed by the Hugging Face Inference API."""

import threading

import numpy as np
from huggingface_hub import InferenceClient

from core.config import EMBED_MODEL_NAME, HF_TOKEN, get_logger
from core.errors import ConfigurationError

logger = get_logger(__name__)

# Singleton instance, guarded because worker threads share it.
_client: InferenceClient | None = None
_client_lock = threading.Lock()


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
            _client = InferenceClient(provider="auto", api_key=HF_TOKEN)
            logger.info("Hugging Face client loaded successfully")
    return _client


def is_client_loaded() -> bool:
    """Whether the Hugging Face client has been constructed."""
    return _client is not None


def encode_query(query: str) -> np.ndarray:
    """
    Encode a query string into an embedding vector.

    Resolves the client through get_huggingface_client() rather than reading
    the module global: the global is only populated by the API's lifespan hook,
    so every other entry point (CLI, scripts, __main__ blocks) would otherwise
    dereference None.
    """
    client = get_huggingface_client()
    embedding = client.feature_extraction(
        query,
        model=EMBED_MODEL_NAME,
        normalize=True,
    )
    return np.asarray(embedding, dtype=np.float32)
