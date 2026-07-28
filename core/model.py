#!/usr/bin/env python3
"""Embedding and reranking backed by the Hugging Face Inference API."""

import time
from functools import lru_cache

import numpy as np
from huggingface_hub import InferenceClient

from core.config import (
    EMBED_CACHE_SIZE,
    EMBED_MODEL_NAME,
    HF_ROUTER_URL,
    HF_TIMEOUT_SECONDS,
    HF_TOKEN,
    RERANK_BACKEND,
    RERANK_BATCH_SIZE,
    RERANK_LOCAL_MODEL_NAME,
    RERANK_MODEL_NAME,
    get_logger,
)
from core.errors import ConfigurationError, EmbeddingError, RerankError

logger = get_logger(__name__)

# Lazily built singletons shared by worker threads. Unlocked: a race builds one
# extra instance that is immediately dropped, which is cheaper than serializing
# every caller behind a mutex for the life of the process.
_client: InferenceClient | None = None
_local_reranker = None

# One retry only: a longer chain would outlive the caller's own timeout budget.
_MAX_ATTEMPTS = 2
_RETRY_BACKOFF_SECONDS = 0.25


def get_huggingface_client() -> InferenceClient:
    """Get the Hugging Face client (constructed once, on first use)."""
    global _client
    if _client is not None:
        return _client

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


# ==============================================================================
# Embedding
# ==============================================================================


@lru_cache(maxsize=EMBED_CACHE_SIZE)
def _encode_query_cached(query: str) -> np.ndarray:
    """Embed a query, memoized on the normalized query string."""
    client = get_huggingface_client()

    def call() -> np.ndarray:
        return client.feature_extraction(
            query,
            model=EMBED_MODEL_NAME,
            normalize=True,
        )

    raw = _with_retry("Query embedding", call)
    embedding = np.asarray(raw, dtype=np.float32)

    # Cached arrays are shared between requests; freeze so a caller cannot
    # corrupt the entry for everyone else.
    embedding.flags.writeable = False
    return embedding


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
    normalized = query.strip().lower()
    if not normalized:
        raise EmbeddingError("Cannot embed an empty query")

    try:
        return _encode_query_cached(normalized)
    except ConfigurationError:
        raise
    except Exception as e:
        raise EmbeddingError(f"Failed to embed query: {e}") from e


def embedding_cache_info() -> dict:
    """Hit/miss statistics for the query embedding cache."""
    info = _encode_query_cached.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "size": info.currsize,
        "maxsize": info.maxsize,
    }


# ==============================================================================
# Reranking
# ==============================================================================


def _rerank_hosted(query: str, documents: list[str]) -> list[float]:
    """
    Score (query, document) pairs with a hosted cross-encoder.

    The reranker is served as a text-classification model over sentence pairs.
    InferenceClient.text_classification only accepts a single string, so the
    pair payload goes to the router endpoint directly, reusing the same token
    and timeout as the rest of the HF integration.
    """
    import httpx

    if not HF_TOKEN:
        raise ConfigurationError("HF_TOKEN is not set; cannot rerank")

    url = f"{HF_ROUTER_URL}/models/{RERANK_MODEL_NAME}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    scores: list[float] = []

    with httpx.Client(timeout=HF_TIMEOUT_SECONDS) as http:
        for start in range(0, len(documents), RERANK_BATCH_SIZE):
            batch = documents[start : start + RERANK_BATCH_SIZE]
            payload = {
                "inputs": [{"text": query, "text_pair": doc} for doc in batch],
            }

            def call():
                response = http.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()

            body = _with_retry("Rerank request", call)
            scores.extend(_parse_rerank_response(body, expected=len(batch)))

    return scores


def _parse_rerank_response(body, expected: int) -> list[float]:
    """
    Pull one relevance score per input out of a text-classification response.

    Two shapes occur in practice. Standard text-classification returns one list
    of label/score dicts per input pair. The hf-inference router serving
    BAAI/bge-reranker-base instead wraps the whole batch in a single extra list
    -- `[[{...}, {...}, ...]]`, one dict per pair, in input order -- so unwrap
    that before scoring. For a two-label model the positive class is the one to
    keep; single-label rerankers emit just the relevance score.
    """
    if not isinstance(body, list):
        raise RerankError(f"Unexpected rerank response shape: {type(body).__name__}")

    if (
        len(body) == 1
        and expected != 1
        and isinstance(body[0], list)
        and len(body[0]) == expected
    ):
        body = body[0]

    if len(body) != expected:
        raise RerankError(
            f"Rerank returned {len(body)} predictions for {expected} inputs"
        )

    scores: list[float] = []
    for entry in body:
        # A single-input request may come back unwrapped.
        labels = entry if isinstance(entry, list) else [entry]
        if not labels:
            raise RerankError("Rerank response contained an empty prediction")

        if len(labels) == 1:
            scores.append(float(labels[0]["score"]))
        else:
            # Two-label cross-encoder: take the highest-indexed label, which is
            # the "relevant" class for the standard LABEL_0/LABEL_1 ordering.
            positive = max(labels, key=lambda item: str(item.get("label", "")))
            scores.append(float(positive["score"]))

    return scores


def _get_local_reranker():
    """Lazily load a local cross-encoder (requires the `rerank` extra)."""
    global _local_reranker
    if _local_reranker is not None:
        return _local_reranker

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ConfigurationError(
            "RERANK_BACKEND=local needs sentence-transformers: "
            "pip install -e '.[rerank]'"
        ) from e

    logger.info("Loading local reranker: %s", RERANK_LOCAL_MODEL_NAME)
    _local_reranker = CrossEncoder(RERANK_LOCAL_MODEL_NAME)
    logger.info("Local reranker loaded")
    return _local_reranker


def _rerank_local(query: str, documents: list[str]) -> list[float]:
    """Score (query, document) pairs with an in-process cross-encoder."""
    model = _get_local_reranker()
    raw = model.predict(
        [(query, doc) for doc in documents],
        batch_size=RERANK_BATCH_SIZE,
    )
    return [float(score) for score in raw]


def rerank(query: str, documents: list[str]) -> list[float]:
    """
    Score each document's relevance to the query. Higher is more relevant.

    Returns one score per document, in input order.

    Raises:
        RerankError: scoring failed; callers should keep their existing order.
    """
    if not documents:
        return []

    try:
        if RERANK_BACKEND == "local":
            scores = _rerank_local(query, documents)
        elif RERANK_BACKEND == "hf":
            scores = _rerank_hosted(query, documents)
        else:
            raise RerankError(f"Unknown RERANK_BACKEND: {RERANK_BACKEND!r}")
    except RerankError:
        raise
    except Exception as e:
        raise RerankError(f"Reranking failed: {e}") from e

    if len(scores) != len(documents):
        raise RerankError(
            f"Reranker returned {len(scores)} scores for {len(documents)} documents"
        )
    return scores
