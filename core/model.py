#!/usr/bin/env python3
"""Embedding and reranking model singletons."""

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from core.config import EMBED_MODEL_NAME, RERANK_MODEL_NAME, get_logger

logger = get_logger(__name__)

# Singleton model instances
_embed_model = None
_reranker = None


def get_embed_model() -> SentenceTransformer:
    """Get the embedding model (loads once on first call)."""
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        logger.info("Embedding model loaded successfully")
    return _embed_model


def get_reranker() -> CrossEncoder:
    """Get the reranker model (loads once on first call)."""
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model: {RERANK_MODEL_NAME}")
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
        logger.info("Reranker model loaded successfully")
    return _reranker


def encode_query(query: str) -> np.ndarray:
    """Encode a query string into an embedding vector."""
    model = get_embed_model()
    return model.encode([query], normalize_embeddings=True).astype(np.float32)[0]


def rerank(query: str, texts: list[str]) -> list[float]:
    """Rerank texts based on relevance to query.
    
    Args:
        query: The search query.
        texts: List of document texts to rerank.
        
    Returns:
        List of relevance scores (higher = more relevant).
    """
    reranker = get_reranker()
    pairs = [(query, text) for text in texts]
    scores = reranker.predict(pairs)
    return [float(s) for s in scores]
