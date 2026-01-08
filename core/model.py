#!/usr/bin/env python3
"""Embedding and reranking model singletons with startup preloading."""

import time

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from core.config import EMBED_MODEL_NAME, RERANK_MODEL_NAME, get_logger

logger = get_logger(__name__)

# Singleton model instances
_embed_model: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None


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


def preload_models() -> dict:
    """
    Preload all models at application startup.
    
    This ensures the first request doesn't have to wait for model loading.
    Call this during FastAPI lifespan or at module import.
    
    Returns:
        Dict with model names and load times
    """
    logger.info("=" * 60)
    logger.info("PRELOADING MODELS AT STARTUP")
    logger.info("=" * 60)
    
    load_times = {}
    
    # Load embedding model
    start = time.time()
    embed_model = get_embed_model()
    load_times["embed_model"] = time.time() - start
    logger.info(f"Embedding model ready in {load_times['embed_model']:.2f}s")
    
    # Warm up embedding model with a test query
    start = time.time()
    _ = embed_model.encode(["warmup query"], normalize_embeddings=True)
    load_times["embed_warmup"] = time.time() - start
    logger.info(f"Embedding warmup in {load_times['embed_warmup']:.2f}s")
    
    total = sum(load_times.values())
    logger.info("=" * 60)
    logger.info(f"ALL MODELS READY | Total: {total:.2f}s")
    logger.info("=" * 60)
    
    return load_times


# def encode_query(query: str) -> str:
#     """Encode a query string into an embedding vector."""
#     model = get_embed_model()
#     embedding = model.encode([query], normalize_embeddings=True).astype(np.float32)[0]
#     vec_str = "[" + ",".join(f"{x:.6f}" for x in embedding.tolist()) + "]"
#     return vec_str

def encode_query(query: str) -> np.ndarray:
    """Encode a query string into an embedding vector."""
    model = get_embed_model()
    embedding = model.encode([query], normalize_embeddings=True).astype(np.float32)[0]
    return np.asarray(embedding, dtype=np.float32)

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
