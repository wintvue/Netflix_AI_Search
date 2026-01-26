#!/usr/bin/env python3
"""Embedding and reranking model singletons with startup preloading."""

import numpy as np
import os
from core.config import get_logger
from huggingface_hub import InferenceClient

logger = get_logger(__name__)

# Singleton model instances
client: InferenceClient | None = None


def get_huggingface_client() -> InferenceClient:
    """Get the Hugging Face client (loads once on first call)."""
    global client
    if client is None:
        logger.info(f"Loading Hugging Face client")
        client = InferenceClient(
            provider="auto",
            api_key=os.environ["HF_TOKEN"],
        )
        logger.info("Hugging Face client loaded successfully") 
    return client


def encode_query(query: str) -> np.ndarray:
    """Encode a query string into an embedding vector."""
    global client
    embedding = client.feature_extraction(
        query,
        model="sentence-transformers/all-MiniLM-L6-v2",
        normalize=True,
    ).astype(np.float32)
    return np.asarray(embedding, dtype=np.float32)
