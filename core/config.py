#!/usr/bin/env python3
"""Configuration and environment variables."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Database configuration
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("HOST")
DB_PORT = os.getenv("PORT")
DB_NAME = os.getenv("DB_NAME")

# Model configuration
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Search defaults
DEFAULT_TOP_K = 100

# Hybrid search configuration (industry-standard RRF approach)
# Candidate pool sizes for each retrieval method
VECTOR_CANDIDATES = 100   # Vector (semantic) search candidates
BM25_CANDIDATES = 100     # Full-text search (BM25/FTS) candidates

# Reciprocal Rank Fusion (RRF) parameters
# RRF formula: score = Σ 1/(k + rank)
# k=60 is the standard constant (from Microsoft's original RRF paper)
RRF_K = 60

# Alpha controls the blend between semantic and keyword search
# alpha=1.0 -> pure vector, alpha=0.0 -> pure BM25
# 0.5-0.7 is typical for balanced hybrid search
HYBRID_ALPHA = 0.8

# Number of candidates to send to the reranker (top RRF results)
RERANK_CANDIDATES = 50

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger."""
    return logging.getLogger(name)

