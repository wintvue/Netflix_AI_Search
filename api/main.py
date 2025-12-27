#!/usr/bin/env python3
"""FastAPI application for movie search."""

import sys
import time
from pathlib import Path

from fastapi import FastAPI, Query, Request

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import DEFAULT_TOP_K, get_logger
from core.search import hybrid_search, search_movies, semantic_search

logger = get_logger(__name__)

app = FastAPI(
    title="Netflix Movie Search API",
    description="Search movies by keyword, semantic similarity, or hybrid search with reranking",
    version="2.0.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and response times."""
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    
    logger.info(
        f"{request.method} {request.url.path} "
        f"| Status: {response.status_code} "
        f"| Duration: {duration_ms:.2f}ms"
    )
    return response


@app.get("/search/keyword")
def search_keyword(query: str = Query(..., description="Search query for movie titles")):
    """Search movies by title keyword."""
    results = search_movies(query)
    return {
        "query": query,
        "count": len(results),
        "results": results,
    }


@app.get("/search/semantic")
def search_semantic_endpoint(
    q: str = Query(..., min_length=1),
    k: int = DEFAULT_TOP_K,
):
    """Search movies using semantic similarity (vector search only)."""
    results = semantic_search(q, k)
    return {
        "query": q,
        "count": len(results),
        "results": results,
    }


@app.get("/search")
def search_hybrid(
    q: str = Query(..., min_length=1),
    k: int = DEFAULT_TOP_K,
):
    """Search movies using hybrid FTS + vector search with reranking.
    
    This is the recommended search endpoint that combines:
    - Anchor resolution (e.g., "like Inception" finds the movie)
    - Vector search using anchor embeddings
    - Full-text search (FTS) for keyword matches
    - Cross-encoder reranking for best results
    
    Returns query, anchors, count, and results.
    """
    return hybrid_search(q, k)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
