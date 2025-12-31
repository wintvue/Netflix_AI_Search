#!/usr/bin/env python3
"""FastAPI application for movie search."""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ai_overview import generate_ai_overview
from core.config import DEFAULT_TOP_K, HYBRID_ALPHA, get_logger
from core.model import preload_models
from core.search import hybrid_search, search_movies, semantic_search

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Preloads all ML models at startup so the first request is fast.
    """
    # Startup: Load all models
    logger.info("Starting Netflix AI Search API...")
    load_times = preload_models()
    app.state.model_load_times = load_times
    
    yield  # Application runs here
    
    # Shutdown: Cleanup if needed
    logger.info("Shutting down Netflix AI Search API...")


app = FastAPI(
    title="Netflix Movie Search API",
    description="Search movies by keyword, semantic similarity, or hybrid search with reranking",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
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
    q: str = Query(..., min_length=1, description="Search query"),
    k: int = Query(DEFAULT_TOP_K, ge=1, le=100, description="Number of results"),
    alpha: float = Query(
        HYBRID_ALPHA,
        ge=0.0,
        le=1.0,
        description="Blend weight: 0=pure keyword, 1=pure semantic, 0.5=balanced"
    ),
    ai_overview: bool = Query(
        False,
        description="Generate AI-powered overview explaining search results"
    ),
):
    """
    Hybrid search using Reciprocal Rank Fusion (RRF).
    
    Industry-standard approach used by Elasticsearch, Weaviate, Pinecone:
    
    1. **Parallel Retrieval**: Vector search + BM25/FTS run simultaneously
    2. **RRF Fusion**: Combine rankings using formula: score = α/(k+rank_vec) + (1-α)/(k+rank_bm25)
    3. **Cross-Encoder Reranking**: Rerank top candidates for precision
    4. **AI Overview** (optional): LLM-generated summary explaining why results match
    
    **Alpha parameter controls the blend:**
    - `alpha=0.0`: Pure keyword search (BM25)
    - `alpha=0.5`: Balanced hybrid (default)
    - `alpha=1.0`: Pure semantic search (vector)
    
    **AI Overview:**
    - `ai_overview=true`: Generates an AI summary of results using Ollama
    - Uses qwen2.5:7b model for fast, accurate explanations
    
    Returns query, config, timings, retrieval stats, ranked results, and optional AI overview.
    """
    response = hybrid_search(q, k, alpha=alpha)
    
    # Generate AI overview if requested
    if ai_overview and response.get("results"):
        overview_result = generate_ai_overview(q, response["results"])
        response["ai_overview"] = overview_result
    
    return response


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check():
    """
    Readiness check - confirms models are loaded and ready.
    
    Returns model load times from startup.
    """
    return {
        "status": "ready",
        "models_loaded": True,
        "load_times": getattr(app.state, "model_load_times", {}),
    }
