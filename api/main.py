#!/usr/bin/env python3
"""FastAPI application for movie search."""

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_cors_origins() -> list[str]:
    """
    Get CORS origins based on environment.

    Set CORS_ORIGINS env var as comma-separated URLs:
    CORS_ORIGINS=https://myapp.com,https://staging.myapp.com
    """
    # Always allow localhost for development
    origins = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ]

    # Add production/staging origins from env
    extra_origins = os.getenv("CORS_ORIGINS", "")
    if extra_origins:
        origins.extend([o.strip() for o in extra_origins.split(",") if o.strip()])

    return origins


from core.ai_overview import generate_ai_overview
from core.config import DEFAULT_TOP_K, HYBRID_ALPHA, get_logger
from core.search import hybrid_search, search_movies, semantic_search
from core.database import create_db_pool, close_connection
from core.model import get_huggingface_client

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Preloads all ML models at startup so the first request is fast.
    """
    # Startup: Load all models
    logger.info("Starting Netflix AI Search API...")
    # load_times = preload_models()
    get_huggingface_client()
    create_db_pool()
    yield  # Application runs here

    # Shutdown: Cleanup if needed
    logger.info("Shutting down Netflix AI Search API...")
    close_connection()
    logger.info("Database connection closed.")


app = FastAPI(
    title="Netflix Movie Search API",
    description="Search movies by keyword, semantic similarity, or hybrid search with reranking",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
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
def search_keyword(
    query: str = Query(..., description="Search query for movie titles")
):
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
async def search_hybrid(
    q: str = Query(..., min_length=1, description="Search query"),
    k: int = Query(DEFAULT_TOP_K, ge=1, le=100, description="Number of results"),
    alpha: float = Query(
        HYBRID_ALPHA,
        ge=0.0,
        le=1.0,
        description="Blend weight: 0=pure keyword, 1=pure semantic, 0.5=balanced",
    ),
    ai_overview: bool = Query(
        False, description="Generate AI-powered overview explaining search results"
    ),
    stream: bool = Query(
        False,
        description="Enable SSE streaming (returns results immediately, then AI overview)",
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

    **Streaming (SSE):**
    - `stream=true`: Returns Server-Sent Events (SSE) stream
    - Events: `results` (search results), `overview` (AI summary), `done` (stream complete)
    - Client receives results immediately while AI overview generates in background

    Returns query, config, timings, retrieval stats, ranked results, and optional AI overview.
    """
    # Run hybrid search in thread pool (blocking operation)
    response = await asyncio.to_thread(hybrid_search, q, k, alpha)

    # Streaming mode: return SSE stream
    if stream and ai_overview:

        async def event_stream():
            # Event 1: Send search results immediately
            yield f"event: results\ndata: {json.dumps(response, default=str)}\n\n"

            # Event 2: Generate AI overview asynchronously in background
            if response.get("results"):
                try:
                    # Run AI overview generation in thread pool (non-blocking)
                    overview_result = await asyncio.to_thread(
                        generate_ai_overview, q, response["results"]
                    )
                    yield f"event: overview\ndata: {json.dumps(overview_result, default=str)}\n\n"
                except Exception as e:
                    logger.error(f"AI overview generation failed: {e}")
                    yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

            # Event 3: Signal stream completion
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    # Non-streaming mode: return regular JSON response
    if ai_overview and response.get("results"):
        overview_result = await asyncio.to_thread(
            generate_ai_overview, q, response["results"]
        )
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
