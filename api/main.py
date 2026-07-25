#!/usr/bin/env python3
"""FastAPI application for movie search."""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

import psycopg2
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from api.ratelimit import SlidingWindowRateLimiter, client_key
from api.schemas import (
    ErrorResponse,
    HealthResponse,
    KeywordSearchResponse,
    ReadinessResponse,
    SearchResponse,
    SemanticSearchResponse,
)
from core.ai_overview import generate_ai_overview
from core.config import (
    AI_OVERVIEW_RATE_LIMIT,
    DEFAULT_TOP_K,
    HYBRID_ALPHA,
    MAX_TOP_K,
    RERANK_ENABLED,
    candidate_pool_problems,
    get_logger,
    setup_logging,
)
from core.database import close_pool, create_db_pool, ping
from core.errors import ConfigurationError, EmbeddingError, SearchBackendError
from core.model import get_huggingface_client, is_client_loaded
from core.search import hybrid_search, search_movies, semantic_search

logger = get_logger(__name__)

_ai_overview_limiter = SlidingWindowRateLimiter(AI_OVERVIEW_RATE_LIMIT)


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Initializes shared clients and the connection pool so the first request
    does not pay for them, and records what actually succeeded so /ready can
    report the truth rather than a hardcoded optimism.
    """
    setup_logging()
    logger.info("Starting Netflix AI Search API...")

    for problem in candidate_pool_problems():
        logger.warning("Configuration | %s", problem)

    app.state.startup = {}

    start = time.time()
    try:
        get_huggingface_client()
        app.state.startup["embedding_client"] = "ok"
    except Exception as e:
        # Starting without the client lets /ready report the problem instead of
        # crash-looping the deploy before any diagnostics are reachable.
        logger.error("Failed to initialize Hugging Face client: %s", e)
        app.state.startup["embedding_client"] = f"error: {e}"
    app.state.startup["embedding_client_ms"] = round((time.time() - start) * 1000, 2)

    start = time.time()
    try:
        create_db_pool()
        app.state.startup["database"] = "ok"
    except Exception as e:
        logger.error("Failed to create database pool: %s", e)
        app.state.startup["database"] = f"error: {e}"
    app.state.startup["database_ms"] = round((time.time() - start) * 1000, 2)

    yield

    logger.info("Shutting down Netflix AI Search API...")
    close_pool()


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
    """Tag each request with an id and log its outcome and duration."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000

    logger.info(
        "%s %s | Status: %s | Duration: %.2fms | rid=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request_id,
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ==============================================================================
# Error handling
# ==============================================================================


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error_response(
    request: Request, status_code: int, error: str, detail: str | None = None
) -> JSONResponse:
    body = ErrorResponse(
        error=error, detail=detail, request_id=_request_id(request)
    ).model_dump()
    return JSONResponse(status_code=status_code, content=body)


@app.exception_handler(ConfigurationError)
async def handle_configuration_error(request: Request, exc: ConfigurationError):
    logger.error("Configuration error | rid=%s | %s", _request_id(request), exc)
    return _error_response(
        request, 503, "service_misconfigured", "The service is misconfigured."
    )


@app.exception_handler(EmbeddingError)
async def handle_embedding_error(request: Request, exc: EmbeddingError):
    logger.error("Embedding error | rid=%s | %s", _request_id(request), exc)
    return _error_response(
        request, 503, "embedding_unavailable", "Could not embed the query."
    )


@app.exception_handler(SearchBackendError)
async def handle_search_backend_error(request: Request, exc: SearchBackendError):
    logger.error("Search backend error | rid=%s | %s", _request_id(request), exc)
    return _error_response(
        request, 503, "search_unavailable", "A search dependency is unavailable."
    )


@app.exception_handler(psycopg2.Error)
async def handle_database_error(request: Request, exc: psycopg2.Error):
    logger.error("Database error | rid=%s | %s", _request_id(request), exc)
    return _error_response(
        request, 503, "database_unavailable", "The database is unavailable."
    )


@app.exception_handler(TimeoutError)
async def handle_timeout_error(request: Request, exc: TimeoutError):
    logger.error("Timeout | rid=%s | %s", _request_id(request), exc)
    return _error_response(request, 504, "timeout", "The request timed out.")


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    # Log the traceback but never return it: internals do not belong in a
    # response body.
    logger.exception("Unhandled error | rid=%s", _request_id(request))
    return _error_response(
        request, 500, "internal_error", "An unexpected error occurred."
    )


# ==============================================================================
# Search endpoints
# ==============================================================================


@app.get("/search/keyword", response_model=KeywordSearchResponse)
async def search_keyword(
    query: str = Query(..., min_length=1, description="Search query for movie titles"),
    k: int = Query(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K, description="Number of results"),
):
    """Keyword-only search (BM25/full-text) over the movie corpus."""
    results = await asyncio.to_thread(search_movies, query, k)
    return {
        "query": query,
        "count": len(results),
        "results": results,
    }


@app.get("/search/semantic", response_model=SemanticSearchResponse)
async def search_semantic_endpoint(
    q: str = Query(..., min_length=1),
    k: int = Query(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K),
):
    """Search movies using semantic similarity (vector search only)."""
    results = await asyncio.to_thread(semantic_search, q, k)
    return {
        "query": q,
        "count": len(results),
        "results": results,
    }


@app.get(
    "/search",
    response_model=SearchResponse,
    responses={
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def search_hybrid(
    request: Request,
    q: str = Query(..., min_length=1, description="Search query"),
    k: int = Query(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K, description="Number of results"),
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
        description="Enable SSE streaming; requires ai_overview=true",
    ),
):
    """
    Hybrid search using Reciprocal Rank Fusion (RRF).

    Industry-standard approach used by Elasticsearch, Weaviate, Pinecone:

    1. **Parallel Retrieval**: Vector search + BM25/FTS run simultaneously
    2. **RRF Fusion**: Combine rankings using formula: score = α/(k+rank_vec) + (1-α)/(k+rank_bm25)
    3. **Cross-Encoder Reranking**: Rerank top candidates for precision. Best-effort:
       if the reranker is unavailable the RRF order is returned and
       `config.reranked` is false.
    4. **AI Overview** (optional): LLM-generated summary explaining why results match

    **Alpha parameter controls the blend:**
    - `alpha=0.0`: Pure keyword search (BM25)
    - `alpha=0.5`: Balanced hybrid
    - `alpha=1.0`: Pure semantic search (vector)

    **AI Overview:**
    - `ai_overview=true`: Generates an AI summary of the top results
    - Rate limited per client, since generation is a paid upstream call

    **Streaming (SSE):**
    - `stream=true`: Returns Server-Sent Events (SSE); requires `ai_overview=true`
    - Events: `results` (search results), `overview` (AI summary), `done` (stream complete)
    - Client receives results immediately while the AI overview generates
    """
    if stream and not ai_overview:
        raise HTTPException(
            status_code=422,
            detail="stream=true requires ai_overview=true; there is nothing to "
            "stream without the AI overview.",
        )

    if ai_overview and not _ai_overview_limiter.allow(client_key(request)):
        retry_after = _ai_overview_limiter.retry_after(client_key(request))
        logger.warning(
            "AI overview rate limited | rid=%s | client=%s",
            _request_id(request),
            client_key(request),
        )
        return JSONResponse(
            status_code=429,
            content=ErrorResponse(
                error="rate_limited",
                detail=f"AI overview limit reached. Retry in {retry_after}s.",
                request_id=_request_id(request),
            ).model_dump(),
            headers={"Retry-After": str(retry_after)},
        )

    # Run hybrid search in thread pool (blocking operation)
    response = await asyncio.to_thread(hybrid_search, q, k, alpha)

    # Streaming mode: return SSE stream
    if stream:

        async def event_stream():
            # Event 1: Send search results immediately
            yield f"event: results\ndata: {json.dumps(response, default=str)}\n\n"

            # Event 2: Generate AI overview asynchronously in background
            if response.get("results"):
                try:
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
        response["ai_overview"] = await asyncio.to_thread(
            generate_ai_overview, q, response["results"]
        )

    return response


# ==============================================================================
# Probes
# ==============================================================================


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Liveness check: the process is up and serving. Deliberately cheap."""
    return {"status": "healthy"}


@app.get("/ready", response_model=ReadinessResponse, responses={503: {"model": ReadinessResponse}})
async def readiness_check():
    """
    Readiness check: confirms the service can actually serve a search.

    Verifies the database answers and reports whether the embedding client was
    constructed, rather than asserting readiness unconditionally.
    """
    db_ok = await asyncio.to_thread(ping)
    client_ok = is_client_loaded()

    checks = {
        k: str(v) for k, v in getattr(app.state, "startup", {}).items()
    }

    ready = db_ok and client_ok
    body = {
        "status": "ready" if ready else "not_ready",
        "database": db_ok,
        "embedding_client": client_ok,
        "rerank_enabled": RERANK_ENABLED,
        "checks": checks,
    }
    return JSONResponse(status_code=200 if ready else 503, content=body)
