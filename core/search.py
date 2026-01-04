#!/usr/bin/env python3
"""
Hybrid Search using Reciprocal Rank Fusion (RRF).

This implements the industry-standard hybrid search approach used by:
- Elasticsearch/OpenSearch (hybrid search with RRF)
- Weaviate, Pinecone, Qdrant (vector databases)
- Microsoft/Bing (original RRF paper authors)
- Cohere (rerank API)

The pipeline:
1. Parallel Retrieval: Vector search + BM25/FTS search run simultaneously
2. Score Normalization: Min-max normalization per retrieval method
3. Reciprocal Rank Fusion: RRF formula combines rankings
4. Cross-Encoder Reranking: Final precision boost on top candidates
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import psycopg2.extras

from core.config import (
    BM25_CANDIDATES,
    DEFAULT_TOP_K,
    HYBRID_ALPHA,
    RERANK_CANDIDATES,
    RRF_K,
    VECTOR_CANDIDATES,
    get_logger,
)
from core.database import get_connection, put_connection
from core.model import encode_query, get_reranker

logger = get_logger(__name__)


# ==============================================================================
# SQL Queries
# ==============================================================================

SQL_VECTOR_SEARCH = """
SELECT
    e.movie_id AS id,
    1 - (e.embedding <=> %s) AS score  -- Convert distance to similarity
FROM movie_embeddings e
ORDER BY e.embedding <=> %s ASC
LIMIT %s;
"""

SQL_BM25_SEARCH = """
SELECT
    id,
    ts_rank_cd(
        to_tsvector('english',
            coalesce(title, '') || ' ' ||
            coalesce(original_title, '') || ' ' ||
            coalesce(overview, '') || ' ' ||
            coalesce(tagline, '')
        ),
        websearch_to_tsquery('english', %s),
        32  -- Normalization: divide by (1 + log(doc_length))
    ) AS score
FROM movies
WHERE to_tsvector('english',
        coalesce(title, '') || ' ' ||
        coalesce(original_title, '') || ' ' ||
        coalesce(overview, '') || ' ' ||
        coalesce(tagline, '')
    ) @@ websearch_to_tsquery('english', %s)
ORDER BY score DESC
LIMIT %s;
"""

SQL_FETCH_MOVIES = """
SELECT
    id, title, original_title, overview, tagline, genres,
    release_date, original_language, poster_path,
    vote_average, vote_count, popularity
FROM movies
WHERE id = ANY(%s);
"""

SQL_SEMANTIC = """
SELECT
    m.id, m.title, m.release_date, m.poster_path,
    m.vote_average, m.vote_count, m.popularity,
    (e.embedding <=> %s) AS distance
FROM movie_embeddings e
JOIN movies m ON m.id = e.movie_id
ORDER BY distance ASC
LIMIT %s;
"""


# ==============================================================================
# Type Definitions
# ==============================================================================

class Movie(TypedDict):
    id: int
    title: str
    description: str


@dataclass
class RetrievalResult:
    """Result from a single retrieval method."""
    id: int
    score: float
    rank: int


@dataclass
class FusedResult:
    """Result after RRF fusion."""
    id: int
    rrf_score: float
    vector_rank: int | None
    bm25_rank: int | None


# ==============================================================================
# Helper Functions
# ==============================================================================

def log_json(data: list | dict) -> str:
    """Format data as JSON for structured logging."""
    return json.dumps(data, indent=2, default=str)


def normalize_scores(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """
    Min-max normalize scores to [0, 1] range.
    
    This ensures scores from different retrieval methods are comparable.
    """
    if not results:
        return results
    
    scores = [r.score for r in results]
    min_score, max_score = min(scores), max(scores)
    
    if max_score == min_score:
        # All scores are the same, set to 1.0
        return [RetrievalResult(r.id, 1.0, r.rank) for r in results]
    
    return [
        RetrievalResult(r.id, (r.score - min_score) / (max_score - min_score), r.rank)
        for r in results
    ]


def compute_rrf(
    vector_results: list[RetrievalResult],
    bm25_results: list[RetrievalResult],
    k: int = RRF_K,
    alpha: float = HYBRID_ALPHA,
) -> list[FusedResult]:
    """
    Compute Reciprocal Rank Fusion (RRF) scores.
    
    RRF Formula: score = α * 1/(k + rank_vector) + (1-α) * 1/(k + rank_bm25)
    
    Args:
        vector_results: Results from vector search (semantic)
        bm25_results: Results from BM25/FTS search (keyword)
        k: RRF constant (default 60, from Microsoft's paper)
        alpha: Weight for vector search (0=pure BM25, 1=pure vector)
    
    Returns:
        Fused results sorted by RRF score descending
    """
    # Build rank maps
    vector_ranks = {r.id: r.rank for r in vector_results}
    bm25_ranks = {r.id: r.rank for r in bm25_results}
    
    # Get all unique IDs
    all_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
    
    fused = []
    for doc_id in all_ids:
        vec_rank = vector_ranks.get(doc_id)
        bm25_rank = bm25_ranks.get(doc_id)
        
        # Compute RRF score
        vec_rrf = alpha * (1.0 / (k + vec_rank)) if vec_rank is not None else 0.0
        bm25_rrf = (1 - alpha) * (1.0 / (k + bm25_rank)) if bm25_rank is not None else 0.0
        
        rrf_score = vec_rrf + bm25_rrf
        
        fused.append(FusedResult(
            id=doc_id,
            rrf_score=rrf_score,
            vector_rank=vec_rank,
            bm25_rank=bm25_rank,
        ))
    
    # Sort by RRF score descending
    fused.sort(key=lambda x: x.rrf_score, reverse=True)
    
    return fused


def build_rerank_text(movie: dict) -> str:
    """Build text representation for cross-encoder reranking."""
    parts = [
        f"Title: {movie.get('title') or ''}",
        f"Genres: {movie.get('genres') or ''}",
        f"Tagline: {movie.get('tagline') or ''}",
        f"Overview: {movie.get('overview') or ''}",
    ]
    return "\n".join(p for p in parts if not p.endswith(": "))


# ==============================================================================
# Legacy Keyword Search (from JSON file)
# ==============================================================================

DATA_PATH = Path(__file__).parent.parent / "data" / "movies.json"
with open(DATA_PATH) as f:
    _MOVIES_DATA = json.load(f)


def search_movies(query: str) -> list[Movie]:
    """Search movies by title keyword (legacy JSON-based search)."""
    logger.info(f"Keyword search | Query: '{query}'")
    
    query_lower = query.lower()
    results: list[Movie] = [
        movie for movie in _MOVIES_DATA["movies"]
        if query_lower in movie["title"].lower()
    ]
    
    logger.info(f"Keyword search | Found: {len(results)} results")
    return results


# ==============================================================================
# Semantic Search (Vector Only)
# ==============================================================================

def semantic_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Pure vector search using semantic similarity.
    
    Args:
        query: Natural language query
        top_k: Number of results to return
    
    Returns:
        List of movies ranked by semantic similarity
    """
    logger.info(f"Semantic search | Query: '{query}' | Top-K: {top_k}")
    
    # Encode query
    start = time.time()
    q_emb = encode_query(query)
    encode_time = (time.time() - start) * 1000
    logger.info(f"Semantic search | Encoding: {encode_time:.2f}ms")

    # Database search
    start = time.time()
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(SQL_SEMANTIC, (q_emb, top_k))
            rows = cur.fetchall()
    finally:
        put_connection(conn)
    
    db_time = (time.time() - start) * 1000
    results = [dict(row) for row in rows]
    
    logger.info(f"Semantic search | DB: {db_time:.2f}ms | Results: {len(results)}")
    logger.info(f"Semantic search | Results:\n{log_json([
        {'rank': i+1, 'id': r['id'], 'title': r['title'], 'distance': round(r['distance'], 4)}
        for i, r in enumerate(results)
    ])}")
    
    return results


# ==============================================================================
# Hybrid Search (RRF-based)
# ==============================================================================

def hybrid_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = HYBRID_ALPHA,
) -> dict:
    """
    Hybrid search using Reciprocal Rank Fusion (RRF).
    
    This is the industry-standard approach for combining semantic and keyword search:
    
    1. **Parallel Retrieval**: Run vector search + BM25/FTS simultaneously
    2. **Reciprocal Rank Fusion**: Combine using RRF formula
    3. **Cross-Encoder Reranking**: Rerank top candidates for precision
    
    Args:
        query: Natural language search query
        top_k: Number of final results to return
        alpha: Weight for semantic vs keyword (0=BM25, 1=vector, 0.5=balanced)
    
    Returns:
        Dict with query, config, retrieval stats, and ranked results
    """
    query = query.strip()
    logger.info(f"Hybrid search | Query: '{query}' | Top-K: {top_k} | Alpha: {alpha}")
    
    timings = {}
    
    # ==========================================================================
    # Stage 1: Encode Query
    # ==========================================================================
    start = time.time()
    q_emb = encode_query(query)
    timings["encode_ms"] = (time.time() - start) * 1000
    logger.info(f"Stage 1 | Query encoding: {timings['encode_ms']:.2f}ms")
    
    # ==========================================================================
    # Stage 2: Parallel Retrieval (Vector + BM25)
    # ==========================================================================
    start = time.time()
    conn = get_connection()
    
    vector_results: list[RetrievalResult] = []
    bm25_results: list[RetrievalResult] = []
    
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Vector search
            logger.info(
                f"Stage 1 | Vector embedding: {(time.time() - start) * 1000}ms | "
            )
            cur.execute(SQL_VECTOR_SEARCH, (q_emb, q_emb, VECTOR_CANDIDATES))
            logger.info(
                f"Stage 1 | Vector embedding: {(time.time() - start) * 1000}ms | "
            )
            for rank, row in enumerate(cur.fetchall(), start=1):
                vector_results.append(RetrievalResult(
                    id=int(row["id"]),
                    score=float(row["score"]),
                    rank=rank,
                ))
            logger.info(
                f"Stage 1 | Vector embedding: {time.time() - start}ms | "
            )
            
            # BM25/FTS search
            cur.execute(SQL_BM25_SEARCH, (query, query, BM25_CANDIDATES))
            for rank, row in enumerate(cur.fetchall(), start=1):
                bm25_results.append(RetrievalResult(
                    id=int(row["id"]),
                    score=float(row["score"]),
                    rank=rank,
                ))
            logger.info(
                f"Stage 1 | Vector embedding: {time.time() - start}ms | "
            )
    finally:
        put_connection(conn)
    
    timings["retrieval_ms"] = (time.time() - start) * 1000
    logger.info(
        f"Stage 2 | Retrieval: {timings['retrieval_ms']:.2f}ms | "
        f"Vector: {len(vector_results)} | BM25: {len(bm25_results)}"
    )
    
    # Log top results from each method
    logger.info(f"Stage 2 | Vector top-5:\n{log_json([
        {'rank': r.rank, 'id': r.id, 'score': round(r.score, 4)}
        for r in vector_results[:5]
    ])}")
    logger.info(f"Stage 2 | BM25 top-5:\n{log_json([
        {'rank': r.rank, 'id': r.id, 'score': round(r.score, 4)}
        for r in bm25_results[:5]
    ])}")
    
    # ==========================================================================
    # Stage 3: Reciprocal Rank Fusion (RRF)
    # ==========================================================================
    start = time.time()
    
    # Normalize scores (optional, for debugging/analysis)
    vector_norm = normalize_scores(vector_results)
    bm25_norm = normalize_scores(bm25_results)
    
    # Compute RRF fusion
    fused_results = compute_rrf(vector_norm, bm25_norm, k=RRF_K, alpha=alpha)
    
    timings["fusion_ms"] = (time.time() - start) * 1000
    logger.info(
        f"Stage 3 | RRF Fusion: {timings['fusion_ms']:.2f}ms | "
        f"Unique candidates: {len(fused_results)}"
    )
    
    # Take top candidates for reranking
    candidates = fused_results[:RERANK_CANDIDATES]
    candidate_ids = [c.id for c in candidates]
    
    logger.info(f"Stage 3 | RRF top-10:\n{log_json([
        {'id': c.id, 'rrf_score': round(c.rrf_score, 6), 'vec_rank': c.vector_rank, 'bm25_rank': c.bm25_rank}
        for c in candidates[:10]
    ])}")
    
    if not candidate_ids:
        return {
            "query": query,
            "config": {"alpha": alpha, "rrf_k": RRF_K},
            "timings": timings,
            "retrieval": {"vector": 0, "bm25": 0, "fused": 0},
            "count": 0,
            "results": [],
        }
    
    # ==========================================================================
    # Stage 4: Fetch Full Movie Data
    # ==========================================================================
    start = time.time()
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(SQL_FETCH_MOVIES, (candidate_ids,))
            movie_rows = {int(row["id"]): dict(row) for row in cur.fetchall()}
    finally:
        put_connection(conn)
    
    timings["fetch_ms"] = (time.time() - start) * 1000
    logger.info(f"Stage 4 | Fetch movies: {timings['fetch_ms']:.2f}ms")
    
    # ==========================================================================
    # Stage 5: Cross-Encoder Reranking
    # ==========================================================================
    start = time.time()
    
    # Prepare rerank pairs (query, document_text)
    rerank_movies = [movie_rows[cid] for cid in candidate_ids if cid in movie_rows]
    pairs = [(query, build_rerank_text(m)) for m in rerank_movies]
    
    # Get reranker and compute scores
    reranker = get_reranker()
    rerank_scores = reranker.predict(pairs)
    
    # Attach scores to movies
    for movie, score in zip(rerank_movies, rerank_scores):
        movie["rerank_score"] = float(score)
    
    # Sort by rerank score
    rerank_movies.sort(key=lambda m: m["rerank_score"], reverse=True)
    
    timings["rerank_ms"] = (time.time() - start) * 1000
    logger.info(f"Stage 5 | Reranking: {timings['rerank_ms']:.2f}ms")
    
    # ==========================================================================
    # Stage 6: Final Results
    # ==========================================================================
    results = rerank_movies[:top_k]
    
    # Add RRF metadata back to results
    rrf_map = {c.id: c for c in candidates}
    for r in results:
        fused = rrf_map.get(r["id"])
        if fused:
            r["rrf_score"] = fused.rrf_score
            r["vector_rank"] = fused.vector_rank
            r["bm25_rank"] = fused.bm25_rank
    
    total_time = sum(timings.values())
    timings["total_ms"] = total_time
    
    logger.info(f"Stage 6 | Final results: {len(results)} | Total: {total_time:.2f}ms")
    logger.info(f"Hybrid search | Final results:\n{log_json([
        {
            'rank': i+1,
            'id': r['id'],
            'title': r['title'],
            'rerank_score': round(r.get('rerank_score', 0), 4),
            'rrf_score': round(r.get('rrf_score', 0), 6),
            'vec_rank': r.get('vector_rank'),
            'bm25_rank': r.get('bm25_rank'),
        }
        for i, r in enumerate(results)
    ])}")
    
    return {
        "query": query,
        "config": {
            "alpha": alpha,
            "rrf_k": RRF_K,
            "vector_candidates": VECTOR_CANDIDATES,
            "bm25_candidates": BM25_CANDIDATES,
            "rerank_candidates": RERANK_CANDIDATES,
        },
        "timings": timings,
        "retrieval": {
            "vector": len(vector_results),
            "bm25": len(bm25_results),
            "fused": len(fused_results),
        },
        "count": len(results),
        "results": results,
    }


# ==============================================================================
# Main (for testing)
# ==============================================================================

if __name__ == "__main__":
    # Test queries
    queries = [
        "mind-bending sci-fi thriller",
        "romantic comedy in Paris",
        "dark superhero movie",
    ]
    
    for q in queries:
        print(f"\n{'='*80}")
        print(f"Query: {q}")
        print("="*80)
        
        response = hybrid_search(q, top_k=5)
        
        print(f"\nConfig: alpha={response['config']['alpha']}, k={response['config']['rrf_k']}")
        print(f"Timings: {response['timings']}")
        print(f"Retrieval: {response['retrieval']}")
        print(f"\nResults ({response['count']}):")
        
        for i, r in enumerate(response["results"], 1):
            print(
                f"  {i}. {r['title']}"
                f" | rerank={r.get('rerank_score', 0):.3f}"
                f" | rrf={r.get('rrf_score', 0):.6f}"
                f" | vec_rank={r.get('vector_rank')}"
                f" | bm25_rank={r.get('bm25_rank')}"
            )
