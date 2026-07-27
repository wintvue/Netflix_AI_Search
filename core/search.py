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
2. Reciprocal Rank Fusion: RRF combines the two rankings by rank position
3. Cross-Encoder Reranking: best-effort precision boost on the top candidates
"""

import copy
import json
import logging
import time
from dataclasses import dataclass

import numpy as np
import psycopg2.extras

from core.cache import TTLCache
from core.config import (
    BM25_CANDIDATES,
    DEFAULT_TOP_K,
    HYBRID_ALPHA,
    RERANK_CANDIDATES,
    RERANK_ENABLED,
    RRF_K,
    SEARCH_CACHE_SIZE,
    SEARCH_CACHE_TTL_SECONDS,
    VECTOR_CANDIDATES,
    get_logger,
)
from core.database import connection
from core.errors import RerankError
from core.model import encode_query, rerank

logger = get_logger(__name__)

_search_cache = TTLCache(SEARCH_CACHE_SIZE, SEARCH_CACHE_TTL_SECONDS)


# ==============================================================================
# SQL Queries
# ==============================================================================

SQL_VECTOR_SEARCH = """
SELECT
    m.id, m.title, m.original_title, m.overview, m.tagline, m.genres,
    m.release_date, m.original_language, m.poster_path,
    m.vote_average, m.vote_count, m.popularity,
    1 - (e.embedding <=> %s) AS score  -- Convert distance to similarity
FROM movie_embeddings_10k e
JOIN movies m ON m.id = e.movie_id
ORDER BY e.embedding <=> %s ASC
LIMIT %s;
"""

SQL_BM25_SEARCH = """
SELECT
    id, title, original_title, overview, tagline, genres,
    release_date, original_language, poster_path,
    vote_average, vote_count, popularity,
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

SQL_SEMANTIC = """
SELECT
    m.id, m.title, m.release_date, m.poster_path,
    m.vote_average, m.vote_count, m.popularity, m.overview,
    (e.embedding <=> %s) AS distance
FROM movie_embeddings_10k e
JOIN movies m ON m.id = e.movie_id
ORDER BY distance ASC
LIMIT %s;
"""


# ==============================================================================
# Type Definitions
# ==============================================================================


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


@dataclass
class RetrievalOutput:
    """Output from the retrieval stage."""

    vector_results: list[RetrievalResult]
    bm25_results: list[RetrievalResult]
    movie_rows: dict[int, dict]
    elapsed_ms: float


@dataclass
class FusionOutput:
    """Output from the RRF fusion stage."""

    fused_results: list[FusedResult]
    candidates: list[FusedResult]
    candidate_ids: list[int]
    elapsed_ms: float


@dataclass
class RerankOutput:
    """Output from the reranking stage."""

    candidate_ids: list[int]
    scores: dict[int, float]
    applied: bool
    elapsed_ms: float


# ==============================================================================
# Helper Functions
# ==============================================================================


def log_json(data: list | dict) -> str:
    """Format data as JSON for structured logging."""
    return json.dumps(data, indent=2, default=str)


def _debug_enabled() -> bool:
    """Whether DEBUG logging is on.

    Guarding the log calls below matters because their arguments serialize the
    candidate set eagerly -- an f-string is evaluated before the logger gets to
    decide the record is beneath the threshold.
    """
    return logger.isEnabledFor(logging.DEBUG)


def compute_rrf(
    vector_results: list[RetrievalResult],
    bm25_results: list[RetrievalResult],
    k: int = RRF_K,
    alpha: float = HYBRID_ALPHA,
) -> list[FusedResult]:
    """
    Compute Reciprocal Rank Fusion (RRF) scores.

    RRF Formula: score = α * 1/(k + rank_vector) + (1-α) * 1/(k + rank_bm25)

    RRF consumes rank positions only, never the raw retrieval scores, which is
    exactly what makes it robust to two retrievers on incomparable scales.

    Args:
        vector_results: Results from vector search (semantic)
        bm25_results: Results from BM25/FTS search (keyword)
        k: RRF constant (default 60, from Microsoft's paper)
        alpha: Weight for vector search (0=pure BM25, 1=pure vector)

    Returns:
        Fused results sorted by RRF score descending
    """
    vector_ranks = {r.id: r.rank for r in vector_results}
    bm25_ranks = {r.id: r.rank for r in bm25_results}

    all_ids = set(vector_ranks) | set(bm25_ranks)

    fused = []
    for doc_id in all_ids:
        vec_rank = vector_ranks.get(doc_id)
        bm25_rank = bm25_ranks.get(doc_id)

        vec_rrf = alpha * (1.0 / (k + vec_rank)) if vec_rank is not None else 0.0
        bm25_rrf = (
            (1 - alpha) * (1.0 / (k + bm25_rank)) if bm25_rank is not None else 0.0
        )

        fused.append(
            FusedResult(
                id=doc_id,
                rrf_score=vec_rrf + bm25_rrf,
                vector_rank=vec_rank,
                bm25_rank=bm25_rank,
            )
        )

    # Sort by score, breaking ties on id so the ordering is deterministic.
    fused.sort(key=lambda x: (-x.rrf_score, x.id))
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
    logger.info("Semantic search | Query: '%s' | Top-K: %d", query, top_k)

    start = time.time()
    q_emb = encode_query(query)
    encode_time = (time.time() - start) * 1000

    start = time.time()
    with connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(SQL_SEMANTIC, (q_emb, top_k))
            rows = cur.fetchall()

    db_time = (time.time() - start) * 1000
    results = [dict(row) for row in rows]

    logger.info(
        "Semantic search | Encode: %.2fms | DB: %.2fms | Results: %d",
        encode_time,
        db_time,
        len(results),
    )
    if _debug_enabled():
        logger.debug(
            "Semantic search | Results:\n%s",
            log_json(
                [
                    {
                        "rank": i + 1,
                        "id": r["id"],
                        "title": r["title"],
                        "distance": round(r["distance"], 4),
                    }
                    for i, r in enumerate(results[:10])
                ]
            ),
        )

    return results


# ==============================================================================
# Hybrid Search Pipeline Stages
# ==============================================================================


def _retrieve_candidates(query: str, query_embedding: np.ndarray) -> RetrievalOutput:
    """
    Stage 2: Parallel retrieval using Vector search and BM25/FTS.

    Fetches candidates from both semantic (vector) and keyword (BM25) indexes,
    storing movie data for later use.
    """
    start = time.time()

    vector_results: list[RetrievalResult] = []
    bm25_results: list[RetrievalResult] = []
    movie_rows: dict[int, dict] = {}

    with connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                SQL_VECTOR_SEARCH,
                (query_embedding, query_embedding, VECTOR_CANDIDATES),
            )
            for rank, row in enumerate(cur.fetchall(), start=1):
                row_dict = dict(row)
                movie_id = int(row_dict["id"])
                vector_results.append(
                    RetrievalResult(
                        id=movie_id, score=float(row_dict["score"]), rank=rank
                    )
                )
                movie_rows[movie_id] = row_dict

            cur.execute(SQL_BM25_SEARCH, (query, query, BM25_CANDIDATES))
            for rank, row in enumerate(cur.fetchall(), start=1):
                row_dict = dict(row)
                movie_id = int(row_dict["id"])
                bm25_results.append(
                    RetrievalResult(
                        id=movie_id, score=float(row_dict["score"]), rank=rank
                    )
                )
                movie_rows.setdefault(movie_id, row_dict)

    elapsed_ms = (time.time() - start) * 1000

    logger.info(
        "Retrieval | %.2fms | Vector: %d | BM25: %d",
        elapsed_ms,
        len(vector_results),
        len(bm25_results),
    )
    if _debug_enabled():
        logger.debug(
            "Retrieval | Vector top-5:\n%s",
            log_json(
                [
                    {"rank": r.rank, "id": r.id, "score": round(r.score, 4)}
                    for r in vector_results[:5]
                ]
            ),
        )
        logger.debug(
            "Retrieval | BM25 top-5:\n%s",
            log_json(
                [
                    {"rank": r.rank, "id": r.id, "score": round(r.score, 4)}
                    for r in bm25_results[:5]
                ]
            ),
        )

    return RetrievalOutput(
        vector_results=vector_results,
        bm25_results=bm25_results,
        movie_rows=movie_rows,
        elapsed_ms=elapsed_ms,
    )


def _fuse_with_rrf(
    vector_results: list[RetrievalResult],
    bm25_results: list[RetrievalResult],
    alpha: float,
) -> FusionOutput:
    """
    Stage 3: Reciprocal Rank Fusion (RRF) to combine rankings.
    """
    start = time.time()

    fused_results = compute_rrf(vector_results, bm25_results, k=RRF_K, alpha=alpha)

    candidates = fused_results[:RERANK_CANDIDATES]
    candidate_ids = [c.id for c in candidates]

    elapsed_ms = (time.time() - start) * 1000

    logger.info(
        "RRF Fusion | %.2fms | Unique candidates: %d", elapsed_ms, len(fused_results)
    )
    if _debug_enabled():
        logger.debug(
            "RRF Fusion | Top-10:\n%s",
            log_json(
                [
                    {
                        "id": c.id,
                        "rrf_score": round(c.rrf_score, 6),
                        "vec_rank": c.vector_rank,
                        "bm25_rank": c.bm25_rank,
                    }
                    for c in candidates[:10]
                ]
            ),
        )

    return FusionOutput(
        fused_results=fused_results,
        candidates=candidates,
        candidate_ids=candidate_ids,
        elapsed_ms=elapsed_ms,
    )


def _rerank_candidates(
    query: str,
    candidate_ids: list[int],
    movie_rows: dict[int, dict],
) -> RerankOutput:
    """
    Stage 4: Cross-encoder reranking for precision.

    Best-effort by design: reranking is an external call, and a search that
    returns RRF-ordered results beats one that returns an error, so any failure
    falls back to the incoming order.
    """
    start = time.time()

    if not RERANK_ENABLED or not candidate_ids:
        return RerankOutput(
            candidate_ids=candidate_ids,
            scores={},
            applied=False,
            elapsed_ms=(time.time() - start) * 1000,
        )

    documents = [build_rerank_text(movie_rows[cid]) for cid in candidate_ids]

    try:
        scores = rerank(query, documents)
    except RerankError as e:
        logger.warning("Rerank | Falling back to RRF order: %s", e)
        return RerankOutput(
            candidate_ids=candidate_ids,
            scores={},
            applied=False,
            elapsed_ms=(time.time() - start) * 1000,
        )

    score_map = dict(zip(candidate_ids, scores))
    reranked_ids = sorted(candidate_ids, key=lambda cid: (-score_map[cid], cid))

    elapsed_ms = (time.time() - start) * 1000
    logger.info("Rerank | %.2fms | Candidates: %d", elapsed_ms, len(candidate_ids))

    return RerankOutput(
        candidate_ids=reranked_ids,
        scores=score_map,
        applied=True,
        elapsed_ms=elapsed_ms,
    )


def _build_final_results(
    candidate_ids: list[int],
    candidates: list[FusedResult],
    movie_rows: dict[int, dict],
    rerank_scores: dict[int, float],
    top_k: int,
) -> list[dict]:
    """
    Stage 5: Build the final result list with fusion and rerank metadata.
    """
    rrf_map = {c.id: c for c in candidates}

    results = []
    for cid in candidate_ids[:top_k]:
        row = movie_rows.get(cid)
        if row is None:
            continue

        # Copy: movie_rows entries are shared with the caller's retrieval
        # output, and cached responses must not alias mutable state.
        result = dict(row)

        # `score` means different things depending on which retriever surfaced
        # the row first, so drop it rather than emit an ambiguous field.
        result.pop("score", None)

        fused = rrf_map.get(cid)
        if fused:
            result["rrf_score"] = fused.rrf_score
            result["vector_rank"] = fused.vector_rank
            result["bm25_rank"] = fused.bm25_rank
        if cid in rerank_scores:
            result["rerank_score"] = rerank_scores[cid]

        results.append(result)

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

    1. **Query Encoding**: Convert query to embedding
    2. **Parallel Retrieval**: Run vector search + BM25/FTS
    3. **RRF Fusion**: Combine rankings using Reciprocal Rank Fusion
    4. **Reranking**: Cross-encoder pass over the top candidates (best-effort)
    5. **Final Results**: Build response with metadata

    Args:
        query: Natural language search query
        top_k: Number of final results to return
        alpha: Weight for semantic vs keyword (0=BM25, 1=vector, 0.5=balanced)

    Returns:
        Dict with query, config, retrieval stats, and ranked results
    """
    query = query.strip()

    cache_key = (query.lower(), top_k, round(alpha, 4))
    cached = _search_cache.get(cache_key)
    if cached is not None:
        logger.info("Hybrid search | Cache hit | Query: '%s'", query)
        return copy.deepcopy(cached)

    logger.info(
        "Hybrid search | Query: '%s' | Top-K: %d | Alpha: %s", query, top_k, alpha
    )

    timings = {}

    # Stage 1: Encode Query
    start = time.time()
    query_embedding = encode_query(query)
    timings["encode_ms"] = (time.time() - start) * 1000

    # Stage 2: Parallel Retrieval
    retrieval = _retrieve_candidates(query, query_embedding)
    timings["retrieval_ms"] = retrieval.elapsed_ms

    # Stage 3: RRF Fusion
    fusion = _fuse_with_rrf(retrieval.vector_results, retrieval.bm25_results, alpha)
    timings["fusion_ms"] = fusion.elapsed_ms

    # Stage 4: Reranking
    reranked = _rerank_candidates(query, fusion.candidate_ids, retrieval.movie_rows)
    timings["rerank_ms"] = reranked.elapsed_ms

    # Stage 5: Build Final Results
    results = _build_final_results(
        reranked.candidate_ids,
        fusion.candidates,
        retrieval.movie_rows,
        reranked.scores,
        top_k,
    )

    timings["total_ms"] = sum(timings.values())

    logger.info(
        "Hybrid search | Final: %d results | Reranked: %s | Total: %.2fms",
        len(results),
        reranked.applied,
        timings["total_ms"],
    )
    if _debug_enabled():
        logger.debug(
            "Hybrid search | Results:\n%s",
            log_json(
                [
                    {
                        "rank": i + 1,
                        "id": r["id"],
                        "title": r["title"],
                        "rerank_score": r.get("rerank_score"),
                        "rrf_score": round(r.get("rrf_score", 0), 6),
                        "vec_rank": r.get("vector_rank"),
                        "bm25_rank": r.get("bm25_rank"),
                    }
                    for i, r in enumerate(results[:10])
                ]
            ),
        )

    response = {
        "query": query,
        "config": {
            "alpha": alpha,
            "rrf_k": RRF_K,
            "vector_candidates": VECTOR_CANDIDATES,
            "bm25_candidates": BM25_CANDIDATES,
            "rerank_candidates": RERANK_CANDIDATES,
            "reranked": reranked.applied,
        },
        "timings": timings,
        "retrieval": {
            "vector": len(retrieval.vector_results),
            "bm25": len(retrieval.bm25_results),
            "fused": len(fusion.fused_results),
        },
        "count": len(results),
        "results": results,
    }

    _search_cache.set(cache_key, response)
    return copy.deepcopy(response)


# ==============================================================================
# Keyword Search
# ==============================================================================


def search_movies(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Keyword-only search over the movie corpus.

    Pure keyword search is hybrid search with alpha=0, so this reuses the same
    pipeline rather than maintaining a second retrieval path.
    """
    logger.info("Keyword search | Query: '%s'", query)
    response = hybrid_search(query, top_k=top_k, alpha=0.0)
    logger.info("Keyword search | Found: %d results", response["count"])
    return response["results"]


def search_cache_info() -> dict:
    """Hit/miss statistics for the search response cache."""
    return _search_cache.info()


def clear_search_cache() -> None:
    """Drop every cached search response."""
    _search_cache.clear()


# ==============================================================================
# Main (for testing)
# ==============================================================================

if __name__ == "__main__":
    from core.config import setup_logging
    from core.database import close_pool, create_db_pool

    setup_logging()
    create_db_pool()

    try:
        for q in [
            "mind-bending sci-fi thriller",
            "romantic comedy in Paris",
            "dark superhero movie",
        ]:
            print(f"\n{'=' * 80}")
            print(f"Query: {q}")
            print("=" * 80)

            response = hybrid_search(q, top_k=5)

            print(
                f"\nConfig: alpha={response['config']['alpha']}, "
                f"k={response['config']['rrf_k']}, "
                f"reranked={response['config']['reranked']}"
            )
            print(f"Timings: {response['timings']}")
            print(f"Retrieval: {response['retrieval']}")
            print(f"\nResults ({response['count']}):")

            for i, r in enumerate(response["results"], 1):
                rerank_score = r.get("rerank_score")
                rerank_display = (
                    f"{rerank_score:.3f}" if rerank_score is not None else "n/a"
                )
                print(
                    f"  {i}. {r['title']}"
                    f" | rerank={rerank_display}"
                    f" | rrf={r.get('rrf_score', 0):.6f}"
                    f" | vec_rank={r.get('vector_rank')}"
                    f" | bm25_rank={r.get('bm25_rank')}"
                )
    finally:
        close_pool()
