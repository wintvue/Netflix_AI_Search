#!/usr/bin/env python3
"""Search functions for movies."""

import json
import time
from pathlib import Path
from typing import TypedDict

import psycopg2.extras

from core.config import (
    DEFAULT_TOP_K,
    FTS_CANDIDATES,
    RERANK_CANDIDATES,
    VEC_CANDIDATES,
    get_logger,
)
from core.database import get_connection
from core.model import encode_query, get_reranker

logger = get_logger(__name__)


def log_json(data: list | dict) -> str:
    """Format data as JSON string for logging."""
    return json.dumps(data, indent=2, default=str)

# SQL Queries
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

SQL_FTS = """
SELECT
  id,
  ts_rank(
    to_tsvector('english',
      coalesce(title,'') || ' ' ||
      coalesce(original_title,'') || ' ' ||
      coalesce(overview,'') || ' ' ||
      coalesce(tagline,'')
    ),
    websearch_to_tsquery('english', %s)
  ) AS fts_score
FROM movies
WHERE to_tsvector('english',
      coalesce(title,'') || ' ' ||
      coalesce(original_title,'') || ' ' ||
      coalesce(overview,'') || ' ' ||
      coalesce(tagline,'')
    ) @@ websearch_to_tsquery('english', %s)
ORDER BY fts_score DESC
LIMIT %s;
"""

SQL_VEC = """
SELECT
  e.movie_id AS id,
  (e.embedding <=> %s) AS distance
FROM movie_embeddings e
ORDER BY distance ASC
LIMIT %s;
"""

SQL_FETCH = """
SELECT
  id, title, original_title, overview, tagline, genres,
  release_date, original_language, poster_path,
  vote_average, vote_count, popularity
FROM movies
WHERE id = ANY(%s);
"""


class Movie(TypedDict):
    id: int
    title: str
    description: str


# Load movies data once at module import (for keyword search)
DATA_PATH = Path(__file__).parent.parent / "data" / "movies.json"
with open(DATA_PATH) as f:
    _MOVIES_DATA = json.load(f)


def search_movies(query: str) -> list[Movie]:
    """Search movies by title keyword.
    
    Args:
        query: Search query string to match against movie titles.
        
    Returns:
        List of movies whose titles contain the query (case-insensitive).
    """
    logger.info(f"Keyword search | Query: '{query}'")
    
    query_lower = query.lower()
    results: list[Movie] = []

    for movie in _MOVIES_DATA["movies"]:
        if query_lower in movie["title"].lower():
            results.append(movie)

    logger.info(f"Keyword search | Found: {len(results)} results")
    return results


def semantic_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Search movies using semantic similarity (vector search only).
    
    Args:
        query: Natural language query describing the movies you want.
        top_k: Number of results to return.
        
    Returns:
        List of movies ranked by semantic similarity to the query.
    """
    logger.info(f"Semantic search | Query: '{query}' | Top-K: {top_k}")
    
    # Encode query
    start = time.time()
    q_emb = encode_query(query)
    encode_time = (time.time() - start) * 1000
    logger.info(f"Semantic search | Encoding took: {encode_time:.2f}ms")

    # Database search
    start = time.time()
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(SQL_SEMANTIC, (q_emb, top_k))
            rows = cur.fetchall()
    finally:
        conn.close()
    
    db_time = (time.time() - start) * 1000
    results = [dict(row) for row in rows]
    
    logger.info(f"Semantic search | DB query took: {db_time:.2f}ms | Found: {len(results)} results")
    
    # Log results in JSON format
    logger.info(f"Semantic search | Results:\n{log_json([
        {'rank': i+1, 'id': r['id'], 'title': r['title'], 'distance': round(r['distance'], 4)}
        for i, r in enumerate(results)
    ])}")
    
    return results


def build_rerank_text(row: dict) -> str:
    """Build text for reranking from a movie row."""
    parts = [
        f"Title: {row.get('title') or ''}",
        f"Original title: {row.get('original_title') or ''}",
        f"Genres: {row.get('genres') or ''}",
        f"Tagline: {row.get('tagline') or ''}",
        f"Overview: {row.get('overview') or ''}",
    ]
    return "\n".join([p for p in parts if not p.endswith(": ")])


def hybrid_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Search movies using hybrid FTS + vector search with reranking.
    
    This combines:
    1. Full-text search (FTS) for keyword matches
    2. Vector search for semantic matches
    3. Cross-encoder reranking for better final ranking
    
    Args:
        query: Natural language query describing the movies you want.
        top_k: Number of results to return.
        
    Returns:
        List of movies ranked by reranker score.
    """
    query = query.strip()
    logger.info(f"Hybrid search | Query: '{query}' | Top-K: {top_k}")
    
    # 1) Encode query for vector retrieval
    start = time.time()
    q_emb = encode_query(query)
    encode_time = (time.time() - start) * 1000
    logger.info(f"Hybrid search | Encoding took: {encode_time:.2f}ms")

    # 2) Candidate retrieval (FTS + Vector)
    start = time.time()
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # FTS candidates
            cur.execute(SQL_FTS, (query, query, FTS_CANDIDATES))
            fts_rows = cur.fetchall()
            logger.info(f"Hybrid search | FTS candidates: {len(fts_rows)}")

            # Vector candidates
            cur.execute(SQL_VEC, (q_emb, VEC_CANDIDATES))
            vec_rows = cur.fetchall()
            logger.info(f"Hybrid search | Vector candidates: {len(vec_rows)}")

            # 3) Merge IDs (deduplicate, preserve order)
            cand_ids = []
            seen = set()
            for r in fts_rows:
                mid = int(r["id"])
                if mid not in seen:
                    seen.add(mid)
                    cand_ids.append(mid)
            for r in vec_rows:
                mid = int(r["id"])
                if mid not in seen:
                    seen.add(mid)
                    cand_ids.append(mid)

            cand_ids = cand_ids[:RERANK_CANDIDATES]
            logger.info(f"Hybrid search | Merged candidates: {len(cand_ids)}")
            
            if not cand_ids:
                return []

            # 4) Fetch full rows
            cur.execute(SQL_FETCH, (cand_ids,))
            rows = cur.fetchall()
    finally:
        conn.close()
    
    retrieval_time = (time.time() - start) * 1000
    logger.info(f"Hybrid search | Retrieval took: {retrieval_time:.2f}ms")

    # 5) Rerank with cross-encoder
    start = time.time()
    reranker = get_reranker()
    pairs = [(query, build_rerank_text(dict(r))) for r in rows]
    scores = reranker.predict(pairs)
    rerank_time = (time.time() - start) * 1000
    logger.info(f"Hybrid search | Reranking took: {rerank_time:.2f}ms")

    # 6) Sort by rerank score desc
    results = [dict(r) for r in rows]
    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)

    results.sort(key=lambda r: r["rerank_score"], reverse=True)

    # 7) Return top K
    results = results[:top_k]
    
    # Log final results in JSON format
    logger.info(f"Hybrid search | Final results:\n{log_json([
        {'rank': i+1, 'id': r['id'], 'title': r['title'], 'genres': r.get('genres'), 'rerank_score': round(r['rerank_score'], 4)}
        for i, r in enumerate(results)
    ])}")
    
    return results


if __name__ == "__main__":
    q = "mind-bending sci-fi like inception but darker"
    results = hybrid_search(q, 10)
    for r in results:
        print(f"{r['rerank_score']:.3f} | {r['title']}")
