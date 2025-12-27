#!/usr/bin/env python3
"""Search functions for movies."""

import json
import re
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

SQL_GET_ANCHOR_EMB = """
SELECT e.embedding
FROM movies m
JOIN movie_embeddings e ON e.movie_id = m.id
WHERE lower(m.title) = lower(%s)
LIMIT 1;
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
        f"Genres: {row.get('genres') or ''}",
        f"Tagline: {row.get('tagline') or ''}",
        f"Overview: {row.get('overview') or ''}",
    ]
    return "\n".join([p for p in parts if not p.endswith(": ")])


def extract_anchor_title(q: str) -> str | None:
    """Extract anchor title from query like 'like Inception'.
    
    Very simple rule: looks for "like <title words>" pattern.
    """
    m = re.search(r"\blike\s+([a-z0-9'\":\- ]{2,60})", q.lower())
    if not m:
        return None
    # Cut off common trailing constraint words
    anchor = m.group(1)
    anchor = re.split(r"\bbut\b|\bwith\b|\bwithout\b|\band\b|\bor\b", anchor)[0].strip()
    # Normalize extra quotes
    anchor = anchor.strip("'\" ")
    return anchor if anchor else None


def genre_has(genres: str | None, needle: str) -> bool:
    """Check if genres string contains the needle (case-insensitive)."""
    if not genres:
        return False
    return needle.lower() in genres.lower()


def hybrid_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Search movies using hybrid vector + FTS search with reranking.
    
    This combines:
    1. Anchor embedding detection (e.g., "like Inception")
    2. Vector search using anchor embedding (main) → top 200
    3. Full-text search (FTS) for keyword matches (secondary) → top 60
    4. Cross-encoder reranking with genre-based score adjustments
    
    Args:
        query: Natural language query describing the movies you want.
        top_k: Number of results to return.
        
    Returns:
        List of movies ranked by final_score.
    """
    query = query.strip()
    logger.info(f"Hybrid search | Query: '{query}' | Top-K: {top_k}")
    
    # 1) Encode query for vector retrieval
    start = time.time()
    q_emb = encode_query(query)
    encode_time = (time.time() - start) * 1000
    logger.info(f"Hybrid search | Encoding took: {encode_time:.2f}ms")

    # Extract anchor title (e.g., "inception" from "like inception")
    anchor = extract_anchor_title(query)
    if anchor:
        logger.info(f"Hybrid search | Detected anchor title: '{anchor}'")
    
    use_emb = q_emb  # Default to query embedding

    # 2) Candidate retrieval
    start = time.time()
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # If anchor exists and we have it in DB, use anchor embedding for retrieval
            if anchor:
                cur.execute(SQL_GET_ANCHOR_EMB, (anchor,))
                row = cur.fetchone()
                if row and row.get("embedding") is not None:
                    use_emb = row["embedding"]
                    logger.info(f"Hybrid search | Using anchor embedding for '{anchor}'")
                else:
                    logger.info(f"Hybrid search | Anchor '{anchor}' not found in DB, using query embedding")

            # Vector candidates (main retrieval)
            cur.execute(SQL_VEC, (use_emb, VEC_CANDIDATES))
            vec_rows = cur.fetchall()
            logger.info(f"Hybrid search | Vector candidates: {len(vec_rows)}")

            # FTS candidates (secondary; helps exact matches, not dominating)
            cur.execute(SQL_FTS, (query, query, FTS_CANDIDATES))
            fts_rows = cur.fetchall()
            logger.info(f"Hybrid search | FTS candidates: {len(fts_rows)}")

            # 3) Merge IDs (vector first, then FTS)
            cand_ids = []
            seen = set()
            for r in vec_rows:
                mid = int(r["id"])
                if mid not in seen:
                    seen.add(mid)
                    cand_ids.append(mid)
            for r in fts_rows:
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
    rerank_scores = reranker.predict(pairs)
    rerank_time = (time.time() - start) * 1000
    logger.info(f"Hybrid search | Reranking took: {rerank_time:.2f}ms")

    # 6) Calculate final scores with genre-based adjustments
    wants_darker = "darker" in query.lower() or "dark" in query.lower()
    
    results = [dict(r) for r in rows]
    for r, s in zip(results, rerank_scores):
        score = float(s)
        
        # Small boosts for "Inception-ish" genres (sci-fi thrillers)
        if genre_has(r.get("genres"), "Science Fiction"):
            score += 0.25
        if genre_has(r.get("genres"), "Thriller"):
            score += 0.15
        
        # Penalize clearly wrong vibe for "darker" queries
        if wants_darker:
            if genre_has(r.get("genres"), "Animation"):
                score -= 1.0
            if genre_has(r.get("genres"), "Family"):
                score -= 1.0
            if genre_has(r.get("genres"), "Comedy"):
                score -= 0.6
        
        r["final_score"] = score

    results.sort(key=lambda r: r["final_score"], reverse=True)

    # 7) Return top K
    results = results[:top_k]
    
    # Log final results in JSON format
    logger.info(f"Hybrid search | Final results:\n{log_json([
        {'rank': i+1, 'id': r['id'], 'title': r['title'], 'genres': r.get('genres'), 'final_score': round(r['final_score'], 4)}
        for i, r in enumerate(results)
    ])}")
    
    return results


if __name__ == "__main__":
    q = "mind-bending sci-fi like inception but darker"
    results = hybrid_search(q, 10)
    for r in results:
        print(f"{r['final_score']:.3f} | {r['title']} | {r.get('genres', '')}")
