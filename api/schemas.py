#!/usr/bin/env python3
"""Response schemas for the search API.

These give the OpenAPI document a real contract and validate what the service
emits, instead of every route returning an undocumented bare dict.
"""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class MovieResult(BaseModel):
    """A single ranked movie."""

    # The retrieval SQL selects a wide row and the pipeline attaches scoring
    # metadata; allow extras so adding a column does not silently drop it.
    model_config = ConfigDict(extra="allow")

    id: int
    title: str | None = None
    original_title: str | None = None
    overview: str | None = None
    tagline: str | None = None
    genres: str | None = None
    release_date: date | None = None
    original_language: str | None = None
    poster_path: str | None = None
    vote_average: float | None = None
    vote_count: int | None = None
    popularity: float | None = None

    rrf_score: float | None = Field(
        None, description="Reciprocal Rank Fusion score (higher is better)"
    )
    vector_rank: int | None = Field(
        None, description="Rank in the vector retrieval pool, if it appeared there"
    )
    bm25_rank: int | None = Field(
        None, description="Rank in the BM25/FTS pool, if it appeared there"
    )
    rerank_score: float | None = Field(
        None, description="Cross-encoder relevance score; absent if reranking was skipped"
    )


class SearchTimings(BaseModel):
    """Per-stage wall-clock timings in milliseconds."""

    model_config = ConfigDict(extra="allow")

    encode_ms: float | None = None
    retrieval_ms: float | None = None
    fusion_ms: float | None = None
    rerank_ms: float | None = None
    total_ms: float | None = None


class SearchConfig(BaseModel):
    """The effective pipeline configuration for this request."""

    model_config = ConfigDict(extra="allow")

    alpha: float
    rrf_k: int
    vector_candidates: int | None = None
    bm25_candidates: int | None = None
    rerank_candidates: int | None = None
    reranked: bool | None = Field(
        None, description="Whether the cross-encoder rerank stage actually ran"
    )


class RetrievalStats(BaseModel):
    """How many candidates each retrieval method contributed."""

    vector: int
    bm25: int
    fused: int


class MovieExplanation(BaseModel):
    """LLM explanation of why one movie matches the query."""

    id: int
    title: str
    explanation: str


class AIMetadata(BaseModel):
    """Provenance for a generated overview."""

    model_config = ConfigDict(extra="allow")

    model: str
    generation_time_ms: float
    status: str
    error: str | None = None
    eval_count: int | None = None
    prompt_eval_count: int | None = None


class AIOverview(BaseModel):
    """LLM-generated summary of a result set."""

    overview: str
    movie_explanations: list[MovieExplanation] = []
    ai_metadata: AIMetadata


class SearchResponse(BaseModel):
    """Full hybrid search response."""

    query: str
    config: SearchConfig
    timings: SearchTimings
    retrieval: RetrievalStats
    count: int
    results: list[MovieResult]
    ai_overview: AIOverview | None = None


class KeywordSearchResponse(BaseModel):
    """Keyword-only search response."""

    query: str
    count: int
    results: list[MovieResult]


class SemanticSearchResponse(BaseModel):
    """Vector-only search response."""

    query: str
    count: int
    results: list[MovieResult]


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str


class ReadinessResponse(BaseModel):
    """Readiness probe response, reflecting real dependency state."""

    status: str
    database: bool
    embedding_client: bool
    rerank_enabled: bool
    checks: dict[str, str] = {}


class ErrorResponse(BaseModel):
    """Uniform error body."""

    error: str
    detail: str | None = None
    request_id: str | None = None
