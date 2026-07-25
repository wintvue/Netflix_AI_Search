"""Tests for RRF fusion and the reranking stage."""

import pytest

from core.errors import RerankError
from core.search import (
    RetrievalResult,
    _build_final_results,
    _rerank_candidates,
    build_rerank_text,
    compute_rrf,
)


def rr(doc_id: int, rank: int, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(id=doc_id, score=score, rank=rank)


class TestComputeRRF:
    def test_alpha_zero_is_pure_keyword(self):
        vector = [rr(1, 1), rr(2, 2)]
        bm25 = [rr(2, 1), rr(3, 2)]

        fused = compute_rrf(vector, bm25, k=60, alpha=0.0)

        # Only the BM25 ranking may influence the order.
        assert [f.id for f in fused] == [2, 3, 1]
        assert next(f for f in fused if f.id == 1).rrf_score == 0.0

    def test_alpha_one_is_pure_semantic(self):
        vector = [rr(1, 1), rr(2, 2)]
        bm25 = [rr(3, 1), rr(2, 2)]

        fused = compute_rrf(vector, bm25, k=60, alpha=1.0)

        assert [f.id for f in fused] == [1, 2, 3]
        assert next(f for f in fused if f.id == 3).rrf_score == 0.0

    def test_documents_in_one_list_only_still_score(self):
        fused = compute_rrf([rr(1, 1)], [rr(2, 1)], k=60, alpha=0.5)

        by_id = {f.id: f for f in fused}
        assert by_id[1].rrf_score > 0
        assert by_id[2].rrf_score > 0
        assert by_id[1].vector_rank == 1 and by_id[1].bm25_rank is None
        assert by_id[2].bm25_rank == 1 and by_id[2].vector_rank is None

    def test_document_in_both_lists_outranks_singletons(self):
        vector = [rr(1, 1), rr(2, 2)]
        bm25 = [rr(2, 1), rr(3, 2)]

        fused = compute_rrf(vector, bm25, k=60, alpha=0.5)

        # id 2 appears in both pools, so it accumulates from both terms.
        assert fused[0].id == 2

    def test_ties_break_deterministically_on_id(self):
        # Identical ranks in both pools produce identical scores.
        vector = [rr(5, 1), rr(3, 1), rr(9, 1)]
        bm25 = []

        fused = compute_rrf(vector, bm25, k=60, alpha=1.0)

        assert [f.id for f in fused] == [3, 5, 9]

    def test_empty_inputs(self):
        assert compute_rrf([], [], k=60, alpha=0.5) == []

    def test_rrf_uses_rank_not_score(self):
        """Raw retrieval scores must not affect the result at all."""
        low = [rr(1, 1, score=0.001), rr(2, 2, score=0.0005)]
        high = [rr(1, 1, score=999.0), rr(2, 2, score=500.0)]

        assert compute_rrf(low, [], alpha=1.0) == compute_rrf(high, [], alpha=1.0)


class TestBuildRerankText:
    def test_includes_populated_fields(self):
        text = build_rerank_text(
            {
                "title": "Inception",
                "genres": "Sci-Fi",
                "tagline": "Your mind is the scene of the crime",
                "overview": "A thief who enters dreams.",
            }
        )

        assert "Title: Inception" in text
        assert "Genres: Sci-Fi" in text
        assert "Overview: A thief who enters dreams." in text

    def test_omits_empty_and_missing_fields(self):
        text = build_rerank_text({"title": "Inception", "tagline": "", "genres": None})

        assert text == "Title: Inception"
        assert "Tagline" not in text
        assert "Genres" not in text


class TestRerankStage:
    movie_rows = {
        1: {"id": 1, "title": "A"},
        2: {"id": 2, "title": "B"},
        3: {"id": 3, "title": "C"},
    }

    def test_reorders_by_rerank_score(self, monkeypatch):
        monkeypatch.setattr("core.search.RERANK_ENABLED", True)
        monkeypatch.setattr("core.search.rerank", lambda q, docs: [0.1, 0.9, 0.5])

        out = _rerank_candidates("query", [1, 2, 3], self.movie_rows)

        assert out.applied is True
        assert out.candidate_ids == [2, 3, 1]
        assert out.scores == {1: 0.1, 2: 0.9, 3: 0.5}

    def test_failure_falls_back_to_input_order(self, monkeypatch):
        """A rerank outage must degrade to RRF order, never fail the search."""
        monkeypatch.setattr("core.search.RERANK_ENABLED", True)

        def boom(query, documents):
            raise RerankError("upstream is down")

        monkeypatch.setattr("core.search.rerank", boom)

        out = _rerank_candidates("query", [1, 2, 3], self.movie_rows)

        assert out.applied is False
        assert out.candidate_ids == [1, 2, 3]
        assert out.scores == {}

    def test_disabled_skips_the_stage(self, monkeypatch):
        monkeypatch.setattr("core.search.RERANK_ENABLED", False)

        def fail(query, documents):
            raise AssertionError("reranker must not be called when disabled")

        monkeypatch.setattr("core.search.rerank", fail)

        out = _rerank_candidates("query", [1, 2, 3], self.movie_rows)

        assert out.applied is False
        assert out.candidate_ids == [1, 2, 3]

    def test_no_candidates(self, monkeypatch):
        monkeypatch.setattr("core.search.RERANK_ENABLED", True)
        out = _rerank_candidates("query", [], {})
        assert out.candidate_ids == []
        assert out.applied is False


class TestBuildFinalResults:
    def test_attaches_metadata_and_respects_top_k(self):
        fused = compute_rrf([rr(1, 1), rr(2, 2)], [rr(2, 1)], k=60, alpha=0.5)
        rows = {
            1: {"id": 1, "title": "A", "score": 0.9},
            2: {"id": 2, "title": "B", "score": 0.8},
        }

        results = _build_final_results([2, 1], fused, rows, {2: 0.99}, top_k=1)

        assert len(results) == 1
        assert results[0]["id"] == 2
        assert results[0]["rerank_score"] == 0.99
        assert "rrf_score" in results[0]
        # The ambiguous per-retriever `score` column is dropped.
        assert "score" not in results[0]

    def test_does_not_mutate_the_source_rows(self):
        """Rows are shared with the retrieval output and the response cache."""
        fused = compute_rrf([rr(1, 1)], [], k=60, alpha=1.0)
        rows = {1: {"id": 1, "title": "A"}}

        _build_final_results([1], fused, rows, {1: 0.5}, top_k=10)

        assert rows[1] == {"id": 1, "title": "A"}

    def test_omits_rerank_score_when_stage_was_skipped(self):
        fused = compute_rrf([rr(1, 1)], [], k=60, alpha=1.0)
        rows = {1: {"id": 1, "title": "A"}}

        results = _build_final_results([1], fused, rows, {}, top_k=10)

        assert "rerank_score" not in results[0]

    def test_skips_ids_with_no_row(self):
        fused = compute_rrf([rr(1, 1)], [], k=60, alpha=1.0)
        results = _build_final_results([1, 99], fused, {1: {"id": 1}}, {}, top_k=10)
        assert [r["id"] for r in results] == [1]


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_rrf_scores_are_bounded_by_alpha(alpha):
    fused = compute_rrf([rr(1, 1)], [rr(1, 1)], k=60, alpha=alpha)
    # Both terms hit rank 1, so the score is exactly 1/(k+1).
    assert fused[0].rrf_score == pytest.approx(1.0 / 61)
