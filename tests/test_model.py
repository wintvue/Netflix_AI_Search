"""Tests for embedding and rerank response handling. No network."""

import numpy as np
import pytest

import core.model as model
from core.errors import EmbeddingError, RerankError


@pytest.fixture(autouse=True)
def clear_embedding_cache():
    model._encode_query_cached.cache_clear()
    yield
    model._encode_query_cached.cache_clear()


class TestEncodeQuery:
    def test_rejects_an_empty_query(self):
        with pytest.raises(EmbeddingError):
            model.encode_query("   ")

    def test_wraps_upstream_failures(self, monkeypatch):
        class Boom:
            def feature_extraction(self, *a, **kw):
                raise ConnectionError("upstream refused")

        monkeypatch.setattr(model, "get_huggingface_client", lambda: Boom())
        monkeypatch.setattr(model, "_RETRY_BACKOFF_SECONDS", 0)

        with pytest.raises(EmbeddingError, match="Failed to embed query"):
            model.encode_query("sci-fi")

    def test_result_is_cached_and_normalized(self, monkeypatch):
        calls = []

        class Client:
            def feature_extraction(self, text, **kw):
                calls.append(text)
                return [0.1, 0.2, 0.3]

        monkeypatch.setattr(model, "get_huggingface_client", lambda: Client())

        first = model.encode_query("Sci-Fi")
        second = model.encode_query("  sci-fi  ")

        # Case and surrounding whitespace collapse onto one cache entry.
        assert calls == ["sci-fi"]
        assert np.array_equal(first, second)
        assert first.dtype == np.float32

    def test_cached_arrays_are_read_only(self, monkeypatch):
        """A caller must not be able to corrupt a shared cache entry."""

        class Client:
            def feature_extraction(self, text, **kw):
                return [0.1, 0.2]

        monkeypatch.setattr(model, "get_huggingface_client", lambda: Client())

        embedding = model.encode_query("q")

        with pytest.raises(ValueError):
            embedding[0] = 99.0

    def test_retries_once_before_succeeding(self, monkeypatch):
        attempts = []

        class Flaky:
            def feature_extraction(self, text, **kw):
                attempts.append(1)
                if len(attempts) == 1:
                    raise ConnectionError("transient")
                return [0.5, 0.5]

        monkeypatch.setattr(model, "get_huggingface_client", lambda: Flaky())
        monkeypatch.setattr(model, "_RETRY_BACKOFF_SECONDS", 0)

        result = model.encode_query("q")

        assert len(attempts) == 2
        assert result.tolist() == [0.5, 0.5]

    def test_gives_up_after_the_retry_budget(self, monkeypatch):
        attempts = []

        class AlwaysDown:
            def feature_extraction(self, text, **kw):
                attempts.append(1)
                raise ConnectionError("down")

        monkeypatch.setattr(model, "get_huggingface_client", lambda: AlwaysDown())
        monkeypatch.setattr(model, "_RETRY_BACKOFF_SECONDS", 0)

        with pytest.raises(EmbeddingError):
            model.encode_query("q")

        assert len(attempts) == model._MAX_ATTEMPTS


class TestParseRerankResponse:
    def test_single_label_per_pair(self):
        body = [[{"label": "LABEL_0", "score": 0.9}], [{"label": "LABEL_0", "score": 0.2}]]
        assert model._parse_rerank_response(body, expected=2) == [0.9, 0.2]

    def test_two_label_model_takes_the_positive_class(self):
        body = [
            [
                {"label": "LABEL_0", "score": 0.3},
                {"label": "LABEL_1", "score": 0.7},
            ]
        ]
        assert model._parse_rerank_response(body, expected=1) == [0.7]

    def test_unwrapped_single_prediction(self):
        body = [{"label": "LABEL_0", "score": 0.55}]
        assert model._parse_rerank_response(body, expected=1) == [0.55]

    def test_wrong_length_is_an_error(self):
        with pytest.raises(RerankError, match="Unexpected rerank response shape"):
            model._parse_rerank_response([[{"label": "a", "score": 1.0}]], expected=3)

    def test_non_list_body_is_an_error(self):
        with pytest.raises(RerankError):
            model._parse_rerank_response({"error": "model loading"}, expected=1)

    def test_empty_prediction_is_an_error(self):
        with pytest.raises(RerankError, match="empty prediction"):
            model._parse_rerank_response([[]], expected=1)


class TestRerank:
    def test_empty_documents_short_circuits(self):
        assert model.rerank("q", []) == []

    def test_unknown_backend_raises(self, monkeypatch):
        monkeypatch.setattr(model, "RERANK_BACKEND", "nonsense")
        with pytest.raises(RerankError, match="Unknown RERANK_BACKEND"):
            model.rerank("q", ["a"])

    def test_score_count_mismatch_is_detected(self, monkeypatch):
        monkeypatch.setattr(model, "RERANK_BACKEND", "hf")
        monkeypatch.setattr(model, "_rerank_hosted", lambda q, docs: [0.5])

        with pytest.raises(RerankError, match="returned 1 scores for 2 documents"):
            model.rerank("q", ["a", "b"])

    def test_wraps_unexpected_failures(self, monkeypatch):
        monkeypatch.setattr(model, "RERANK_BACKEND", "hf")

        def boom(q, docs):
            raise ConnectionError("router down")

        monkeypatch.setattr(model, "_rerank_hosted", boom)

        with pytest.raises(RerankError, match="Reranking failed"):
            model.rerank("q", ["a"])
