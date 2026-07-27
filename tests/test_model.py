"""Tests for the embedding client. No network."""

import numpy as np
import pytest

import core.model as model
from core.errors import ConfigurationError, EmbeddingError


@pytest.fixture(autouse=True)
def reset_client(monkeypatch):
    monkeypatch.setattr(model, "_client", None)
    yield


class TestGetHuggingFaceClient:
    def test_missing_token_raises_an_actionable_error(self, monkeypatch):
        monkeypatch.setattr(model, "HF_TOKEN", None)

        with pytest.raises(ConfigurationError, match="HF_TOKEN"):
            model.get_huggingface_client()

    def test_is_client_loaded_reflects_state(self, monkeypatch):
        assert model.is_client_loaded() is False
        monkeypatch.setattr(model, "_client", object())
        assert model.is_client_loaded() is True


class TestEncodeQuery:
    def test_resolves_the_client_instead_of_reading_a_global(self, monkeypatch):
        """Regression: encode_query used to dereference an unset global.

        Any entry point that skips the API lifespan hook (CLI, scripts,
        __main__ blocks) hit AttributeError on None.
        """
        monkeypatch.setattr(model, "HF_TOKEN", None)

        with pytest.raises(ConfigurationError):
            model.encode_query("mind-bending sci-fi")

    def test_returns_a_float32_vector(self, monkeypatch):
        class Client:
            def feature_extraction(self, text, **kwargs):
                return [0.1, 0.2, 0.3]

        monkeypatch.setattr(model, "get_huggingface_client", lambda: Client())

        embedding = model.encode_query("sci-fi")

        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert embedding.tolist() == pytest.approx([0.1, 0.2, 0.3])

    def test_requests_the_configured_model(self, monkeypatch):
        seen = {}

        class Client:
            def feature_extraction(self, text, **kwargs):
                seen.update(kwargs, text=text)
                return [0.0]

        monkeypatch.setattr(model, "get_huggingface_client", lambda: Client())

        model.encode_query("sci-fi")

        assert seen["model"] == model.EMBED_MODEL_NAME
        assert seen["normalize"] is True
        assert seen["text"] == "sci-fi"

    def test_rejects_an_empty_query(self, monkeypatch):
        monkeypatch.setattr(model, "get_huggingface_client", lambda: object())

        with pytest.raises(EmbeddingError):
            model.encode_query("   ")


class TestRetryAndErrorWrapping:
    def test_upstream_failures_are_wrapped(self, monkeypatch):
        class Boom:
            def feature_extraction(self, *a, **kw):
                raise ConnectionError("upstream refused")

        monkeypatch.setattr(model, "get_huggingface_client", lambda: Boom())
        monkeypatch.setattr(model, "_RETRY_BACKOFF_SECONDS", 0)

        with pytest.raises(EmbeddingError, match="Failed to embed query"):
            model.encode_query("sci-fi")

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
        """One retry only: more would outlive the caller's timeout budget."""
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

    def test_client_is_constructed_with_a_timeout(self, monkeypatch):
        captured = {}

        class FakeInferenceClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(model, "InferenceClient", FakeInferenceClient)
        monkeypatch.setattr(model, "HF_TOKEN", "token")

        model.get_huggingface_client()

        assert captured["timeout"] == model.HF_TIMEOUT_SECONDS
