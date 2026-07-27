"""Tests for the embedding client. No network."""

import numpy as np
import pytest

import core.model as model
from core.errors import ConfigurationError


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
