"""Tests for AI overview prompt building and failure handling."""

import json

import pytest

from core.ai_overview import (
    _strip_markdown_fences,
    format_movies_context,
    generate_ai_overview,
)

MOVIES = [
    {
        "id": i,
        "title": f"Movie {i}",
        "tagline": f"Tagline {i}",
        "genres": "Sci-Fi",
        "overview": f"Overview {i}",
        "release_date": "2010-07-16",
        "vote_average": 8.0,
    }
    for i in range(1, 21)
]


class FakeClient:
    """Stands in for ollama.Client."""

    def __init__(self, content=None, error=None):
        self._content = content
        self._error = error
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return {"message": {"content": self._content}, "eval_count": 42}


@pytest.fixture
def fake_client(monkeypatch):
    def install(client):
        monkeypatch.setattr("core.ai_overview.get_ollama_client", lambda: client)
        return client

    return install


class TestStripMarkdownFences:
    def test_bare_json_untouched(self):
        assert _strip_markdown_fences('{"a": 1}') == '{"a": 1}'

    def test_json_language_fence(self):
        assert _strip_markdown_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_plain_fence(self):
        assert _strip_markdown_fences('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_surrounding_whitespace(self):
        assert _strip_markdown_fences('\n\n  ```json\n{"a": 1}\n```  \n') == '{"a": 1}'


class TestFormatMoviesContext:
    def test_caps_the_number_of_movies(self):
        context = format_movies_context("query", MOVIES, max_movies=3)

        assert "Movie 3" in context
        assert "Movie 4" not in context

    def test_includes_query_and_populated_fields(self):
        context = format_movies_context("space opera", MOVIES[:1], max_movies=5)

        assert "User Query: space opera" in context
        assert "Title: Movie 1" in context
        assert "Tagline: Tagline 1" in context
        assert "Rating: 8.0/10" in context

    def test_omits_missing_fields(self):
        context = format_movies_context(
            "q", [{"id": 1, "title": "Solo"}], max_movies=5
        )

        assert "Title: Solo" in context
        assert "Tagline" not in context
        assert "Rating" not in context


class TestGenerateAIOverview:
    def test_success(self, fake_client):
        payload = {
            "overview": "These are sci-fi films.",
            "movie_explanations": [
                {"id": 1, "title": "Movie 1", "explanation": "Because."}
            ],
        }
        fake_client(FakeClient(content=json.dumps(payload)))

        result = generate_ai_overview("sci-fi", MOVIES[:2])

        assert result["overview"] == "These are sci-fi films."
        assert result["ai_metadata"]["status"] == "success"
        assert len(result["movie_explanations"]) == 1

    def test_strips_fences_before_parsing(self, fake_client):
        fake_client(
            FakeClient(content='```json\n{"overview": "ok", "movie_explanations": []}\n```')
        )

        result = generate_ai_overview("sci-fi", MOVIES[:1])

        assert result["ai_metadata"]["status"] == "success"
        assert result["overview"] == "ok"

    def test_malformed_json_degrades_to_parse_error(self, fake_client):
        fake_client(FakeClient(content="I am not JSON at all"))

        result = generate_ai_overview("sci-fi", MOVIES[:1])

        assert result["ai_metadata"]["status"] == "parse_error"
        # The raw text is preserved so the caller still has something to show.
        assert result["overview"] == "I am not JSON at all"

    def test_upstream_exception_degrades_to_error(self, fake_client):
        fake_client(FakeClient(error=TimeoutError("upstream timed out")))

        result = generate_ai_overview("sci-fi", MOVIES[:1])

        assert result["ai_metadata"]["status"] == "error"
        assert "upstream timed out" in result["ai_metadata"]["error"]
        assert result["movie_explanations"] == []

    def test_no_results_short_circuits(self, fake_client):
        client = fake_client(FakeClient(content="{}"))

        result = generate_ai_overview("sci-fi", [])

        assert result["ai_metadata"]["status"] == "no_results"
        assert client.calls == [], "must not call the LLM with no results"

    def test_prompt_is_capped_regardless_of_result_count(self, fake_client, monkeypatch):
        """A 100-result page must not become a 100-movie prompt."""
        monkeypatch.setattr("core.ai_overview.AI_OVERVIEW_MAX_MOVIES", 5)
        client = fake_client(
            FakeClient(content='{"overview": "ok", "movie_explanations": []}')
        )

        generate_ai_overview("sci-fi", MOVIES)

        user_message = client.calls[0]["messages"][1]["content"]
        assert "Movie 5" in user_message
        assert "Movie 6" not in user_message
