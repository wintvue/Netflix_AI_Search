#!/usr/bin/env python3
"""AI Overview generation using Ollama."""

import json
import os
import re
import time
from typing import Literal, TypedDict

from ollama import Client

from core.config import get_logger

logger = get_logger(__name__)

# Configuration
OLLAMA_MODEL = "ministral-3:8b-cloud"
OLLAMA_KEEP_ALIVE = "10m"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

# Singleton client
_ollama_client: Client | None = None

# Status type
Status = Literal["success", "parse_error", "error", "no_results"]


def get_ollama_client() -> Client:
    """Get or create the Ollama client instance."""
    global _ollama_client
    if _ollama_client is None:
        headers = (
            {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}
        )
        _ollama_client = Client(host=OLLAMA_HOST, headers=headers or None)
    return _ollama_client


SYSTEM_PROMPT = """You are an AI assistant that summarizes and explains movie search results.

STRICT RULES:
- Do NOT add new trivia, cast, awards, or opinions.
- If information is missing, do not guess.
- If you know the movie plot, explain it.
- Do NOT make up plot points if you don't know the movie plot.

Your task:
1. Write a short overview explaining why these movies match the user search query.
2. For each movie, write 3–4 sentences explaining the match.

Tone:
- Neutral, factual, concise.
- No hype language.
- Dont use sentence like "This movie is a great match for the user search query" or "This movie is a good match for the user search query".

Output MUST be valid JSON matching this schema exactly:
{
  "overview": "Brief summary of why these movies in the list match the user search query",
  "movie_explanations": [
    {
      "id": <movie_id>,
      "title": "<movie_title>",
      "explanation": "3-4 sentences explaining the plot and why this movie matches the search query"
    }
  ]
}

Respond ONLY with valid JSON. No markdown, no code fences, no extra text."""


class MovieExplanation(TypedDict):
    """Explanation for a single movie match."""

    id: int
    title: str
    explanation: str


class AIOverviewResponse(TypedDict):
    """AI overview response structure."""

    overview: str
    movie_explanations: list[MovieExplanation]


def _strip_markdown_fences(content: str) -> str:
    """Remove markdown code fences from JSON content."""
    content = content.strip()
    # Remove opening fence with optional language specifier
    content = re.sub(r"^```(?:json)?\s*", "", content)
    # Remove closing fence
    content = re.sub(r"\s*```$", "", content)
    return content.strip()


def _build_response(
    overview: str,
    movie_explanations: list[MovieExplanation],
    model: str,
    generation_time_ms: float,
    status: Status,
    error: str | None = None,
    eval_count: int | None = None,
    prompt_eval_count: int | None = None,
) -> dict:
    """Build a standardized AI overview response."""
    metadata = {
        "model": model,
        "generation_time_ms": round(generation_time_ms, 2),
        "status": status,
    }
    if status == "success":
        metadata["eval_count"] = eval_count
        metadata["prompt_eval_count"] = prompt_eval_count
    elif error:
        metadata["error"] = error

    return {
        "overview": overview,
        "movie_explanations": movie_explanations,
        "ai_metadata": metadata,
    }


# Movie fields to include in context
_MOVIE_FIELDS = [
    ("tagline", "Tagline"),
    ("genres", "Genres"),
    ("overview", "Overview"),
    ("release_date", "Release Date"),
]


def format_movies_context(query: str, movies: list[dict]) -> str:
    """Format movie results as context for the AI model."""
    lines = [f"User Query: {query}", "", "Search Results:"]

    for i, movie in enumerate(movies, 1):
        lines.append(f"\n{i}. Movie ID: {movie.get('id')}")
        lines.append(f"   Title: {movie.get('title', 'Unknown')}")

        for key, label in _MOVIE_FIELDS:
            if value := movie.get(key):
                lines.append(f"   {label}: {value}")

        if rating := movie.get("vote_average"):
            lines.append(f"   Rating: {rating}/10")

    return "\n".join(lines)


def generate_ai_overview(
    query: str,
    movies: list[dict],
    model: str = OLLAMA_MODEL,
) -> dict:
    """
    Generate an AI overview explaining search results using Ollama.

    Args:
        query: The user's search query
        movies: List of movie dicts from search results
        model: Ollama model to use

    Returns:
        Dict containing overview, movie explanations, and generation metadata
    """
    if not movies:
        return _build_response(
            overview="No movies found matching your query.",
            movie_explanations=[],
            model=model,
            generation_time_ms=0,
            status="no_results",
        )

    logger.info(
        f"AI Overview | Query: '{query}' | Movies: {len(movies)} | Model: {model}"
    )
    context = format_movies_context(query, movies)
    start = time.time()

    try:
        response = get_ollama_client().chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            keep_alive=OLLAMA_KEEP_ALIVE,
        )

        generation_time = (time.time() - start) * 1000
        raw_content = response["message"]["content"]

        logger.info(f"AI Overview | Generation: {generation_time:.2f}ms")
        logger.debug(f"AI Overview | Raw response: {raw_content[:500]}...")

        # Parse JSON response
        content = _strip_markdown_fences(raw_content)
        parsed: AIOverviewResponse = json.loads(content)

        return _build_response(
            overview=parsed.get("overview", ""),
            movie_explanations=parsed.get("movie_explanations", []),
            model=model,
            generation_time_ms=generation_time,
            status="success",
            eval_count=response.get("eval_count"),
            prompt_eval_count=response.get("prompt_eval_count"),
        )

    except json.JSONDecodeError as e:
        generation_time = (time.time() - start) * 1000
        logger.warning(f"AI Overview | JSON parse error: {e}")
        logger.warning(f"AI Overview | Raw content: {raw_content}")

        return _build_response(
            overview=raw_content,
            movie_explanations=[],
            model=model,
            generation_time_ms=generation_time,
            status="parse_error",
            error=str(e),
        )

    except Exception as e:
        generation_time = (time.time() - start) * 1000
        logger.error(f"AI Overview | Error: {e}")

        return _build_response(
            overview="",
            movie_explanations=[],
            model=model,
            generation_time_ms=generation_time,
            status="error",
            error=str(e),
        )


if __name__ == "__main__":
    test_movies = [
        {
            "id": 1,
            "title": "Inception",
            "tagline": "Your mind is the scene of the crime",
            "genres": "Action, Science Fiction, Adventure",
            "overview": "A thief who enters the dreams of others to steal secrets from their subconscious.",
            "release_date": "2010-07-16",
            "vote_average": 8.4,
        },
        {
            "id": 2,
            "title": "The Matrix",
            "tagline": "The fight for the future begins",
            "genres": "Action, Science Fiction",
            "overview": "A computer hacker learns about the true nature of reality and his role in the war against its controllers.",
            "release_date": "1999-03-31",
            "vote_average": 8.2,
        },
    ]

    result = generate_ai_overview("mind-bending sci-fi movies", test_movies)
    print(json.dumps(result, indent=2))
