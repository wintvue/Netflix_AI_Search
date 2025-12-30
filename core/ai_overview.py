#!/usr/bin/env python3
"""AI Overview generation using Ollama."""

import json
import os
import time
from typing import TypedDict

from ollama import Client

from core.config import get_logger

logger = get_logger(__name__)

# Configuration
OLLAMA_MODEL = "ministral-3:8b-cloud"
OLLAMA_KEEP_ALIVE = "10m"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

# Initialize Ollama client
_ollama_client: Client | None = None


def get_ollama_client() -> Client:
    """Get or create the Ollama client instance."""
    global _ollama_client
    if _ollama_client is None:
        client_kwargs = {"host": OLLAMA_HOST}
        if OLLAMA_API_KEY:
            client_kwargs["headers"] = {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
        _ollama_client = Client(**client_kwargs)
    return _ollama_client

SYSTEM_PROMPT = """You are an AI assistant that summarizes and explains movie search results.

STRICT RULES:
- You may ONLY use facts explicitly provided in the input.
- Do NOT add new plot points, trivia, cast, awards, or opinions.
- If information is missing, do not guess.
- Do not use external knowledge.
- Do not say "it is widely known" or similar phrases.

Your task:
1. Write a short overview explaining why these movies match the user query.
2. For each movie, write 1–2 sentences explaining the match.

Tone:
- Neutral, factual, concise.
- No hype language.

Output MUST be valid JSON matching this schema exactly:
{
  "overview": "Brief summary of why these movies match the query",
  "movie_explanations": [
    {
      "id": <movie_id>,
      "title": "<movie_title>",
      "explanation": "1-2 sentences explaining why this movie matches the query"
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


def format_movies_context(query: str, movies: list[dict]) -> str:
    """
    Format movie results as context for the AI model.
    
    Args:
        query: The user's search query
        movies: List of movie dicts from search results
        
    Returns:
        Formatted context string for the AI prompt
    """
    context_parts = [f"User Query: {query}", "", "Search Results:"]
    
    for i, movie in enumerate(movies, 1):
        movie_info = [
            f"\n{i}. Movie ID: {movie.get('id')}",
            f"   Title: {movie.get('title', 'Unknown')}",
        ]
        
        if movie.get('tagline'):
            movie_info.append(f"   Tagline: {movie['tagline']}")
        
        if movie.get('genres'):
            movie_info.append(f"   Genres: {movie['genres']}")
        
        if movie.get('overview'):
            movie_info.append(f"   Overview: {movie['overview']}")
        
        if movie.get('release_date'):
            movie_info.append(f"   Release Date: {movie['release_date']}")
        
        if movie.get('vote_average'):
            movie_info.append(f"   Rating: {movie['vote_average']}/10")
        
        context_parts.extend(movie_info)
    
    return "\n".join(context_parts)


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
        model: Ollama model to use (default: qwen2.5:7b)
        
    Returns:
        Dict containing overview, movie explanations, and generation metadata
    """
    if not movies:
        return {
            "overview": "No movies found matching your query.",
            "movie_explanations": [],
            "ai_metadata": {
                "model": model,
                "generation_time_ms": 0,
                "status": "no_results",
            }
        }
    
    logger.info(f"AI Overview | Query: '{query}' | Movies: {len(movies)} | Model: {model}")
    
    # Format the context
    context = format_movies_context(query, movies)
    
    start = time.time()
    
    try:
        client = get_ollama_client()
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": context,
                },
            ],
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        
        generation_time = (time.time() - start) * 1000
        raw_content = response["message"]["content"]
        
        logger.info(f"AI Overview | Generation: {generation_time:.2f}ms")
        logger.debug(f"AI Overview | Raw response: {raw_content[:500]}...")
        
        # Parse the JSON response
        try:
            # Clean up potential markdown code fences
            content = raw_content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            parsed: AIOverviewResponse = json.loads(content)
            
            return {
                "overview": parsed.get("overview", ""),
                "movie_explanations": parsed.get("movie_explanations", []),
                "ai_metadata": {
                    "model": model,
                    "generation_time_ms": round(generation_time, 2),
                    "status": "success",
                    "eval_count": response.get("eval_count"),
                    "prompt_eval_count": response.get("prompt_eval_count"),
                }
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"AI Overview | JSON parse error: {e}")
            logger.warning(f"AI Overview | Raw content: {raw_content}")
            
            # Return raw content as fallback
            return {
                "overview": raw_content,
                "movie_explanations": [],
                "ai_metadata": {
                    "model": model,
                    "generation_time_ms": round(generation_time, 2),
                    "status": "parse_error",
                    "error": str(e),
                }
            }
            
    except Exception as e:
        generation_time = (time.time() - start) * 1000
        logger.error(f"AI Overview | Error: {e}")
        
        return {
            "overview": "",
            "movie_explanations": [],
            "ai_metadata": {
                "model": model,
                "generation_time_ms": round(generation_time, 2),
                "status": "error",
                "error": str(e),
            }
        }


# ==============================================================================
# Main (for testing)
# ==============================================================================

if __name__ == "__main__":
    # Test with sample movies
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

