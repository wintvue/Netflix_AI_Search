#!/usr/bin/env python3

import argparse

from core.config import DEFAULT_TOP_K, setup_logging
from core.database import close_pool, create_db_pool
from core.search import hybrid_search, search_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Movie Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser(
        "search", help="Keyword search (BM25/full-text)"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("-k", type=int, default=DEFAULT_TOP_K)

    hybrid_parser = subparsers.add_parser(
        "hybrid", help="Hybrid search (vector + BM25, fused and reranked)"
    )
    hybrid_parser.add_argument("query", type=str, help="Search query")
    hybrid_parser.add_argument("-k", type=int, default=DEFAULT_TOP_K)
    hybrid_parser.add_argument(
        "--alpha", type=float, default=None, help="0=pure keyword, 1=pure semantic"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    setup_logging()
    create_db_pool()
    try:
        match args.command:
            case "search":
                print(f"Searching for: {args.query}")
                for i, movie in enumerate(search_movies(args.query, args.k), 1):
                    print(f"{i}. {movie['title']}")
            case "hybrid":
                kwargs = {"top_k": args.k}
                if args.alpha is not None:
                    kwargs["alpha"] = args.alpha
                response = hybrid_search(args.query, **kwargs)
                print(f"Searching for: {args.query}")
                for i, movie in enumerate(response["results"], 1):
                    score = movie.get("rerank_score")
                    suffix = f" (rerank={score:.3f})" if score is not None else ""
                    print(f"{i}. {movie['title']}{suffix}")
    finally:
        close_pool()


if __name__ == "__main__":
    main()
