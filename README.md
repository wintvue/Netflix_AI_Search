# Netflix AI Search

Hybrid movie search over Postgres + pgvector. A query is embedded, retrieved
through two independent indexes, fused by Reciprocal Rank Fusion, reranked by a
cross-encoder, and optionally summarized by an LLM.

## Pipeline

```
query
  │
  ├─ 1. Embed ................ Hugging Face Inference API (all-MiniLM-L6-v2, 384d)
  │                            cached in-process, one retry, hard timeout
  │
  ├─ 2. Retrieve ............. vector search (pgvector, cosine)   ─┐
  │                            BM25/FTS (websearch_to_tsquery)    ─┤
  │                                                                │
  ├─ 3. Fuse ................. Reciprocal Rank Fusion  ◄───────────┘
  │                            score = α/(k+rank_vec) + (1-α)/(k+rank_bm25)
  │                            uses rank position only, never raw scores
  │
  ├─ 4. Rerank ............... cross-encoder over the top candidates
  │                            best-effort: falls back to RRF order on failure
  │
  └─ 5. Overview (optional) .. LLM summary of the top results, streamable as SSE
```

The candidate pool narrows at every stage, so the sizes must satisfy:

```
VECTOR_CANDIDATES / BM25_CANDIDATES  >=  RERANK_CANDIDATES  >=  MAX_TOP_K
```

Violating this silently truncates results. The app checks it at startup and
logs a warning.

## Endpoints

| Method | Path               | Description                                        |
| ------ | ------------------ | -------------------------------------------------- |
| GET    | `/search`          | Hybrid search (RRF + rerank), optional AI overview  |
| GET    | `/search/semantic` | Vector search only                                  |
| GET    | `/search/keyword`  | Keyword search only (equivalent to `alpha=0`)       |
| GET    | `/health`          | Liveness — cheap, no dependencies touched           |
| GET    | `/ready`           | Readiness — pings the database, reports client state |
| GET    | `/docs`            | OpenAPI UI with full response schemas               |

`GET /search` parameters:

| Param         | Default | Notes                                                   |
| ------------- | ------- | ------------------------------------------------------- |
| `q`           | —       | Required, non-empty                                     |
| `k`           | `10`    | 1 … `MAX_TOP_K`                                         |
| `alpha`       | `0.8`   | 0 = pure keyword, 1 = pure semantic                     |
| `ai_overview` | `false` | Paid LLM call; rate limited per client IP               |
| `stream`      | `false` | SSE. Requires `ai_overview=true`, otherwise `422`       |

SSE emits three events in order: `results`, `overview`, `done` — results reach
the client while the overview is still generating.

`config.reranked` in the response says whether the rerank stage actually ran.
Reranking is best-effort: if the reranker is unavailable the RRF ordering is
returned and the flag is `false`, rather than failing the search.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # then fill in DB_*, HF_TOKEN
```

`.env.example` documents every variable the app reads.

> **Use `DB_HOST` / `DB_PORT`, not bare `HOST` / `PORT`.** Managed platforms
> inject `PORT` for the web listener — reading the database port from it points
> Postgres connections at the web port. Bare names still work locally (with a
> warning) and are refused outright on Render/Heroku/Cloud Run.

### Seeding

Seeding embeds the corpus locally and needs the extra ML stack, which the web
service never imports:

```bash
pip install -e ".[seed]"
python scripts/seed_embeddings.py
```

### Running

```bash
uvicorn api.main:app --reload
# http://localhost:8000/docs
```

CLI:

```bash
movie-search search "matrix"
movie-search hybrid "mind-bending sci-fi" -k 5 --alpha 0.8
```

## Reranking backends

`RERANK_BACKEND` selects where the cross-encoder runs:

- **`hf`** (default) — hosted Inference API, `BAAI/bge-reranker-base`. No local
  memory cost. Note that `cross-encoder/ms-marco-MiniLM-L-6-v2` cannot be used
  here: it has been renamed upstream and no inference provider serves it.
- **`local`** — in-process cross-encoder. Needs `pip install -e ".[rerank]"`
  and enough RAM for torch; too heavy for a free-tier dyno.
- **`none`** — skip the stage.

## Tests

```bash
pytest
```

The suite is pure unit tests — no database and no network. External calls are
mocked, so it runs in under a second.

## Deployment

Render config lives in `render.yaml`. Set in the dashboard:
`DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `HF_TOKEN`,
and optionally `OLLAMA_API_KEY` and `CORS_ORIGINS`.
