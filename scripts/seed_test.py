import os
import numpy as np
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
# IMPORTANT: enables psycopg2 adaptation for pgvector type
from pgvector.psycopg2 import register_vector

load_dotenv()
USER = os.getenv("DB_USER")
PASSWORD = os.getenv("DB_PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
DBNAME = os.getenv("DB_NAME")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims


def clean(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def build_text(row: dict) -> str:
    parts = [
        f"Title: {clean(row.get('title'))}",
        f"Original title: {clean(row.get('original_title'))}",
        f"Tagline: {clean(row.get('tagline'))}",
        f"Overview: {clean(row.get('overview'))}",
        f"Genres: {clean(row.get('genres'))}",
        f"Cast: {clean(row.get('cast'))}",
        f"Director: {clean(row.get('director'))}",
        f"Writers: {clean(row.get('writers'))}",
        f"Countries: {clean(row.get('production_countries'))}",
        f"Languages: {clean(row.get('spoken_languages'))}",
    ]
    # Remove empty "Field: " lines
    return "\n".join([p for p in parts if not p.endswith(": ")])


def fetch_movies(conn):
    # DictCursor gives dict-like rows
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
              id,
              title,
              original_title,
              tagline,
              overview,
              genres,
              "cast",
              director,
              writers,
              production_countries,
              spoken_languages
            FROM movies
            LIMIT 10
            """
        )
        return cur.fetchall()


def upsert_embeddings(conn, movie_ids, embeddings, chunk_size=2000):
    upsert_sql = """
    INSERT INTO movie_embeddings (movie_id, embedding, updated_at)
    VALUES (%s, %s, NOW())
    ON CONFLICT (movie_id)
    DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = NOW();
    """

    with conn.cursor() as cur:
        for i in range(0, len(movie_ids), chunk_size):
            batch_ids = movie_ids[i : i + chunk_size]
            batch_emb = embeddings[i : i + chunk_size]

            # pgvector adapter supports passing numpy arrays or python lists
            params = [(mid, emb) for mid, emb in zip(batch_ids, batch_emb)]

            psycopg2.extras.execute_batch(cur, upsert_sql, params, page_size=500)
            conn.commit()


def main():
    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to database with: ", USER, HOST, PORT, DBNAME)
    conn = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    print("Connection successful!")
    conn.autocommit = False
    register_vector(conn)  # register pgvector adapter

    try:
        rows = fetch_movies(conn)
        movie_ids = [r["id"] for r in rows]
        texts = [build_text(r) for r in rows]

        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine similarity friendly
        ).astype(np.float32)

        upsert_embeddings(conn, movie_ids, embeddings, chunk_size=2000)

        print(f"Done: stored embeddings for {len(movie_ids)} movies")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
