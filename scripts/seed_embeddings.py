import os
import numpy as np
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer

# IMPORTANT: enables psycopg2 adaptation for pgvector type
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

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
            """
        )
        return cur.fetchall()


def create_connection():
    """Create a new database connection."""
    print("Connecting to database...")
    conn = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME,
        connect_timeout=0,
        options="-c statement_timeout=0"
    )
    conn.autocommit = False
    register_vector(conn)
    print("Connection successful!")
    return conn


def upsert_embeddings(conn, movie_ids, embeddings, chunk_size=500, page_size=50):
    upsert_sql = """
    INSERT INTO movie_embeddings (movie_id, embedding, updated_at)
    VALUES (%s, %s, NOW())
    ON CONFLICT (movie_id)
    DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = NOW();
    """

    total = len(movie_ids)
    with conn.cursor() as cur:
        for i in range(0, total, chunk_size):
            batch_ids = movie_ids[i : i + chunk_size]
            batch_emb = embeddings[i : i + chunk_size]

            # Convert numpy arrays to Python lists
            params = [(mid, emb.tolist()) for mid, emb in zip(batch_ids, batch_emb)]

            psycopg2.extras.execute_batch(cur, upsert_sql, params, page_size=page_size)
            conn.commit()
            print(f"Upserted {min(i + chunk_size, total)}/{total} embeddings...")


def main():
    model = SentenceTransformer(MODEL_NAME)

    # Step 1: Connect and fetch movies
    conn = create_connection()
    try:
        rows = fetch_movies(conn)
        movie_ids = [r["id"] for r in rows]
        texts = [build_text(r) for r in rows]
        print(f"Fetched {len(rows)} movies")
    finally:
        conn.close()
        print("Connection closed for encoding phase")

    # Step 2: Encode embeddings (no DB connection)
    print("Encoding embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    print(f"Generated {len(embeddings)} embeddings")

    # Step 3: Reconnect and upsert
    conn = create_connection()
    try:
        upsert_embeddings(conn, movie_ids, embeddings)
        print(f"Done: stored embeddings for {len(movie_ids)} movies")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
