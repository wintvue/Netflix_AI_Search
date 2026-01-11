#!/usr/bin/env python3
"""Database connection handling."""

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from psycopg2.pool import ThreadedConnectionPool
import numpy as np
import time
from core.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER, get_logger

logger = get_logger(__name__)

conn = None

pool: ThreadedConnectionPool | None = None

_registered_conn_ids: set[int] | None = set()


def create_db_pool():
    """Create a new database connection with pgvector support."""
    global pool
    logger.info(f"Getting database connection pool: {pool is None}")
    if pool is None:
        logger.info("Creating database connection pool")
        pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            connect_timeout=5,
        )
    return pool


def get_connection():
    global pool
    conn = pool.getconn()
    print("Connection:", conn)
    cid = id(conn)
    global _registered_conn_ids
    if cid not in _registered_conn_ids:
        register_vector(conn)
        _registered_conn_ids.add(cid)

    return conn  # returns a connection from the pool


def put_connection(conn, close: bool = False):
    global pool
    cid = id(conn)
    if close:
        _registered_conn_ids.discard(cid)
        pool.putconn(conn, close=True)
    else:
        pool.putconn(conn)


def init_pool_vectors(pool):
    conns = []
    try:
        # Initialize all existing connections in the pool
        for _ in range(pool.minconn):
            conn = pool.getconn()
            register_vector(conn)
            conns.append(conn)
    finally:
        for c in conns:
            pool.putconn(c)


# def get_connection():
#     """Create a new database connection with pgvector support."""
#     global conn
#     logger.info(f"Getting database connection: {conn is None}")
#     if conn is None:
#         logger.debug("Creating database connection")
#         conn = psycopg2.connect(
#             user=DB_USER,
#             password=DB_PASSWORD,
#             host=DB_HOST,
#             port=DB_PORT,
#             dbname=DB_NAME,
#         )
#         register_vector(conn)
#     return conn


def close_connection():
    """Close the database connection."""
    global conn
    if conn is not None:
        conn.close()
        conn = None


WARM_QUERY = """
SELECT movie_id
FROM movie_embeddings
ORDER BY embedding <=> %s
LIMIT 1;
"""


def warm_loop():
    # all-MiniLM-L6-v2 = 384 dimensions
    dummy = np.zeros((384,), dtype=np.float32)

    while True:
        try:
            conn = get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(WARM_QUERY, (dummy,))
                    cur.fetchone()
                    logger.info("Warm loop query executed")
            finally:
                put_connection(conn)
        except Exception as e:
            logger.warning(f"Warm loop failed: {e}")

        time.sleep(60)  # 1 minute
