#!/usr/bin/env python3
"""Database connection pooling with pgvector support."""

import threading
from contextlib import contextmanager

from pgvector.psycopg2 import register_vector
from psycopg2.pool import ThreadedConnectionPool

from core.config import (
    DB_CONNECT_TIMEOUT,
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_POOL_MAX,
    DB_POOL_MIN,
    DB_PORT,
    DB_USER,
    get_logger,
    missing_db_settings,
)
from core.errors import ConfigurationError

logger = get_logger(__name__)

pool: ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def create_db_pool() -> ThreadedConnectionPool:
    """Create the connection pool if it does not already exist."""
    global pool

    if pool is not None:
        return pool

    with _pool_lock:
        if pool is None:
            missing = missing_db_settings()
            if missing:
                raise ConfigurationError(
                    f"Missing database settings: {', '.join(missing)}. "
                    "See .env.example."
                )

            logger.info(
                "Creating database connection pool (min=%d, max=%d) to %s:%s/%s",
                DB_POOL_MIN,
                DB_POOL_MAX,
                DB_HOST,
                DB_PORT,
                DB_NAME,
            )
            pool = ThreadedConnectionPool(
                minconn=DB_POOL_MIN,
                maxconn=DB_POOL_MAX,
                host=DB_HOST,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                port=DB_PORT,
                connect_timeout=DB_CONNECT_TIMEOUT,
            )
    return pool


def get_connection():
    """
    Check out a pooled connection with pgvector registered on it.

    Every caller must return it via `put_connection`; prefer the `connection()`
    context manager, which does that for you.
    """
    if pool is None:
        raise ConfigurationError(
            "Database pool is not initialized; call create_db_pool() first"
        )

    conn = pool.getconn()

    # Track registration on the connection object itself. Keying a set on
    # id(conn) would be wrong: CPython recycles object ids, so a new connection
    # could inherit a closed one's id and skip pgvector registration.
    if not getattr(conn, "_pgvector_registered", False):
        try:
            register_vector(conn)
        except Exception:
            put_connection(conn, close=True)
            raise
        conn._pgvector_registered = True

    return conn


def put_connection(conn, close: bool = False) -> None:
    """Return a connection to the pool, optionally discarding it."""
    if pool is None:
        return
    pool.putconn(conn, close=close)


@contextmanager
def connection():
    """Context manager that always returns the connection to the pool."""
    conn = get_connection()
    try:
        yield conn
    finally:
        put_connection(conn)


def close_pool() -> None:
    """Close every pooled connection. Safe to call more than once."""
    global pool
    with _pool_lock:
        if pool is not None:
            pool.closeall()
            pool = None
            logger.info("Database connection pool closed")
