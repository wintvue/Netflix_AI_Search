#!/usr/bin/env python3
"""Database connection pooling with pgvector support."""

import threading
import weakref
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

# Created once at startup and closed once at shutdown, both before any request
# thread exists, so neither needs a mutex.
pool: ThreadedConnectionPool | None = None

# Connections that already have the pgvector typecaster registered.
# psycopg2's connection is a C type with no __dict__, so the flag cannot live on
# the object itself; and a set of id(conn) values would be unsound because
# CPython recycles object ids, letting a fresh connection inherit a closed one's
# id and skip registration. A WeakSet keys on identity and drops entries when the
# connection is collected, which gets both right.
_pgvector_registered: weakref.WeakSet = weakref.WeakSet()

# psycopg2's pool raises PoolError the moment it is exhausted rather than
# waiting for a connection. Requests arrive on a threadpool far larger than the
# pool (asyncio.to_thread and FastAPI's sync-endpoint executor both default to
# ~32 workers), so callers queue on this semaphore instead of erroring out.
_pool_slots: threading.Semaphore | None = None

# How long a caller waits for a pool slot before giving up.
_ACQUIRE_TIMEOUT_SECONDS = 10


def create_db_pool() -> ThreadedConnectionPool:
    """Create the connection pool if it does not already exist."""
    global pool, _pool_slots

    if pool is not None:
        return pool

    missing = missing_db_settings()
    if missing:
        raise ConfigurationError(
            f"Missing database settings: {', '.join(missing)}. See .env.example."
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
    _pool_slots = threading.Semaphore(DB_POOL_MAX)
    return pool


def get_connection():
    """
    Check out a pooled connection with pgvector registered on it.

    Every caller must return it via `put_connection`; prefer the `connection()`
    context manager, which does that for you.
    """
    if pool is None or _pool_slots is None:
        raise ConfigurationError(
            "Database pool is not initialized; call create_db_pool() first"
        )

    if not _pool_slots.acquire(timeout=_ACQUIRE_TIMEOUT_SECONDS):
        raise TimeoutError(
            f"Timed out after {_ACQUIRE_TIMEOUT_SECONDS}s waiting for a database "
            f"connection (pool size {DB_POOL_MAX})"
        )

    try:
        conn = pool.getconn()
    except Exception:
        _pool_slots.release()
        raise

    # No lock here: register_vector does a round trip to look up the vector OID,
    # and registering twice is harmless, so a race costs one redundant query.
    if conn not in _pgvector_registered:
        try:
            register_vector(conn)
        except Exception:
            put_connection(conn, close=True)
            raise
        _pgvector_registered.add(conn)

    return conn


def put_connection(conn, close: bool = False) -> None:
    """Return a connection to the pool, optionally discarding it."""
    if pool is None:
        return
    try:
        pool.putconn(conn, close=close)
    finally:
        if _pool_slots is not None:
            _pool_slots.release()


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
    global pool, _pool_slots
    if pool is not None:
        pool.closeall()
        pool = None
        _pgvector_registered.clear()
        _pool_slots = None
        logger.info("Database connection pool closed")


def ping() -> bool:
    """Round-trip a trivial query to prove the database is reachable."""
    try:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        return True
    except Exception as e:
        logger.warning("Database ping failed: %s", e)
        return False
