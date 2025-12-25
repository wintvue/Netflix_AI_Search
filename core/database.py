#!/usr/bin/env python3
"""Database connection handling."""

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from core.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER, get_logger

logger = get_logger(__name__)


def get_connection():
    """Create a new database connection with pgvector support."""
    logger.debug("Creating database connection")
    conn = psycopg2.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
    )
    register_vector(conn)
    return conn

