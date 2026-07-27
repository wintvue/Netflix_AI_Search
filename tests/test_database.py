"""Tests for pool checkout and pgvector registration bookkeeping."""

import pytest

from core import database
from core.errors import ConfigurationError


class SlottedConnection:
    """
    Stand-in for psycopg2's connection: a C type with no ``__dict__``, so
    ``conn.anything = ...`` raises AttributeError. ``__slots__`` reproduces that
    while still supporting weak references, exactly like the real object.
    """

    __slots__ = ("closed", "__weakref__")

    def __init__(self):
        self.closed = False


class FakePool:
    def __init__(self, connections):
        self._connections = list(connections)
        self.checked_out = []
        self.returned = []

    def getconn(self):
        conn = self._connections.pop(0)
        self.checked_out.append(conn)
        return conn

    def putconn(self, conn, close=False):
        self.returned.append((conn, close))


@pytest.fixture
def registered(monkeypatch):
    """Record every connection register_vector is called on."""
    calls = []
    monkeypatch.setattr(database, "register_vector", calls.append)
    monkeypatch.setattr(database, "_pgvector_registered", database.weakref.WeakSet())
    return calls


@pytest.fixture
def pooled(monkeypatch):
    def install(connections):
        fake = FakePool(connections)
        monkeypatch.setattr(database, "pool", fake)
        return fake

    yield install
    monkeypatch.setattr(database, "pool", None)


def test_get_connection_requires_an_initialized_pool(monkeypatch):
    monkeypatch.setattr(database, "pool", None)
    with pytest.raises(ConfigurationError):
        database.get_connection()


def test_registers_pgvector_on_a_connection_that_rejects_new_attributes(
    pooled, registered
):
    conn = SlottedConnection()
    pooled([conn])

    assert database.get_connection() is conn
    assert registered == [conn]


def test_registration_happens_once_per_connection(pooled, registered):
    conn = SlottedConnection()
    pooled([conn, conn])

    database.get_connection()
    database.put_connection(conn)
    database.get_connection()

    assert registered == [conn]


def test_distinct_connections_are_each_registered(pooled, registered):
    first, second = SlottedConnection(), SlottedConnection()
    pooled([first, second])

    database.get_connection()
    database.get_connection()

    assert registered == [first, second]


def test_failed_registration_discards_the_connection(pooled, monkeypatch):
    conn = SlottedConnection()
    fake = pooled([conn])

    def boom(_conn):
        raise RuntimeError("no vector extension")

    monkeypatch.setattr(database, "register_vector", boom)
    monkeypatch.setattr(database, "_pgvector_registered", database.weakref.WeakSet())

    with pytest.raises(RuntimeError):
        database.get_connection()

    assert fake.returned == [(conn, True)]


def test_connection_context_manager_always_returns_the_connection(pooled, registered):
    conn = SlottedConnection()
    fake = pooled([conn])

    with pytest.raises(ValueError):
        with database.connection() as c:
            assert c is conn
            raise ValueError("caller blew up")

    assert fake.returned == [(conn, False)]
