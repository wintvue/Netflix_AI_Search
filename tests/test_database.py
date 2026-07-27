"""Tests for pool checkout, slot accounting and pgvector registration."""

import threading

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
    def install(connections, slots=None):
        fake = FakePool(connections)
        monkeypatch.setattr(database, "pool", fake)
        monkeypatch.setattr(
            database,
            "_pool_slots",
            threading.Semaphore(len(connections) if slots is None else slots),
        )
        return fake

    yield install
    monkeypatch.setattr(database, "pool", None)
    monkeypatch.setattr(database, "_pool_slots", None)


def test_get_connection_requires_an_initialized_pool(monkeypatch):
    monkeypatch.setattr(database, "pool", None)
    monkeypatch.setattr(database, "_pool_slots", None)
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


class TestSlotAccounting:
    """
    The semaphore bounds concurrency to the pool size. Every path that takes a
    slot must give it back, or the pool deadlocks after enough failures.
    """

    def test_waiting_past_the_timeout_raises(self, pooled, registered, monkeypatch):
        pooled([SlottedConnection()], slots=1)
        monkeypatch.setattr(database, "_ACQUIRE_TIMEOUT_SECONDS", 0.01)

        database.get_connection()  # takes the only slot and never returns it

        with pytest.raises(TimeoutError):
            database.get_connection()

    def test_a_failed_checkout_releases_the_slot(self, pooled, registered):
        fake = pooled([SlottedConnection()], slots=1)

        def boom():
            raise RuntimeError("pool is broken")

        fake.getconn = boom

        with pytest.raises(RuntimeError):
            database.get_connection()

        assert database._pool_slots.acquire(blocking=False), "slot was leaked"

    def test_a_failed_registration_releases_the_slot(self, pooled, monkeypatch):
        pooled([SlottedConnection()], slots=1)
        monkeypatch.setattr(
            database, "register_vector", lambda _c: (_ for _ in ()).throw(RuntimeError)
        )
        monkeypatch.setattr(
            database, "_pgvector_registered", database.weakref.WeakSet()
        )

        with pytest.raises(RuntimeError):
            database.get_connection()

        assert database._pool_slots.acquire(blocking=False), "slot was leaked"

    def test_a_failed_putconn_still_releases_the_slot(self, pooled, registered):
        fake = pooled([SlottedConnection()], slots=1)
        conn = database.get_connection()

        def boom(_conn, close=False):
            raise RuntimeError("putconn failed")

        fake.putconn = boom

        with pytest.raises(RuntimeError):
            database.put_connection(conn)

        assert database._pool_slots.acquire(blocking=False), "slot was leaked"
