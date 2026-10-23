"""psycopg 3 connection pool."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg_pool import ConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


def init_pool() -> ConnectionPool:
    global _pool
    if _pool is not None:
        return _pool
    _pool = ConnectionPool(
        conninfo=settings.postgres_conninfo(),
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True},
        open=True,
    )
    logger.info("PostgreSQL pool opened")
    return _pool


def get_pool() -> ConnectionPool:
    if _pool is None:
        return init_pool()
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
        logger.info("PostgreSQL pool closed")


def ping_db() -> None:
    with get_pool().connection() as conn:
        conn.execute("SELECT 1")


def connect() -> psycopg.Connection:
    return psycopg.connect(settings.postgres_conninfo())


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    with get_pool().connection() as conn:
        yield conn
