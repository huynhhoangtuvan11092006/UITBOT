"""Apply static schema and embedding tables."""

from __future__ import annotations

import logging
from pathlib import Path

import psycopg

from app.db.embeddings import ensure_embedding_table

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def apply_schema(conn: psycopg.Connection) -> None:
    sql_text = SCHEMA_PATH.read_text(encoding="utf-8")
    for statement in sql_text.split(";"):
        stmt = statement.strip()
        if stmt:
            conn.execute(stmt)
    logger.info("Applied %s", SCHEMA_PATH.name)


def prepare_database(
    conn: psycopg.Connection,
    *,
    embedding_model_name: str,
    embedding_dim: int,
) -> str:
    apply_schema(conn)
    table = ensure_embedding_table(
        conn,
        model_name=embedding_model_name,
        dimension=embedding_dim,
    )
    logger.info("Embedding table ready: %s (dim=%s)", table, embedding_dim)
    return table
