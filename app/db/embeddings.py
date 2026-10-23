"""Dynamic embedding table names and DDL."""

from __future__ import annotations

import re

import psycopg
from psycopg import sql

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def model_slug(model_name: str) -> str:
    slug = _SLUG_RE.sub("_", model_name.lower()).strip("_")
    if not slug:
        raise ValueError("Embedding model name produced an empty slug.")
    return slug[:63]


def embedding_table_name(model_name: str) -> str:
    return f"chunk_embeddings_{model_slug(model_name)}"


def ensure_embedding_table(
    conn: psycopg.Connection,
    *,
    model_name: str,
    dimension: int,
) -> str:
    if dimension < 1 or dimension > 16_000:
        raise ValueError(f"Unsupported embedding dimension: {dimension}")
    table = embedding_table_name(model_name)
    if not re.fullmatch(r"[a-z][a-z0-9_]*", table):
        raise ValueError(f"Unsafe embedding table name: {table}")
    dim_sql = sql.SQL(str(int(dimension)))
    ident = sql.Identifier(table)
    index_name = f"{table}_hnsw_cosine"
    if not re.fullmatch(r"[a-z][a-z0-9_]*", index_name):
        raise ValueError(f"Unsafe index name: {index_name}")
    conn.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {} (
                chunk_id TEXT PRIMARY KEY REFERENCES chunks (chunk_id) ON DELETE CASCADE,
                embedding vector({}) NOT NULL
            )
            """
        ).format(ident, dim_sql)
    )
    conn.execute(
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {} ON {}
            USING hnsw (embedding vector_cosine_ops)
            """
        ).format(sql.Identifier(index_name), ident)
    )
    return table
