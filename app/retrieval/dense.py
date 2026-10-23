"""Dense retrieval using pgvector HNSW cosine index."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from app.config import settings
from app.db.embeddings import embedding_table_name
from app.db.pool import connect
from app.ingestion.chunkers import default_chunk_set
from app.ingestion.embed_and_load import load_embedding_model
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


def _vector_to_sql_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vector) + "]"


def retrieve_dense(
    query: str,
    *,
    top_k: int = 10,
    chunk_set: str | None = None,
    model_name: str | None = None,
    conn=None,
    model=None,
) -> list[RetrievedChunk]:
    """Retrieve top_k chunks using pgvector HNSW index.

    Never scans the whole table into memory; uses ORDER BY <=> LIMIT.
    """
    model_name = model_name or settings.embedding_model_name
    chunk_set = chunk_set or default_chunk_set()
    table = embedding_table_name(model_name)

    model = model or load_embedding_model(model_name)
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if hasattr(query_vector, "tolist"):
        query_vector = query_vector.tolist()
    vec_literal = _vector_to_sql_literal(query_vector)

    sql = f"""
    SELECT
        c.chunk_id,
        c.source_file,
        c.page_start,
        c.page_end,
        c.chapter,
        c.article,
        c.article_title,
        c.text,
        1.0 - (e.embedding <=> %s::vector) AS score
    FROM {table} e
    JOIN chunks c ON c.chunk_id = e.chunk_id
    WHERE c.chunk_set = %s
    ORDER BY e.embedding <=> %s::vector
    LIMIT %s;
    """

    close_after = False
    if conn is None:
        conn = connect()
        close_after = True

    try:
        with conn.cursor() as cur:
            cur.execute(sql, (vec_literal, chunk_set, vec_literal, top_k))
            rows = cur.fetchall()

        results: list[RetrievedChunk] = []
        for row in rows:
            results.append(
                RetrievedChunk(
                    chunk_id=str(row[0]),
                    source_file=str(row[1]),
                    page_start=int(row[2]),
                    page_end=int(row[3]),
                    chapter=row[4],
                    article=row[5],
                    article_title=row[6],
                    text=str(row[7]),
                    score=float(row[8]),
                )
            )
        return results
    finally:
        if close_after:
            conn.close()
