"""Embed chunks and upsert them into PostgreSQL."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import psycopg
from psycopg import sql

from app.config import settings
from app.db.embeddings import embedding_table_name, ensure_embedding_table
from app.ingestion.models import Chunk

logger = logging.getLogger(__name__)

_model = None


def load_embedding_model(model_name: str | None = None):
    global _model
    name = model_name or settings.embedding_model_name
    if _model is not None and getattr(_model, "_uitbot_name", None) == name:
        return _model
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s", name)
    model = SentenceTransformer(name)
    model._uitbot_name = name  # type: ignore[attr-defined]
    _model = model
    return model


def embedding_dimension(model) -> int:
    dim = model.get_sentence_embedding_dimension()
    if not dim:
        raise RuntimeError("Could not determine embedding dimension from model")
    return int(dim)


def embed_texts(
    texts: Sequence[str], *, model=None, batch_size: int | None = None
) -> list[list[float]]:
    model = model or load_embedding_model()
    size = settings.embedding_batch_size if batch_size is None else batch_size
    vectors = model.encode(
        list(texts),
        batch_size=size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def upsert_chunks(
    conn: psycopg.Connection,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
    *,
    embedding_table: str,
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")
    if not chunks:
        return
    with conn.cursor() as cursor:
        for chunk, vector in zip(chunks, embeddings, strict=True):
            cursor.execute(
                """
                INSERT INTO chunks (
                    chunk_id, chunk_set, source_file, page_start, page_end,
                    chapter, article, article_title, text,
                    extraction_method, ocr_confidence
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    chunk_set = EXCLUDED.chunk_set,
                    source_file = EXCLUDED.source_file,
                    page_start = EXCLUDED.page_start,
                    page_end = EXCLUDED.page_end,
                    chapter = EXCLUDED.chapter,
                    article = EXCLUDED.article,
                    article_title = EXCLUDED.article_title,
                    text = EXCLUDED.text,
                    extraction_method = EXCLUDED.extraction_method,
                    ocr_confidence = EXCLUDED.ocr_confidence
                """,
                (
                    chunk.chunk_id,
                    chunk.chunk_set,
                    chunk.source_file,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.chapter,
                    chunk.article,
                    chunk.article_title,
                    chunk.text,
                    chunk.extraction_method,
                    chunk.ocr_confidence,
                ),
            )
            vector_literal = "[" + ",".join(str(float(v)) for v in vector) + "]"
            cursor.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (chunk_id, embedding)
                    VALUES (%s, %s::vector)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding
                    """
                ).format(sql.Identifier(embedding_table)),
                (chunk.chunk_id, vector_literal),
            )
        source_file = chunks[0].source_file
        chunk_set = chunks[0].chunk_set
        keep_ids = [chunk.chunk_id for chunk in chunks]
        cursor.execute(
            """
            DELETE FROM chunks
            WHERE source_file = %s
              AND chunk_set = %s
              AND NOT (chunk_id = ANY(%s))
            """,
            (source_file, chunk_set, keep_ids),
        )
    logger.info(
        "Upserted %s chunks for %s / %s",
        len(chunks),
        source_file,
        chunk_set,
    )


def load_chunks(
    conn: psycopg.Connection,
    chunks: Sequence[Chunk],
    *,
    model_name: str | None = None,
    model=None,
) -> str:
    model = model or load_embedding_model(model_name)
    dim = embedding_dimension(model)
    table = ensure_embedding_table(
        conn,
        model_name=model_name or settings.embedding_model_name,
        dimension=dim,
    )
    if table != embedding_table_name(model_name or settings.embedding_model_name):
        raise RuntimeError("Embedding table name mismatch")
    embeddings = embed_texts([chunk.text for chunk in chunks], model=model)
    upsert_chunks(conn, chunks, embeddings, embedding_table=table)
    return table
