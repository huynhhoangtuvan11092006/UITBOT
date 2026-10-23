"""BM25 sparse retrieval using rank-bm25 and underthesea tokenization."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from rank_bm25 import BM25Okapi

from app.config import settings
from app.db.pool import connect
from app.ingestion.chunkers import default_chunk_set
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize_simple(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def _tokenize_underthesea(text: str) -> list[str]:
    try:
        from underthesea import word_tokenize

        tokenized = word_tokenize(text, format="text")
        return [tok.lower() for tok in tokenized.split() if tok.strip()]
    except Exception:
        logger.debug("underthesea tokenization failed; falling back to regex word tokenization")
        return _tokenize_simple(text)


def get_tokenizer(name: str | None = None) -> Callable[[str], list[str]]:
    choice = (name or settings.bm25_tokenizer).lower()
    if choice == "underthesea":
        return _tokenize_underthesea
    return _tokenize_simple


class BM25Index:
    """In-memory BM25 index built per chunk_set."""

    def __init__(
        self,
        chunk_set: str,
        chunks: list[RetrievedChunk],
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.chunk_set = chunk_set
        self.chunks = chunks
        self.tokenizer = tokenizer or get_tokenizer()
        corpus_tokens = [self.tokenizer(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        if not self._bm25 or not self.chunks:
            return []
        query_tokens = self.tokenizer(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]

        results: list[RetrievedChunk] = []
        for idx in top_indices:
            score = float(scores[idx])
            chunk = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    source_file=chunk.source_file,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    chapter=chunk.chapter,
                    article=chunk.article,
                    article_title=chunk.article_title,
                    text=chunk.text,
                    score=score,
                )
            )
        return results


_INDEX_CACHE: dict[str, BM25Index] = {}


def load_chunks_from_db(conn, chunk_set: str) -> list[RetrievedChunk]:
    sql = """
    SELECT
        chunk_id, source_file, page_start, page_end,
        chapter, article, article_title, text
    FROM chunks
    WHERE chunk_set = %s
    ORDER BY id ASC;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (chunk_set,))
        rows = cur.fetchall()

    return [
        RetrievedChunk(
            chunk_id=str(r[0]),
            source_file=str(r[1]),
            page_start=int(r[2]),
            page_end=int(r[3]),
            chapter=r[4],
            article=r[5],
            article_title=r[6],
            text=str(r[7]),
            score=0.0,
        )
        for r in rows
    ]


def get_bm25_index(
    chunk_set: str | None = None,
    *,
    conn=None,
    tokenizer: Callable[[str], list[str]] | None = None,
) -> BM25Index:
    chunk_set = chunk_set or default_chunk_set()
    if chunk_set in _INDEX_CACHE:
        return _INDEX_CACHE[chunk_set]

    close_after = False
    if conn is None:
        conn = connect()
        close_after = True

    try:
        chunks = load_chunks_from_db(conn, chunk_set)
        index = BM25Index(chunk_set, chunks, tokenizer=tokenizer)
        _INDEX_CACHE[chunk_set] = index
        logger.info("Built BM25 index for chunk_set=%s with %d chunks", chunk_set, len(chunks))
        return index
    finally:
        if close_after:
            conn.close()


def rebuild(
    chunk_set: str | None = None,
    *,
    conn=None,
    tokenizer: Callable[[str], list[str]] | None = None,
) -> BM25Index:
    """Explicitly rebuild the in-memory BM25 index."""
    chunk_set = chunk_set or default_chunk_set()
    if chunk_set in _INDEX_CACHE:
        del _INDEX_CACHE[chunk_set]
    return get_bm25_index(chunk_set=chunk_set, conn=conn, tokenizer=tokenizer)


def retrieve_bm25(
    query: str,
    *,
    top_k: int = 10,
    chunk_set: str | None = None,
    conn=None,
    index: BM25Index | None = None,
) -> list[RetrievedChunk]:
    """Retrieve top_k chunks using BM25 sparse index."""
    if index is None:
        index = get_bm25_index(chunk_set=chunk_set, conn=conn)
    return index.search(query, top_k=top_k)
