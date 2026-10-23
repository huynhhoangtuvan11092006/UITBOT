"""Hybrid retrieval using Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from app.config import settings
from app.ingestion.chunkers import default_chunk_set
from app.retrieval.bm25 import retrieve_bm25
from app.retrieval.dense import retrieve_dense
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = 60,
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    score(d) = sum(1 / (k + rank_m(d))) where rank_m is 1-indexed.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank_zero, chunk in enumerate(ranked_list):
            rank = rank_zero + 1
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (1.0 / (k + rank))
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]

    fused_results: list[RetrievedChunk] = []
    for cid in sorted_ids:
        orig = chunk_map[cid]
        fused_results.append(
            RetrievedChunk(
                chunk_id=orig.chunk_id,
                source_file=orig.source_file,
                page_start=orig.page_start,
                page_end=orig.page_end,
                chapter=orig.chapter,
                article=orig.article,
                article_title=orig.article_title,
                text=orig.text,
                score=scores[cid],
            )
        )
    return fused_results


def retrieve_hybrid(
    query: str,
    *,
    top_k: int = 10,
    candidate_count: int | None = None,
    k: int | None = None,
    chunk_set: str | None = None,
    model_name: str | None = None,
    conn=None,
    dense_results: Sequence[RetrievedChunk] | None = None,
    bm25_results: Sequence[RetrievedChunk] | None = None,
) -> list[RetrievedChunk]:
    """Retrieve chunks by combining dense and BM25 retrievers with RRF."""
    candidate_count = candidate_count or settings.retrieval_candidate_count
    k = k or settings.rrf_k
    chunk_set = chunk_set or default_chunk_set()

    if dense_results is None:
        dense_results = retrieve_dense(
            query,
            top_k=candidate_count,
            chunk_set=chunk_set,
            model_name=model_name,
            conn=conn,
        )

    if bm25_results is None:
        bm25_results = retrieve_bm25(
            query,
            top_k=candidate_count,
            chunk_set=chunk_set,
            conn=conn,
        )

    return reciprocal_rank_fusion(
        [dense_results, bm25_results],
        k=k,
        top_k=top_k,
    )
