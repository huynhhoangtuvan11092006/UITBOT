"""Cross-encoder reranking using BAAI/bge-reranker-v2-m3."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from app.config import settings
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

_RERANKER_CACHE: dict[str, object] = {}


def load_reranker_model(model_name: str | None = None):
    """Lazy-load the CrossEncoder model."""
    name = model_name or settings.reranker_model_name
    if name in _RERANKER_CACHE:
        return _RERANKER_CACHE[name]

    from sentence_transformers import CrossEncoder

    logger.info("Loading CrossEncoder reranker model: %s", name)
    model = CrossEncoder(name)
    _RERANKER_CACHE[name] = model
    return model


def rerank(
    query: str,
    chunks: Sequence[RetrievedChunk],
    *,
    top_k: int | None = None,
    model_name: str | None = None,
    batch_size: int | None = None,
    model=None,
    enabled: bool | None = None,
) -> list[RetrievedChunk]:
    """Score (query, chunk.text) pairs using CrossEncoder and sort descending.

    If reranking is disabled, preserves original order and returns top_k.
    """
    if not chunks:
        return []

    is_enabled = settings.reranker_enabled if enabled is None else enabled
    if not is_enabled and model is None:
        logger.debug("Reranker is disabled; returning un-reranked chunks")
        return list(chunks[:top_k]) if top_k is not None else list(chunks)

    model = model or load_reranker_model(model_name)
    batch_size = batch_size or settings.reranker_batch_size

    pairs = [[query, chunk.text] for chunk in chunks]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    reranked: list[RetrievedChunk] = []
    for chunk, score in zip(chunks, scores, strict=True):
        reranked.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                source_file=chunk.source_file,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                chapter=chunk.chapter,
                article=chunk.article,
                article_title=chunk.article_title,
                text=chunk.text,
                score=float(score),
            )
        )

    reranked.sort(key=lambda item: item.score, reverse=True)
    if top_k is not None:
        return reranked[:top_k]
    return reranked
