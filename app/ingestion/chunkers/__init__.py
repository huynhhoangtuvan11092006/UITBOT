"""Chunking strategies."""

from __future__ import annotations

from app.config import settings
from app.ingestion.chunkers.fixed import chunk_fixed
from app.ingestion.chunkers.sentence import chunk_sentence
from app.ingestion.chunkers.structured import chunk_structured
from app.ingestion.models import Chunk, ConcatenatedDocument


def default_chunk_set(strategy: str | None = None) -> str:
    name = strategy or settings.chunk_strategy
    if name == "fixed":
        return f"fixed_{settings.chunk_size}_{settings.chunk_overlap}"
    if name == "sentence":
        return f"sentence_{settings.chunk_size}_{settings.chunk_sentence_overlap}"
    if name == "structured":
        return f"structured_{settings.chunk_size}"
    raise ValueError(f"Unknown chunk strategy: {name}")


def chunk_document(
    document: ConcatenatedDocument,
    *,
    strategy: str | None = None,
    chunk_set: str | None = None,
) -> list[Chunk]:
    name = strategy or settings.chunk_strategy
    label = chunk_set or default_chunk_set(name)
    if name == "fixed":
        return chunk_fixed(document, chunk_set=label)
    if name == "sentence":
        return chunk_sentence(document, chunk_set=label)
    if name == "structured":
        return chunk_structured(document, chunk_set=label)
    raise ValueError(f"Unknown chunk strategy: {name}")
