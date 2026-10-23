"""Fixed-size recursive character chunking."""

from __future__ import annotations

from app.config import settings
from app.ingestion.chunkers.assemble import spans_to_chunks
from app.ingestion.chunkers.split import recursive_split
from app.ingestion.models import Chunk, ConcatenatedDocument


def chunk_fixed(
    document: ConcatenatedDocument,
    *,
    chunk_set: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    size = settings.chunk_size if chunk_size is None else chunk_size
    overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    spans = recursive_split(
        document.text,
        chunk_size=size,
        chunk_overlap=overlap,
    )
    return spans_to_chunks(document, spans, chunk_set=chunk_set)
