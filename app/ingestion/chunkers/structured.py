"""Chunk by Chương / Điều / Khoản, falling back to fixed splits."""

from __future__ import annotations

from app.config import settings
from app.ingestion.chunkers.assemble import spans_to_chunks
from app.ingestion.chunkers.split import recursive_split
from app.ingestion.models import Chunk, ConcatenatedDocument
from app.ingestion.structure import find_clause_spans


def chunk_structured(
    document: ConcatenatedDocument,
    *,
    chunk_set: str,
    chunk_size: int | None = None,
) -> list[Chunk]:
    size = settings.chunk_size if chunk_size is None else chunk_size
    overlap = min(settings.chunk_overlap, size - 1)
    spans: list[tuple[int, int]] = []
    has_article = any(region.article for region in document.regions)

    if not has_article:
        spans = recursive_split(document.text, chunk_size=size, chunk_overlap=overlap)
        return spans_to_chunks(document, spans, chunk_set=chunk_set)

    for region in document.regions:
        length = region.end - region.start
        if length <= 0:
            continue
        if region.article is None:
            spans.extend(
                recursive_split(
                    document.text[region.start : region.end],
                    chunk_size=size,
                    chunk_overlap=overlap,
                    base_offset=region.start,
                )
            )
            continue
        if length <= size:
            spans.append((region.start, region.end))
            continue
        clause_spans = find_clause_spans(document.text, region.start, region.end)
        for clause_start, clause_end in clause_spans:
            if clause_end - clause_start <= size:
                spans.append((clause_start, clause_end))
            else:
                spans.extend(
                    recursive_split(
                        document.text[clause_start:clause_end],
                        chunk_size=size,
                        chunk_overlap=overlap,
                        base_offset=clause_start,
                    )
                )
    return spans_to_chunks(document, spans, chunk_set=chunk_set)
