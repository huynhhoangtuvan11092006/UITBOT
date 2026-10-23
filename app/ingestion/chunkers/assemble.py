"""Turn character spans into Chunk objects with article and page metadata."""

from __future__ import annotations

from app.ingestion.document import lookup_pages, lookup_region
from app.ingestion.models import Chunk, ConcatenatedDocument


def spans_to_chunks(
    document: ConcatenatedDocument,
    spans: list[tuple[int, int]],
    *,
    chunk_set: str,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, (start, end) in enumerate(spans):
        text = document.text[start:end].strip()
        if not text:
            continue
        pages = lookup_pages(document.page_spans, start, end)
        region = lookup_region(document.regions, start, end)
        page_numbers = [page.page for page in pages] or [1]
        methods = {page.extraction_method for page in pages}
        confidences = [page.ocr_confidence for page in pages if page.ocr_confidence is not None]
        extraction_method = "ocr" if "ocr" in methods else "text"
        ocr_confidence = min(confidences) if confidences else None
        chunk_id = f"{document.source_file}::{chunk_set}::{index:04d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                chunk_set=chunk_set,
                source_file=document.source_file,
                page_start=min(page_numbers),
                page_end=max(page_numbers),
                chapter=region.chapter if region else None,
                article=region.article if region else None,
                article_title=region.article_title if region else None,
                text=text,
                extraction_method=extraction_method,
                ocr_confidence=ocr_confidence,
            )
        )
    return chunks
