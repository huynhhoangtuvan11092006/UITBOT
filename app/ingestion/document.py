"""Concatenate pages and look up page/structure metadata by character offset."""

from __future__ import annotations

from app.ingestion.models import (
    ConcatenatedDocument,
    ExtractedPage,
    PageSpan,
    StructureRegion,
)


def concatenate_pages(pages: list[ExtractedPage]) -> tuple[str, tuple[PageSpan, ...]]:
    if not pages:
        return "", ()
    parts: list[str] = []
    spans: list[PageSpan] = []
    offset = 0
    source = pages[0].source_file
    for page in pages:
        if page.source_file != source:
            raise ValueError("Cannot concatenate pages from different files")
        if parts:
            parts.append("\n\n")
            offset += 2
        start = offset
        parts.append(page.text)
        offset += len(page.text)
        spans.append(PageSpan(start=start, end=offset, page=page))
    return "".join(parts), tuple(spans)


def lookup_pages(spans: tuple[PageSpan, ...], start: int, end: int) -> list[ExtractedPage]:
    found: list[ExtractedPage] = []
    for span in spans:
        if span.end <= start or span.start >= end:
            continue
        found.append(span.page)
    return found


def lookup_region(
    regions: tuple[StructureRegion, ...], start: int, end: int
) -> StructureRegion | None:
    best: StructureRegion | None = None
    best_overlap = 0
    for region in regions:
        overlap = min(region.end, end) - max(region.start, start)
        if overlap > best_overlap:
            best = region
            best_overlap = overlap
    return best


def build_document(
    pages: list[ExtractedPage],
    regions: tuple[StructureRegion, ...],
) -> ConcatenatedDocument:
    text, spans = concatenate_pages(pages)
    source = pages[0].source_file if pages else ""
    return ConcatenatedDocument(
        source_file=source,
        text=text,
        page_spans=spans,
        regions=regions,
    )
