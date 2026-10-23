"""Dataclasses shared across the ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedPage:
    source_file: str
    page: int
    text: str
    extraction_method: str
    ocr_confidence: float | None


@dataclass(frozen=True)
class PageSpan:
    start: int
    end: int
    page: ExtractedPage


@dataclass(frozen=True)
class StructureRegion:
    start: int
    end: int
    chapter: str | None
    article: str | None
    article_title: str | None


@dataclass(frozen=True)
class ConcatenatedDocument:
    source_file: str
    text: str
    page_spans: tuple[PageSpan, ...]
    regions: tuple[StructureRegion, ...]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    chunk_set: str
    source_file: str
    page_start: int
    page_end: int
    chapter: str | None
    article: str | None
    article_title: str | None
    text: str
    extraction_method: str
    ocr_confidence: float | None
