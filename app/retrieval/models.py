"""Retrieved chunk data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    chapter: str | None
    article: str | None
    article_title: str | None
    text: str
    score: float
