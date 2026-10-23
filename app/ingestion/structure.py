"""Detect Chương / Điều / Khoản and map character offsets to article metadata."""

from __future__ import annotations

import re

from app.ingestion.models import StructureRegion

_CHAPTER_RE = re.compile(
    r"(?m)^(Chương\s+[IVXLCDM0-9]+)\s*[:.\-]?\s*(.*)$",
    re.IGNORECASE,
)
_ARTICLE_RE = re.compile(
    r"(?m)^(Điều\s+\d+[a-z]?)\s*[:.\-]?\s*(.*)$",
    re.IGNORECASE,
)
_CLAUSE_RE = re.compile(
    r"(?m)^(?:\d+[.)]|[a-z]\)|[a-z]\.)\s+\S",
    re.IGNORECASE,
)


def parse_structure(text: str) -> tuple[StructureRegion, ...]:
    events: list[tuple[int, str, str, str]] = []
    for match in _CHAPTER_RE.finditer(text):
        events.append((match.start(), "chapter", match.group(1).strip(), match.group(2).strip()))
    for match in _ARTICLE_RE.finditer(text):
        events.append((match.start(), "article", match.group(1).strip(), match.group(2).strip()))
    events.sort(key=lambda item: (item[0], 0 if item[1] == "chapter" else 1))

    if not events:
        return (
            StructureRegion(
                start=0,
                end=len(text),
                chapter=None,
                article=None,
                article_title=None,
            ),
        )

    chapter: str | None = None
    article: str | None = None
    article_title: str | None = None
    start = 0
    regions: list[StructureRegion] = []

    def flush(end: int) -> None:
        nonlocal start
        if end <= start:
            return
        regions.append(
            StructureRegion(
                start=start,
                end=end,
                chapter=chapter,
                article=article,
                article_title=article_title or None,
            )
        )
        start = end

    for pos, kind, name, title in events:
        flush(pos)
        if kind == "chapter":
            chapter = name
            article = None
            article_title = None
        else:
            article = name
            article_title = title or None

    flush(len(text))
    if not regions:
        return (
            StructureRegion(
                start=0,
                end=len(text),
                chapter=None,
                article=None,
                article_title=None,
            ),
        )
    return tuple(regions)


def find_clause_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    fragment = text[start:end]
    matches = list(_CLAUSE_RE.finditer(fragment))
    if len(matches) < 2:
        return [(start, end)]
    spans: list[tuple[int, int]] = []
    prefix_end = start + matches[0].start()
    if prefix_end > start:
        spans.append((start, prefix_end))
    for index, match in enumerate(matches):
        span_start = start + match.start()
        span_end = start + matches[index + 1].start() if index + 1 < len(matches) else end
        spans.append((span_start, span_end))
    return [(s, e) for s, e in spans if e > s]
