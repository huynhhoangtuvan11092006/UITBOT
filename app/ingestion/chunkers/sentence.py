"""Sentence packing for Vietnamese text."""

from __future__ import annotations

import logging
import re

from app.config import settings
from app.ingestion.chunkers.assemble import spans_to_chunks
from app.ingestion.chunkers.split import recursive_split
from app.ingestion.models import Chunk, ConcatenatedDocument

logger = logging.getLogger(__name__)

_FALLBACK_SENT_RE = re.compile(r"(?<=[\.!?…])\s+")


def _sent_tokenize(text: str) -> list[str]:
    try:
        from underthesea import sent_tokenize

        sentences = sent_tokenize(text)
        if sentences:
            return sentences
    except Exception:
        logger.warning("underthesea sent_tokenize failed; using regex fallback", exc_info=True)
    return [part.strip() for part in _FALLBACK_SENT_RE.split(text) if part.strip()] or [text]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    sentences = _sent_tokenize(text)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence in sentences:
        index = text.find(sentence, cursor)
        if index < 0:
            index = cursor
        start = index
        end = start + len(sentence)
        spans.append((start, end))
        cursor = end
    return spans


def chunk_sentence(
    document: ConcatenatedDocument,
    *,
    chunk_set: str,
    chunk_size: int | None = None,
    sentence_overlap: int | None = None,
) -> list[Chunk]:
    size = settings.chunk_size if chunk_size is None else chunk_size
    overlap_n = settings.chunk_sentence_overlap if sentence_overlap is None else sentence_overlap
    sentence_spans = _sentence_spans(document.text)
    packed: list[tuple[int, int]] = []
    index = 0
    while index < len(sentence_spans):
        start = sentence_spans[index][0]
        end = sentence_spans[index][1]
        last = index
        if end - start > size:
            packed.extend(
                recursive_split(
                    document.text[start:end],
                    chunk_size=size,
                    chunk_overlap=min(settings.chunk_overlap, size - 1),
                    base_offset=start,
                )
            )
            index += 1
            continue
        while last + 1 < len(sentence_spans):
            nxt_end = sentence_spans[last + 1][1]
            if nxt_end - start > size:
                break
            last += 1
            end = nxt_end
        packed.append((start, end))
        next_index = last + 1 - overlap_n
        if next_index <= index:
            next_index = index + 1
        index = next_index
    return spans_to_chunks(document, packed, chunk_set=chunk_set)
