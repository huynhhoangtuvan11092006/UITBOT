"""Recursive character splitting without LangChain."""

from __future__ import annotations

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def recursive_split(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
    base_offset: int = 0,
) -> list[tuple[int, int]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
    if not text:
        return []
    separators = separators if separators is not None else DEFAULT_SEPARATORS
    spans = _split_recursive(text, chunk_size, separators, base_offset)
    return _apply_overlap(text, spans, chunk_size, chunk_overlap, base_offset)


def _split_recursive(
    text: str,
    chunk_size: int,
    separators: list[str],
    base_offset: int,
) -> list[tuple[int, int]]:
    if len(text) <= chunk_size:
        return [(base_offset, base_offset + len(text))] if text else []

    separator = ""
    remaining: list[str] = []
    for index, candidate in enumerate(separators):
        if candidate == "":
            separator = ""
            remaining = separators[index + 1 :]
            break
        if candidate in text:
            separator = candidate
            remaining = separators[index + 1 :]
            break

    if separator == "":
        spans: list[tuple[int, int]] = []
        for start in range(0, len(text), chunk_size):
            end = min(len(text), start + chunk_size)
            spans.append((base_offset + start, base_offset + end))
        return spans

    pieces = text.split(separator)
    spans = []
    cursor = 0
    buffer = ""
    buffer_start = 0
    for index, piece in enumerate(pieces):
        extra = separator if index < len(pieces) - 1 else ""
        candidate = piece + extra
        if not buffer:
            buffer_start = cursor
        if len(buffer) + len(candidate) <= chunk_size:
            buffer += candidate
        else:
            if buffer:
                spans.extend(
                    _split_recursive(
                        buffer,
                        chunk_size,
                        remaining or [""],
                        base_offset + buffer_start,
                    )
                )
            if len(candidate) > chunk_size:
                spans.extend(
                    _split_recursive(
                        candidate,
                        chunk_size,
                        remaining or [""],
                        base_offset + cursor,
                    )
                )
                buffer = ""
            else:
                buffer = candidate
                buffer_start = cursor
        cursor += len(candidate)
    if buffer:
        spans.extend(
            _split_recursive(
                buffer,
                chunk_size,
                remaining or [""],
                base_offset + buffer_start,
            )
        )
    return spans


def _apply_overlap(
    text: str,
    spans: list[tuple[int, int]],
    chunk_size: int,
    chunk_overlap: int,
    base_offset: int,
) -> list[tuple[int, int]]:
    if not spans or chunk_overlap == 0:
        return spans
    overlapped: list[tuple[int, int]] = []
    for index, (start, end) in enumerate(spans):
        if index == 0:
            overlapped.append((start, end))
            continue
        new_start = max(base_offset, start - chunk_overlap)
        if end - new_start > chunk_size * 2:
            new_start = end - chunk_size
        overlapped.append((new_start, end))
    return overlapped
