"""Text cleaning for Vietnamese regulation PDFs."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_SPACES_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> str:
    """Normalize Unicode and PDF artifacts while keeping line breaks for Điều headings."""
    text = unicodedata.normalize("NFC", text.replace("\x00", " ").replace("\u00a0", " "))
    text = _CONTROL_RE.sub(" ", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    lines = [_SPACES_RE.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _edge_lines(page_text: str, *, count: int = 2) -> list[str]:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if not lines:
        return []
    edges = lines[:count] + lines[-count:]
    return [unicodedata.normalize("NFC", line) for line in edges]


def detect_repeated_headers_footers(page_texts: list[str], *, min_ratio: float = 0.5) -> set[str]:
    if len(page_texts) < 3:
        return set()
    counts: Counter[str] = Counter()
    for text in page_texts:
        for line in set(_edge_lines(text)):
            if len(line) < 3:
                continue
            counts[line] += 1
    threshold = max(2, int(len(page_texts) * min_ratio))
    return {line for line, n in counts.items() if n >= threshold}


def strip_repeated_lines(text: str, repeated: set[str]) -> str:
    if not repeated:
        return text
    kept = [line for line in text.splitlines() if line.strip() not in repeated]
    return "\n".join(kept)
