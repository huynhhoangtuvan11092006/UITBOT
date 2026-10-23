"""Decide whether a PDF page's extracted text is usable."""

from __future__ import annotations

import re
import unicodedata

from app.config import settings

_VIET_CHAR_RE = re.compile(
    r"[ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ"
    r"àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]"
)


def replacement_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    return text.count("\ufffd") / len(text)


def _letters(text: str) -> list[str]:
    return [char for char in text if unicodedata.category(char).startswith("L")]


def vietnamese_letter_ratio(text: str) -> float:
    letters = _letters(text)
    if not letters:
        return 0.0
    viet = sum(1 for char in letters if _VIET_CHAR_RE.fullmatch(char))
    return viet / len(letters)


def needs_ocr(
    text: str,
    *,
    min_chars: int | None = None,
    replacement_ratio: float | None = None,
    min_vietnamese_ratio: float | None = None,
) -> bool:
    min_chars = settings.ocr_min_chars if min_chars is None else min_chars
    replacement_ratio = (
        settings.ocr_replacement_char_ratio if replacement_ratio is None else replacement_ratio
    )
    min_vietnamese_ratio = (
        settings.ocr_min_vietnamese_ratio if min_vietnamese_ratio is None else min_vietnamese_ratio
    )
    stripped = text.strip()
    if len(stripped) < min_chars:
        return True
    if replacement_char_ratio(stripped) > replacement_ratio:
        return True
    letters = _letters(stripped)
    if len(letters) >= 20 and vietnamese_letter_ratio(stripped) < min_vietnamese_ratio:
        return True
    return False
