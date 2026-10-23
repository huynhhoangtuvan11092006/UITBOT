"""Extract text from PDFs, with per-page OCR fallback."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import fitz

from app.config import settings
from app.ingestion.clean import (
    clean_text,
    detect_repeated_headers_footers,
    strip_repeated_lines,
)
from app.ingestion.models import ExtractedPage
from app.ingestion.ocr import OcrEngine, OcrResult, get_ocr_engine
from app.ingestion.quality import needs_ocr

logger = logging.getLogger(__name__)


def _render_page_png(page: fitz.Page, dpi: int) -> bytes:
    pixmap = page.get_pixmap(dpi=dpi)
    return pixmap.tobytes("png")


def extract_pdf(
    path: Path,
    *,
    engine: OcrEngine | None = None,
    dpi: int | None = None,
) -> list[ExtractedPage]:
    engine = engine or get_ocr_engine()
    dpi = settings.ocr_dpi if dpi is None else dpi
    document = fitz.open(path)
    raw_pages: list[tuple[int, str, fitz.Page]] = []
    try:
        for index, page in enumerate(document, start=1):
            raw = page.get_text("text") or ""
            raw_pages.append((index, raw, page))

        repeated = detect_repeated_headers_footers([text for _, text, _ in raw_pages])
        extracted: list[ExtractedPage] = []
        for page_number, raw, page in raw_pages:
            stripped_raw = strip_repeated_lines(raw, repeated)
            cleaned = clean_text(stripped_raw)
            method = "text"
            confidence: float | None = None
            text = cleaned
            if needs_ocr(cleaned):
                if engine.name == "none":
                    logger.warning(
                        "Page %s of %s looks like a scan but OCR is disabled",
                        page_number,
                        path.name,
                    )
                else:
                    image_bytes = _render_page_png(page, dpi)
                    result: OcrResult = engine.ocr_image(image_bytes)
                    ocr_text = clean_text(result.text)
                    if ocr_text:
                        text = ocr_text
                        method = "ocr"
                        confidence = result.confidence
                    else:
                        logger.warning(
                            "OCR returned empty text for %s page %s",
                            path.name,
                            page_number,
                        )
            extracted.append(
                ExtractedPage(
                    source_file=path.name,
                    page=page_number,
                    text=text,
                    extraction_method=method,
                    ocr_confidence=confidence,
                )
            )
        return extracted
    finally:
        document.close()


def write_ocr_review(pages: list[ExtractedPage], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_file", "page", "extraction_method", "ocr_confidence", "preview"])
        for page in pages:
            if page.extraction_method != "ocr":
                continue
            writer.writerow(
                [
                    page.source_file,
                    page.page,
                    page.extraction_method,
                    page.ocr_confidence if page.ocr_confidence is not None else "",
                    page.text[:300],
                ]
            )
    logger.info("Wrote OCR review file to %s", output)
