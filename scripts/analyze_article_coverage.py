"""Analyze the percentage of chunks with non-empty article.

Breakdown by:
1. Each chunk_set (fixed_400_50, fixed_800_150, fixed_1200_200, sentence_800_1, structured_800)
2. Each source_file
3. Separated by OCR pages vs Clean text pages
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion.chunkers.fixed import chunk_fixed  # noqa: E402
from app.ingestion.chunkers.sentence import chunk_sentence  # noqa: E402
from app.ingestion.chunkers.structured import chunk_structured  # noqa: E402
from app.ingestion.models import Chunk  # noqa: E402
from scripts.run_chunking_ablation import load_cached_documents  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("uitbot.eval.article_coverage")


def is_chunk_ocr(chunk: Chunk, ocr_page_map: dict[tuple[str, int], bool]) -> bool:
    """Check if chunk originated from an OCR page."""
    sf = chunk.source_file.strip().lower()
    for p in range(chunk.page_start, chunk.page_end + 1):
        if ocr_page_map.get((sf, p), False):
            return True
    return False


def analyze_article_coverage(
    cache_file: Path | str = "data/processed/extracted_pages.json",
) -> dict[str, Any]:
    documents = load_cached_documents(cache_file)

    # Build OCR page map
    raw_pages = json.loads(Path(cache_file).read_text(encoding="utf-8"))
    ocr_page_map: dict[tuple[str, int], bool] = {}
    for p in raw_pages:
        sf = p["source_file"].strip().lower()
        page_num = int(p["page"])
        ocr_page_map[(sf, page_num)] = p.get("extraction_method") == "ocr"

    chunk_strategies = [
        (
            "fixed_400_50",
            lambda d: chunk_fixed(d, chunk_size=400, chunk_overlap=50, chunk_set="fixed_400_50"),
        ),
        (
            "fixed_800_150",
            lambda d: chunk_fixed(d, chunk_size=800, chunk_overlap=150, chunk_set="fixed_800_150"),
        ),
        (
            "fixed_1200_200",
            lambda d: chunk_fixed(
                d, chunk_size=1200, chunk_overlap=200, chunk_set="fixed_1200_200"
            ),
        ),
        (
            "sentence_800_1",
            lambda d: chunk_sentence(
                d, chunk_size=800, sentence_overlap=1, chunk_set="sentence_800_1"
            ),
        ),
        (
            "structured_800",
            lambda d: chunk_structured(d, chunk_size=800, chunk_set="structured_800"),
        ),
    ]

    all_results: dict[str, Any] = {
        "summary_by_chunk_set": {},
        "by_file_and_set": {},
        "ocr_vs_clean_by_set": {},
    }

    for set_name, chunk_fn in chunk_strategies:
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = chunk_fn(doc)
            all_chunks.extend(chunks)

        total_chunks = len(all_chunks)
        with_art = sum(1 for c in all_chunks if c.article and c.article.strip())
        coverage_pct = round(with_art / max(total_chunks, 1) * 100, 2)

        # Separate OCR vs Clean
        ocr_chunks = [c for c in all_chunks if is_chunk_ocr(c, ocr_page_map)]
        clean_chunks = [c for c in all_chunks if not is_chunk_ocr(c, ocr_page_map)]

        ocr_with_art = sum(1 for c in ocr_chunks if c.article and c.article.strip())
        clean_with_art = sum(1 for c in clean_chunks if c.article and c.article.strip())

        ocr_pct = round(ocr_with_art / max(len(ocr_chunks), 1) * 100, 2)
        clean_pct = round(clean_with_art / max(len(clean_chunks), 1) * 100, 2)

        all_results["summary_by_chunk_set"][set_name] = {
            "total_chunks": total_chunks,
            "chunks_with_article": with_art,
            "article_coverage_pct": coverage_pct,
        }

        all_results["ocr_vs_clean_by_set"][set_name] = {
            "clean_total": len(clean_chunks),
            "clean_with_article": clean_with_art,
            "clean_coverage_pct": clean_pct,
            "ocr_total": len(ocr_chunks),
            "ocr_with_article": ocr_with_art,
            "ocr_coverage_pct": ocr_pct,
        }

        # Breakdown by file
        file_map: dict[str, list[Chunk]] = {}
        for c in all_chunks:
            file_map.setdefault(c.source_file, []).append(c)

        for sf, f_chunks in file_map.items():
            f_with_art = sum(1 for c in f_chunks if c.article and c.article.strip())
            f_pct = round(f_with_art / max(len(f_chunks), 1) * 100, 2)
            f_is_ocr = any(is_chunk_ocr(c, ocr_page_map) for c in f_chunks)

            all_results["by_file_and_set"].setdefault(sf, {})[set_name] = {
                "is_ocr": f_is_ocr,
                "total": len(f_chunks),
                "with_article": f_with_art,
                "coverage_pct": f_pct,
            }

    out_file = Path("evaluation/article_coverage_report.json")
    out_file.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved article coverage report to %s", out_file)
    return all_results


if __name__ == "__main__":
    analyze_article_coverage()
