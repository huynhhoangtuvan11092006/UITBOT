"""Extract, chunk, embed, and upsert regulation PDFs."""

from __future__ import annotations

import logging
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.db.migrate import prepare_database  # noqa: E402
from app.db.pool import connect  # noqa: E402
from app.ingestion.chunkers import chunk_document, default_chunk_set  # noqa: E402
from app.ingestion.document import build_document, concatenate_pages  # noqa: E402
from app.ingestion.embed_and_load import (  # noqa: E402
    embedding_dimension,
    load_chunks,
    load_embedding_model,
)
from app.ingestion.extract import extract_pdf, write_ocr_review  # noqa: E402
from app.ingestion.models import Chunk, ExtractedPage  # noqa: E402
from app.ingestion.structure import parse_structure  # noqa: E402
from app.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger("uitbot.ingest")


def _log_stats(
    pages: list[ExtractedPage],
    chunks: list[Chunk],
    chunk_set: str,
) -> None:
    text_pages = sum(1 for page in pages if page.extraction_method == "text")
    ocr_pages = sum(1 for page in pages if page.extraction_method == "ocr")
    low_ocr = sum(
        1
        for page in pages
        if page.extraction_method == "ocr"
        and page.ocr_confidence is not None
        and page.ocr_confidence < settings.ocr_low_confidence
    )
    lengths = [len(chunk.text) for chunk in chunks]
    logger.info("chunk_set=%s", chunk_set)
    logger.info("pages text=%s ocr=%s ocr_low_confidence=%s", text_pages, ocr_pages, low_ocr)
    if lengths:
        logger.info(
            "chunks count=%s length avg=%.1f min=%s max=%s",
            len(lengths),
            statistics.mean(lengths),
            min(lengths),
            max(lengths),
        )
    else:
        logger.info("chunks count=0")


def ingest() -> int:
    setup_logging(debug=settings.debug)
    pdfs = sorted(settings.data_raw_dir.glob("*.pdf"))
    if not pdfs:
        logger.error("No PDF files found in %s", settings.data_raw_dir.resolve())
        return 1

    model = load_embedding_model()
    dim = embedding_dimension(model)
    chunk_set = default_chunk_set()
    all_pages: list[ExtractedPage] = []
    all_chunks: list[Chunk] = []

    connection = connect()
    try:
        connection.autocommit = True
        prepare_database(
            connection,
            embedding_model_name=settings.embedding_model_name,
            embedding_dim=dim,
        )
        for pdf_path in pdfs:
            logger.info("Ingesting %s", pdf_path.name)
            pages = extract_pdf(pdf_path)
            all_pages.extend(pages)
            text, _spans = concatenate_pages(pages)
            regions = parse_structure(text)
            document = build_document(pages, regions)
            chunks = chunk_document(document, chunk_set=chunk_set)
            load_chunks(
                connection,
                chunks,
                model_name=settings.embedding_model_name,
                model=model,
            )
            all_chunks.extend(chunks)
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM chunks
                    WHERE source_file = %s AND chunk_set = %s
                    """,
                    (pdf_path.name, chunk_set),
                )
                db_count = cursor.fetchone()[0]
            if db_count != len(chunks):
                logger.error(
                    "Chunk count mismatch for %s: pipeline=%s database=%s",
                    pdf_path.name,
                    len(chunks),
                    db_count,
                )
                return 1
            logger.info("Database chunks for %s: %s", pdf_path.name, db_count)

        review_path = settings.data_processed_dir / "ocr_review.csv"
        write_ocr_review(all_pages, review_path)
        _log_stats(all_pages, all_chunks, chunk_set)

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM chunks WHERE chunk_set = %s",
                (chunk_set,),
            )
            total = cursor.fetchone()[0]
        if total != len(all_chunks):
            logger.error(
                "Total chunk mismatch: pipeline=%s database=%s",
                len(all_chunks),
                total,
            )
            return 1
        logger.info("Ingest complete. database_chunks=%s", total)
        return 0
    finally:
        connection.close()


def main() -> None:
    raise SystemExit(ingest())


if __name__ == "__main__":
    main()
