"""Ablation benchmark for Chunking Strategies: fixed 400/800/1200, sentence, structured."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion.chunkers.fixed import chunk_fixed  # noqa: E402
from app.ingestion.chunkers.sentence import chunk_sentence  # noqa: E402
from app.ingestion.chunkers.structured import chunk_structured  # noqa: E402
from app.ingestion.document import build_document, concatenate_pages  # noqa: E402
from app.ingestion.models import Chunk, ExtractedPage  # noqa: E402
from app.ingestion.structure import parse_structure  # noqa: E402
from app.retrieval.bm25 import BM25Index, get_tokenizer  # noqa: E402
from app.retrieval.models import RetrievedChunk  # noqa: E402
from evaluation.metrics import compute_retrieval_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("uitbot.eval.chunking")


def load_cached_documents(cache_file: Path | str = "data/processed/extracted_pages.json"):
    """Load cached ExtractedPage objects and reconstruct ConcatenatedDocuments."""
    path = Path(cache_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Extracted pages cache not found at {path}. Run extraction cache first."
        )

    raw_pages = json.loads(path.read_text(encoding="utf-8"))
    pages_by_doc: dict[str, list[ExtractedPage]] = {}
    for p in raw_pages:
        page_obj = ExtractedPage(
            source_file=p["source_file"],
            page=p["page"],
            text=p["text"],
            extraction_method=p["extraction_method"],
            ocr_confidence=p.get("ocr_confidence"),
        )
        pages_by_doc.setdefault(p["source_file"], []).append(page_obj)

    documents = []
    for _sf, pages in sorted(pages_by_doc.items()):
        text, _ = concatenate_pages(pages)
        regions = parse_structure(text)
        doc = build_document(pages, regions)
        documents.append(doc)

    return documents


def evaluate_chunk_set(
    chunk_set_name: str,
    chunks: list[Chunk],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build BM25 index on chunk set and measure retrieval accuracy on golden questions."""
    # Convert Chunk to RetrievedChunk format for BM25Index
    retrieved_chunk_list = [
        RetrievedChunk(
            chunk_id=c.chunk_id,
            source_file=c.source_file,
            page_start=c.page_start,
            page_end=c.page_end,
            chapter=c.chapter,
            article=c.article,
            article_title=c.article_title,
            text=c.text,
            score=0.0,
        )
        for c in chunks
    ]

    tokenizer = get_tokenizer("underthesea")
    index = BM25Index(chunk_set_name, retrieved_chunk_list, tokenizer=tokenizer)

    queries = [q["question"] for q in questions]
    labels = [q["relevant"] for q in questions]

    retrieved_all: list[list[RetrievedChunk]] = []
    start_t = time.perf_counter()
    for q_text in queries:
        top_chunks = index.search(q_text, top_k=5)
        retrieved_all.append(top_chunks)
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0 / len(queries)

    page_scores = compute_retrieval_metrics(retrieved_all, labels, level="page")
    art_scores = compute_retrieval_metrics(retrieved_all, labels, level="article")

    lengths = [len(c.text) for c in chunks]
    avg_len = sum(lengths) / max(len(lengths), 1)

    return {
        "chunk_strategy": chunk_set_name,
        "total_chunks": len(chunks),
        "avg_chunk_length": round(avg_len, 1),
        "page_recall_at_3": page_scores.recall_at_3,
        "page_recall_at_5": page_scores.recall_at_5,
        "page_mrr_at_5": page_scores.mrr_at_5,
        "article_recall_at_3": art_scores.recall_at_3,
        "article_recall_at_5": art_scores.recall_at_5,
        "article_mrr_at_5": art_scores.mrr_at_5,
        "latency_ms": round(elapsed_ms, 2),
    }


def run_chunking_ablation(
    golden_dataset_path: Path | str = "evaluation/golden_dataset.jsonl",
    cache_file: Path | str = "data/processed/extracted_pages.json",
) -> list[dict[str, Any]]:
    docs = load_cached_documents(cache_file)
    logger.info("Loaded %d documents from cache.", len(docs))

    with Path(golden_dataset_path).open("r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    verified = [q for q in questions if q.get("verified", False)]
    logger.info("Evaluating on %d verified questions.", len(verified))

    strategies = [
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

    results: list[dict[str, Any]] = []

    for name, chunker_fn in strategies:
        logger.info("Chunking documents using strategy: %s...", name)
        chunks: list[Chunk] = []
        for d in docs:
            chunks.extend(chunker_fn(d))
        logger.info("  Generated %d chunks. Evaluating retrieval...", len(chunks))
        res = evaluate_chunk_set(name, chunks, verified)
        results.append(res)
        logger.info(
            "  %s -> Page R@5: %.3f, Art R@5: %.3f, MRR@5: %.3f",
            name,
            res["page_recall_at_5"],
            res["article_recall_at_5"],
            res["page_mrr_at_5"],
        )

    out_file = Path("evaluation/chunking_ablation_results.json")
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Chunking ablation complete. Saved to %s", out_file)
    return results


if __name__ == "__main__":
    run_chunking_ablation()
