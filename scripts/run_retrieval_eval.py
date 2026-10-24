"""Evaluate Retrieval Pipeline: Ablation, Chunking, Embedding, and Breakdowns."""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.db.pool import connect  # noqa: E402
from app.generation.router import route_and_rewrite  # noqa: E402
from app.retrieval import retrieve_dense, retrieve_hybrid  # noqa: E402
from app.retrieval.models import RetrievedChunk  # noqa: E402
from app.retrieval.reranker import rerank  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    compute_latency_stats,
    compute_retrieval_metrics,
)
from openai import OpenAI  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("uitbot.eval.retrieval")

REWRITE_CACHE_PATH = Path("evaluation/rewritten_queries_cache.json")


def load_rewrite_cache() -> dict[str, str]:
    if REWRITE_CACHE_PATH.exists():
        try:
            return json.loads(REWRITE_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_rewrite_cache(cache: dict[str, str]) -> None:
    REWRITE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWRITE_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def get_or_rewrite_query(query: str, cache: dict[str, str], client: OpenAI | None = None) -> str:
    if query in cache:
        return cache[query]
    try:
        decision = route_and_rewrite(query, client=client)
        rewritten = decision.rewritten_query or query
    except Exception as exc:
        logger.warning("Rewrite failed for '%s': %s", query, exc)
        rewritten = query
    cache[query] = rewritten
    return rewritten


@dataclass
class RetrievalModeResult:
    mode_name: str
    description: str
    page_metrics: dict[str, float]
    article_metrics: dict[str, float]
    latency: dict[str, float]
    llm_calls_per_query: int


def run_retrieval_ablation(
    questions: list[dict[str, Any]],
) -> list[RetrievalModeResult]:
    """Run retrieval pipeline ablation across 4 configurations."""
    rewrite_cache = load_rewrite_cache()
    client = OpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key or "sk-dummy",
    )

    queries = [q["question"] for q in questions]
    labels = [q["relevant"] for q in questions]

    modes = [
        ("Dense (bge-m3)", "pgvector cosine similarity top_k=5", False, False, False),
        ("Hybrid (Dense + BM25)", "RRF(dense, bm25, k=60) top_k=5", False, True, False),
        ("Hybrid + Rerank", "Hybrid + bge-reranker-v2-m3 top_k=5", False, True, True),
        ("Rewrite + Hybrid + Rerank", "LLM rewrite + Hybrid + Rerank top_k=5", True, True, True),
    ]

    results: list[RetrievalModeResult] = []

    for name, desc, use_rewrite, use_hybrid, use_rerank in modes:
        logger.info("Evaluating retrieval mode: %s...", name)
        retrieved_all: list[list[RetrievedChunk]] = []
        latencies: list[float] = []

        for idx, q_text in enumerate(queries):
            start_t = time.perf_counter()

            # 1. Query preprocessing
            if use_rewrite:
                search_q = get_or_rewrite_query(q_text, rewrite_cache, client=client)
            else:
                search_q = q_text

            # 2. Retrieval
            if not use_hybrid:
                chunks = retrieve_dense(search_q, top_k=5)
            else:
                chunks = retrieve_hybrid(search_q, top_k=5)

            # 3. Reranking
            if use_rerank:
                chunks = rerank(search_q, chunks, top_k=5)

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            latencies.append(elapsed_ms)
            retrieved_all.append(chunks)

            if (idx + 1) % 25 == 0 or idx == len(queries) - 1:
                logger.info("  [%d/%d] processed for %s", idx + 1, len(queries), name)

        save_rewrite_cache(rewrite_cache)

        page_scores = compute_retrieval_metrics(retrieved_all, labels, level="page")
        art_scores = compute_retrieval_metrics(retrieved_all, labels, level="article")
        lat_stats = compute_latency_stats(latencies)

        results.append(
            RetrievalModeResult(
                mode_name=name,
                description=desc,
                page_metrics=asdict(page_scores),
                article_metrics=asdict(art_scores),
                latency=asdict(lat_stats),
                llm_calls_per_query=1 if use_rewrite else 0,
            )
        )

    return results


def run_group_breakdown(
    questions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute performance breakdown by question category."""
    rewrite_cache = load_rewrite_cache()
    client = OpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key or "sk-dummy",
    )

    groups: dict[str, list[dict[str, Any]]] = {}
    for q in questions:
        grp = q.get("group", "khác")
        groups.setdefault(grp, []).append(q)

    group_results: dict[str, dict[str, Any]] = {}

    for grp, grp_questions in groups.items():
        retrieved_all: list[list[RetrievedChunk]] = []
        latencies: list[float] = []
        labels = [q["relevant"] for q in grp_questions]

        for q in grp_questions:
            q_text = q["question"]
            start_t = time.perf_counter()
            search_q = get_or_rewrite_query(q_text, rewrite_cache, client=client)
            candidates = retrieve_hybrid(search_q, top_k=5)
            reranked = rerank(search_q, candidates, top_k=5)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            latencies.append(elapsed_ms)
            retrieved_all.append(reranked)

        page_scores = compute_retrieval_metrics(retrieved_all, labels, level="page")
        art_scores = compute_retrieval_metrics(retrieved_all, labels, level="article")
        lat_stats = compute_latency_stats(latencies)

        group_results[grp] = {
            "count": len(grp_questions),
            "page_recall_at_3": page_scores.recall_at_3,
            "page_recall_at_5": page_scores.recall_at_5,
            "page_mrr_at_5": page_scores.mrr_at_5,
            "article_recall_at_3": art_scores.recall_at_3,
            "article_recall_at_5": art_scores.recall_at_5,
            "article_mrr_at_5": art_scores.mrr_at_5,
            "mean_latency_ms": lat_stats.mean_ms,
        }

    return group_results


def run_ocr_breakdown(
    questions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Breakdown by whether the target regulation page was extracted via Clean Text vs OCR."""
    rewrite_cache = load_rewrite_cache()
    client = OpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key or "sk-dummy",
    )

    # Get OCR vs text classification from chunks table
    conn = connect()
    page_method_map: dict[tuple[str, int], str] = {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT source_file, page_start, extraction_method
            FROM chunks
            WHERE extraction_method IS NOT NULL
            """
        )
        for sf, p, m in cur.fetchall():
            page_method_map[(sf.strip().lower(), int(p))] = m
    conn.close()

    text_questions: list[dict[str, Any]] = []
    ocr_questions: list[dict[str, Any]] = []

    for q in questions:
        is_ocr = False
        for rel in q.get("relevant", []):
            sf = rel["source_file"].strip().lower()
            p = int(rel["page"])
            if page_method_map.get((sf, p)) == "ocr":
                is_ocr = True
                break
        if is_ocr:
            ocr_questions.append(q)
        else:
            text_questions.append(q)

    breakdown: dict[str, dict[str, Any]] = {}
    subsets = [
        ("Clean Text Pages", text_questions),
        ("Scanned / OCR Pages", ocr_questions),
    ]
    for subset_name, subset in subsets:
        if not subset:
            continue
        labels = [q["relevant"] for q in subset]
        retrieved_all = []
        for q in subset:
            q_text = q["question"]
            search_q = get_or_rewrite_query(q_text, rewrite_cache, client=client)
            candidates = retrieve_hybrid(search_q, top_k=5)
            reranked = rerank(search_q, candidates, top_k=5)
            retrieved_all.append(reranked)

        page_scores = compute_retrieval_metrics(retrieved_all, labels, level="page")
        art_scores = compute_retrieval_metrics(retrieved_all, labels, level="article")

        breakdown[subset_name] = {
            "count": len(subset),
            "page_recall_at_3": page_scores.recall_at_3,
            "page_recall_at_5": page_scores.recall_at_5,
            "page_mrr_at_5": page_scores.mrr_at_5,
            "article_recall_at_3": art_scores.recall_at_3,
            "article_recall_at_5": art_scores.recall_at_5,
            "article_mrr_at_5": art_scores.mrr_at_5,
        }

    return breakdown


def run_all_retrieval_eval(
    dataset_path: Path | str = "evaluation/golden_dataset.jsonl",
) -> dict[str, Any]:
    dataset_file = Path(dataset_path)
    with dataset_file.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    verified = [r for r in records if r.get("verified", False)]
    logger.info("Loaded %d verified questions from %s", len(verified), dataset_file)

    logger.info("=== STEP 1: Retrieval Pipeline Ablation ===")
    ablation_results = run_retrieval_ablation(verified)

    logger.info("=== STEP 2: Category Breakdown ===")
    group_results = run_group_breakdown(verified)

    logger.info("=== STEP 3: Text vs OCR Breakdown ===")
    ocr_results = run_ocr_breakdown(verified)

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_verified_questions": len(verified),
        "ablation_retrieval": [asdict(r) for r in ablation_results],
        "category_breakdown": group_results,
        "ocr_breakdown": ocr_results,
    }

    out_file = Path("evaluation/retrieval_eval_results.json")
    out_file.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Retrieval evaluation completed. Results written to %s", out_file)
    return final_payload


if __name__ == "__main__":
    run_all_retrieval_eval()
