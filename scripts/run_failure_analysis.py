"""Failure analysis and extended retrieval evaluation for Phase 5b.

Tasks:
1. Compute Recall@3, Recall@5, Recall@20, Recall@50 for Hybrid (page and article) + bootstrap 95% CIs.
2. Compute batched Hybrid + Rerank top 5 on all 105 verified questions.
3. Compute bootstrap 95% CIs for all retrieval modes, chunking strategies, and guardrail metrics.
4. Export evaluation/failure_analysis.csv with queries missing Page R@5 in Hybrid+Rerank.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.pool import connect  # noqa: E402
from app.retrieval.hybrid import retrieve_hybrid  # noqa: E402
from app.retrieval.models import RetrievedChunk  # noqa: E402
from app.retrieval.reranker import load_reranker_model  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    compute_retrieval_metrics,
    is_article_match,
    is_page_match,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("uitbot.eval.failure_analysis")


def load_extracted_pages_map(
    cache_path: Path | str = "data/processed/extracted_pages.json",
) -> dict[tuple[str, int], str]:
    """Map (source_file, page) -> page text."""
    path = Path(cache_path)
    if not path.exists():
        return {}
    pages = json.loads(path.read_text(encoding="utf-8"))
    res = {}
    for p in pages:
        sf = p["source_file"].strip().lower()
        res[(sf, int(p["page"]))] = p.get("text", "")
    return res


def run_phase_5b():
    # 1. Load verified questions
    dataset_file = Path("evaluation/golden_dataset.jsonl")
    with dataset_file.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    verified = [r for r in records if r.get("verified", False)]
    logger.info("Loaded %d verified questions", len(verified))

    queries = [q["question"] for q in verified]
    labels_list = [q["relevant"] for q in verified]
    pages_map = load_extracted_pages_map()

    # 2. Hybrid Top 50 Retrieval
    logger.info("Running Hybrid Top-50 retrieval on %d questions...", len(queries))
    conn = connect()
    hybrid_top50_all: list[list[RetrievedChunk]] = []
    for idx, q_text in enumerate(queries):
        chunks = retrieve_hybrid(q_text, top_k=50, candidate_count=60, conn=conn)
        hybrid_top50_all.append(chunks)
        if (idx + 1) % 25 == 0 or idx == len(queries) - 1:
            logger.info("  Hybrid retrieved [%d/%d]", idx + 1, len(queries))
    conn.close()

    # Compute Hybrid metrics at 3, 5, 20, 50
    page_scores_hybrid = compute_retrieval_metrics(hybrid_top50_all, labels_list, level="page")
    art_scores_hybrid = compute_retrieval_metrics(hybrid_top50_all, labels_list, level="article")

    logger.info(
        "Hybrid Page: R@3=%.3f, R@5=%.3f, R@20=%.3f, R@50=%.3f, MRR@5=%.3f",
        page_scores_hybrid.recall_at_3,
        page_scores_hybrid.recall_at_5,
        page_scores_hybrid.recall_at_20 or 0.0,
        page_scores_hybrid.recall_at_50 or 0.0,
        page_scores_hybrid.mrr_at_5,
    )
    logger.info(
        "Hybrid Art:  R@3=%.3f, R@5=%.3f, R@20=%.3f, R@50=%.3f, MRR@5=%.3f",
        art_scores_hybrid.recall_at_3,
        art_scores_hybrid.recall_at_5,
        art_scores_hybrid.recall_at_20 or 0.0,
        art_scores_hybrid.recall_at_50 or 0.0,
        art_scores_hybrid.mrr_at_5,
    )

    # 3. Hybrid + Rerank (top 10 candidates -> rerank top 5)
    logger.info("Running Batched CrossEncoder Rerank (top 10 candidates)...")
    reranker_model = load_reranker_model()

    # Build all pairs for batched prediction
    all_pairs: list[list[str]] = []
    query_pair_slices: list[tuple[int, int]] = []
    hybrid_top10_all = [c_list[:10] for c_list in hybrid_top50_all]

    curr_idx = 0
    for q_text, c_list in zip(queries, hybrid_top10_all, strict=True):
        start = curr_idx
        for c in c_list:
            all_pairs.append([q_text, c.text])
            curr_idx += 1
        end = curr_idx
        query_pair_slices.append((start, end))

    logger.info("Predicting CrossEncoder scores on %d pairs with batch_size=32...", len(all_pairs))
    all_scores = reranker_model.predict(all_pairs, batch_size=32, show_progress_bar=False)

    # Distribute back and sort
    reranked_top5_all: list[list[RetrievedChunk]] = []
    for (start, end), c_list in zip(query_pair_slices, hybrid_top10_all, strict=True):
        scores = all_scores[start:end]
        scored_chunks = []
        for c, s in zip(c_list, scores, strict=True):
            scored_chunks.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    source_file=c.source_file,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    chapter=c.chapter,
                    article=c.article,
                    article_title=c.article_title,
                    text=c.text,
                    score=float(s),
                )
            )
        scored_chunks.sort(key=lambda x: x.score, reverse=True)
        reranked_top5_all.append(scored_chunks[:5])

    page_scores_rerank = compute_retrieval_metrics(reranked_top5_all, labels_list, level="page")
    art_scores_rerank = compute_retrieval_metrics(reranked_top5_all, labels_list, level="article")
    logger.info(
        "Hybrid+Rerank Page: R@3=%.3f, R@5=%.3f, MRR@5=%.3f",
        page_scores_rerank.recall_at_3,
        page_scores_rerank.recall_at_5,
        page_scores_rerank.mrr_at_5,
    )

    # 4. Export failure_analysis.csv for all questions missing Page R@5 in Hybrid+Rerank
    failure_rows: list[dict[str, Any]] = []
    for q_item, top5 in zip(verified, reranked_top5_all, strict=True):
        q_id = q_item["id"]
        group = q_item.get("group", "")
        q_text = q_item["question"]
        relevant = q_item.get("relevant", [])

        # Check hits
        page_hit = any(is_page_match(c, rel) for c in top5 for rel in relevant)
        art_hit = any(is_article_match(c, rel) for c in top5 for rel in relevant)

        if not page_hit:  # trượt Page R@5
            # Format ground truth labels
            gt_label_str = "; ".join(
                f"{r['source_file']} (Điều {r.get('article') or '?'}, Trang {r.get('page') or '?'})"
                for r in relevant
            )

            # Ground truth text lookup
            gt_texts: list[str] = []
            for r in relevant:
                sf = r["source_file"].strip().lower()
                p = int(r["page"])
                text = pages_map.get((sf, p), "")
                if text:
                    gt_texts.append(f"[{r['source_file']} P.{p}]:\n{text[:400]}...")
            gt_text_str = "\n---\n".join(gt_texts) if gt_texts else "Chưa có đoạn văn trong cache."

            row: dict[str, Any] = {
                "question_id": q_id,
                "group": group,
                "question": q_text,
                "ground_truth_labels": gt_label_str,
                "ground_truth_text": gt_text_str,
                "page_hit_at_5": page_hit,
                "article_hit_at_5": art_hit,
            }

            for rank_i, c in enumerate(top5, start=1):
                preview = c.text.replace("\n", " ").strip()[:150]
                row[f"top{rank_i}_retrieved"] = (
                    f"{c.source_file} (P.{c.page_start}-{c.page_end}, {c.article or 'Không rõ Điều'}, score={c.score:.3f}): {preview}..."
                )

            row["error_category_suggestion"] = ""
            row["notes"] = ""
            failure_rows.append(row)

    out_csv = Path("evaluation/failure_analysis.csv")
    fieldnames = [
        "question_id",
        "group",
        "question",
        "ground_truth_labels",
        "ground_truth_text",
        "page_hit_at_5",
        "article_hit_at_5",
        "top1_retrieved",
        "top2_retrieved",
        "top3_retrieved",
        "top4_retrieved",
        "top5_retrieved",
        "error_category_suggestion",
        "notes",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failure_rows)

    logger.info("Exported %d failure cases to %s", len(failure_rows), out_csv)

    # 5. Save extended results JSON with bootstrap CIs
    extended_results = {
        "hybrid_top50": {
            "page_recall_at_3": page_scores_hybrid.recall_at_3,
            "page_recall_at_5": page_scores_hybrid.recall_at_5,
            "page_recall_at_20": page_scores_hybrid.recall_at_20,
            "page_recall_at_50": page_scores_hybrid.recall_at_50,
            "page_mrr_at_5": page_scores_hybrid.mrr_at_5,
            "page_ci_95": page_scores_hybrid.ci_95,
            "article_recall_at_3": art_scores_hybrid.recall_at_3,
            "article_recall_at_5": art_scores_hybrid.recall_at_5,
            "article_recall_at_20": art_scores_hybrid.recall_at_20,
            "article_recall_at_50": art_scores_hybrid.recall_at_50,
            "article_mrr_at_5": art_scores_hybrid.mrr_at_5,
            "article_ci_95": art_scores_hybrid.ci_95,
        },
        "hybrid_rerank_top5": {
            "page_recall_at_3": page_scores_rerank.recall_at_3,
            "page_recall_at_5": page_scores_rerank.recall_at_5,
            "page_mrr_at_5": page_scores_rerank.mrr_at_5,
            "page_ci_95": page_scores_rerank.ci_95,
            "article_recall_at_3": art_scores_rerank.recall_at_3,
            "article_recall_at_5": art_scores_rerank.recall_at_5,
            "article_mrr_at_5": art_scores_rerank.mrr_at_5,
            "article_ci_95": art_scores_rerank.ci_95,
        },
        "total_failures_page_r5": len(failure_rows),
    }

    out_json = Path("evaluation/retrieval_extended_results.json")
    out_json.write_text(
        json.dumps(extended_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved extended retrieval results to %s", out_json)
    return extended_results


if __name__ == "__main__":
    run_phase_5b()
