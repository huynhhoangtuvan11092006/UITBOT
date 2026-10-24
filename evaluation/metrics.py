"""Evaluation metrics for retrieval and ranking."""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.retrieval.models import RetrievedChunk

_ARTICLE_RE = re.compile(r"điều\s+(\d+[a-z]?)", re.IGNORECASE)


def normalize_article(art: str | None) -> str | None:
    """Normalize article string for robust comparison (e.g. 'Điều 8.' -> 'điều 8')."""
    if not art:
        return None
    match = _ARTICLE_RE.search(art.strip())
    if match:
        return f"điều {match.group(1).lower()}"
    return art.strip().lower()


def is_page_match(chunk: RetrievedChunk, label: dict[str, Any]) -> bool:
    """Check if chunk covers the labeled page (+/- 1 page tolerance for scan offset)."""
    if chunk.source_file.strip().lower() != label["source_file"].strip().lower():
        return False
    label_page = int(label["page"])
    return (chunk.page_start - 1) <= label_page <= (chunk.page_end + 1)


def is_article_match(chunk: RetrievedChunk, label: dict[str, Any]) -> bool:
    """Check if chunk matches the labeled article."""
    expected_art = label.get("article")
    if not expected_art or not chunk.article:
        return False
    if chunk.source_file.strip().lower() != label["source_file"].strip().lower():
        return False
    norm_chunk = normalize_article(chunk.article)
    norm_label = normalize_article(expected_art)
    return norm_chunk is not None and norm_chunk == norm_label


def is_hit(chunk: RetrievedChunk, labels: Sequence[dict[str, Any]], level: str = "page") -> bool:
    """Check if a chunk matches any ground-truth label under given match criteria."""
    for label in labels:
        if level == "page":
            if is_page_match(chunk, label):
                return True
        elif level == "article":
            if is_article_match(chunk, label):
                return True
        elif level == "any":
            if is_page_match(chunk, label) or is_article_match(chunk, label):
                return True
        elif level == "both":
            if is_page_match(chunk, label) and is_article_match(chunk, label):
                return True
        else:
            raise ValueError(f"Unknown match level: {level}")
    return False


@dataclass(frozen=True)
class RetrievalScores:
    recall_at_3: float
    recall_at_5: float
    mrr_at_5: float


def compute_retrieval_metrics(
    retrieved_lists: Sequence[Sequence[RetrievedChunk]],
    labels_list: Sequence[Sequence[dict[str, Any]]],
    *,
    level: str = "page",
) -> RetrievalScores:
    """Compute Recall@3, Recall@5, and MRR@5 for a list of query results."""
    if not retrieved_lists or len(retrieved_lists) != len(labels_list):
        return RetrievalScores(0.0, 0.0, 0.0)

    total = len(retrieved_lists)
    hits_at_3 = 0
    hits_at_5 = 0
    rr_at_5_sum = 0.0

    for retrieved, labels in zip(retrieved_lists, labels_list, strict=True):
        # Recall@3
        if any(is_hit(c, labels, level=level) for c in retrieved[:3]):
            hits_at_3 += 1

        # Recall@5 and MRR@5
        first_rank: int | None = None
        for rank, chunk in enumerate(retrieved[:5], start=1):
            if is_hit(chunk, labels, level=level):
                first_rank = rank
                break

        if first_rank is not None:
            hits_at_5 += 1
            rr_at_5_sum += 1.0 / first_rank

    return RetrievalScores(
        recall_at_3=round(hits_at_3 / total, 4),
        recall_at_5=round(hits_at_5 / total, 4),
        mrr_at_5=round(rr_at_5_sum / total, 4),
    )


@dataclass(frozen=True)
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p95_ms: float


def compute_latency_stats(latencies_ms: Sequence[float]) -> LatencyStats:
    """Compute mean, p50, and p95 latency in milliseconds."""
    if not latencies_ms:
        return LatencyStats(0.0, 0.0, 0.0)
    sorted_l = sorted(latencies_ms)
    mean_val = statistics.mean(sorted_l)
    p50_val = statistics.median(sorted_l)
    p95_idx = int(round(0.95 * (len(sorted_l) - 1)))
    p95_val = sorted_l[p95_idx]
    return LatencyStats(
        mean_ms=round(mean_val, 2),
        p50_ms=round(p50_val, 2),
        p95_ms=round(p95_val, 2),
    )
