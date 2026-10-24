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


import random
from dataclasses import dataclass, field


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Calculate non-parametric bootstrap confidence interval for mean of values."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    boot_means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    alpha = (1.0 - ci) / 2.0
    low_idx = int(round(alpha * (n_boot - 1)))
    high_idx = int(round((1.0 - alpha) * (n_boot - 1)))
    return (round(boot_means[low_idx], 4), round(boot_means[high_idx], 4))


def compute_hits_at_k(
    retrieved_lists: Sequence[Sequence[RetrievedChunk]],
    labels_list: Sequence[Sequence[dict[str, Any]]],
    k: int,
    level: str = "page",
) -> list[int]:
    """Return binary hit vector (0 or 1) per query at top-k."""
    hits: list[int] = []
    for retrieved, labels in zip(retrieved_lists, labels_list, strict=True):
        hit = 1 if any(is_hit(c, labels, level=level) for c in retrieved[:k]) else 0
        hits.append(hit)
    return hits


def compute_rr_at_k(
    retrieved_lists: Sequence[Sequence[RetrievedChunk]],
    labels_list: Sequence[Sequence[dict[str, Any]]],
    k: int = 5,
    level: str = "page",
) -> list[float]:
    """Return reciprocal rank per query up to rank k."""
    rrs: list[float] = []
    for retrieved, labels in zip(retrieved_lists, labels_list, strict=True):
        rr = 0.0
        for rank, chunk in enumerate(retrieved[:k], start=1):
            if is_hit(chunk, labels, level=level):
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return rrs


@dataclass(frozen=True)
class RetrievalScores:
    recall_at_3: float
    recall_at_5: float
    mrr_at_5: float
    recall_at_20: float | None = None
    recall_at_50: float | None = None
    ci_95: dict[str, tuple[float, float]] = field(default_factory=dict)


def compute_retrieval_metrics(
    retrieved_lists: Sequence[Sequence[RetrievedChunk]],
    labels_list: Sequence[Sequence[dict[str, Any]]],
    *,
    level: str = "page",
    compute_ci: bool = True,
    n_boot: int = 1000,
) -> RetrievalScores:
    """Compute Recall@3, Recall@5, MRR@5, and optionally Recall@20/50 with 95% bootstrap CIs."""
    if not retrieved_lists or len(retrieved_lists) != len(labels_list):
        return RetrievalScores(0.0, 0.0, 0.0)

    total = len(retrieved_lists)
    max_k = max(len(r) for r in retrieved_lists)

    hits_3 = compute_hits_at_k(retrieved_lists, labels_list, k=3, level=level)
    hits_5 = compute_hits_at_k(retrieved_lists, labels_list, k=5, level=level)
    rrs_5 = compute_rr_at_k(retrieved_lists, labels_list, k=5, level=level)

    r20_val = None
    hits_20 = None
    if max_k >= 20:
        hits_20 = compute_hits_at_k(retrieved_lists, labels_list, k=20, level=level)
        r20_val = round(sum(hits_20) / total, 4)

    r50_val = None
    hits_50 = None
    if max_k >= 50:
        hits_50 = compute_hits_at_k(retrieved_lists, labels_list, k=50, level=level)
        r50_val = round(sum(hits_50) / total, 4)

    ci_map: dict[str, tuple[float, float]] = {}
    if compute_ci:
        ci_map["recall_at_3"] = bootstrap_ci(hits_3, n_boot=n_boot)
        ci_map["recall_at_5"] = bootstrap_ci(hits_5, n_boot=n_boot)
        ci_map["mrr_at_5"] = bootstrap_ci(rrs_5, n_boot=n_boot)
        if hits_20 is not None:
            ci_map["recall_at_20"] = bootstrap_ci(hits_20, n_boot=n_boot)
        if hits_50 is not None:
            ci_map["recall_at_50"] = bootstrap_ci(hits_50, n_boot=n_boot)

    return RetrievalScores(
        recall_at_3=round(sum(hits_3) / total, 4),
        recall_at_5=round(sum(hits_5) / total, 4),
        mrr_at_5=round(sum(rrs_5) / total, 4),
        recall_at_20=r20_val,
        recall_at_50=r50_val,
        ci_95=ci_map,
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
