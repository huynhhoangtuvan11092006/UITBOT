"""Unit tests for evaluation.metrics."""

from __future__ import annotations

from app.retrieval.models import RetrievedChunk
from evaluation.metrics import (
    compute_latency_stats,
    compute_retrieval_metrics,
    is_article_match,
    is_page_match,
    normalize_article,
)


def test_normalize_article():
    assert normalize_article("Điều 8") == "điều 8"
    assert normalize_article("điều 8.") == "điều 8"
    assert normalize_article("Điều 10a: Học vụ") == "điều 10a"
    assert normalize_article(None) is None


def test_page_match():
    chunk = RetrievedChunk(
        chunk_id="test::0",
        source_file="quy_che_dao_tao.pdf",
        page_start=7,
        page_end=8,
        chapter="Chương II",
        article="Điều 8",
        article_title="Đăng ký học phần",
        text="Sample text",
        score=0.9,
    )
    # Exact page within span
    assert is_page_match(chunk, {"source_file": "quy_che_dao_tao.pdf", "page": 7}) is True
    # 1 page tolerance for cover scan offset
    assert is_page_match(chunk, {"source_file": "quy_che_dao_tao.pdf", "page": 6}) is True
    assert is_page_match(chunk, {"source_file": "quy_che_dao_tao.pdf", "page": 9}) is True
    # Out of range
    assert is_page_match(chunk, {"source_file": "quy_che_dao_tao.pdf", "page": 11}) is False
    # Different file
    assert is_page_match(chunk, {"source_file": "other.pdf", "page": 7}) is False


def test_article_match():
    chunk = RetrievedChunk(
        chunk_id="test::0",
        source_file="quy_che_dao_tao.pdf",
        page_start=7,
        page_end=8,
        chapter="Chương II",
        article="Điều 8",
        article_title="Đăng ký học phần",
        text="Sample text",
        score=0.9,
    )
    assert (
        is_article_match(chunk, {"source_file": "quy_che_dao_tao.pdf", "article": "Điều 8"})
        is True
    )
    assert (
        is_article_match(chunk, {"source_file": "quy_che_dao_tao.pdf", "article": "Điều 9"})
        is False
    )
    assert is_article_match(chunk, {"source_file": "quy_che_dao_tao.pdf", "article": None}) is False


def test_compute_retrieval_metrics():
    labels_q1 = [{"source_file": "quy_che.pdf", "page": 5, "article": "Điều 4"}]
    labels_q2 = [{"source_file": "quy_che.pdf", "page": 10, "article": "Điều 9"}]

    c_hit_q1 = RetrievedChunk("c1", "quy_che.pdf", 5, 5, "C1", "Điều 4", "Title", "Text", 0.9)
    c_miss = RetrievedChunk("c2", "quy_che.pdf", 20, 20, "C2", "Điều 15", "Title", "Text", 0.5)
    c_hit_q2 = RetrievedChunk("c3", "quy_che.pdf", 10, 10, "C3", "Điều 9", "Title", "Text", 0.8)

    # Q1: Hit at rank 1
    # Q2: Hit at rank 2
    retrieved = [
        [c_hit_q1, c_miss],
        [c_miss, c_hit_q2],
    ]
    labels = [labels_q1, labels_q2]

    scores = compute_retrieval_metrics(retrieved, labels, level="page")
    assert scores.recall_at_3 == 1.0
    assert scores.recall_at_5 == 1.0
    # Q1 rr = 1.0, Q2 rr = 0.5 => average MRR = 0.75
    assert scores.mrr_at_5 == 0.75


def test_compute_latency_stats():
    lats = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = compute_latency_stats(lats)
    assert stats.mean_ms == 30.0
    assert stats.p50_ms == 30.0


def test_bootstrap_ci():
    from evaluation.metrics import bootstrap_ci
    values = [1.0, 1.0, 1.0, 0.0, 0.0]  # mean = 0.6
    low, high = bootstrap_ci(values, n_boot=500, ci=0.95, seed=42)
    assert 0.1 <= low <= 0.6
    assert 0.6 <= high <= 1.0

