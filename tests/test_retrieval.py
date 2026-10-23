"""Unit tests for Phase 2 retrieval: Dense, BM25, Hybrid RRF, and Reranker."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from app.retrieval.bm25 import BM25Index, _tokenize_simple
from app.retrieval.dense import _vector_to_sql_literal, retrieve_dense
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import rerank


def _make_chunk(
    cid: str, text: str, article: str | None = None, score: float = 0.0
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        source_file="test.pdf",
        page_start=1,
        page_end=1,
        chapter="Chương I",
        article=article,
        article_title="Tiêu đề",
        text=text,
        score=score,
    )


def test_rrf_scoring_math() -> None:
    c1 = _make_chunk("c1", "text 1")
    c2 = _make_chunk("c2", "text 2")
    c3 = _make_chunk("c3", "text 3")
    c4 = _make_chunk("c4", "text 4")

    # List 1: c1 (rank 1), c2 (rank 2), c3 (rank 3)
    list1 = [c1, c2, c3]
    # List 2: c2 (rank 1), c1 (rank 2), c4 (rank 3)
    list2 = [c2, c1, c4]

    fused = reciprocal_rank_fusion([list1, list2], k=60, top_k=10)

    # Both c1 and c2 appear at ranks 1 & 2 in alternating lists:
    # expected score = 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    expected_top_score = (1.0 / 61.0) + (1.0 / 62.0)
    top_ids = {fused[0].chunk_id, fused[1].chunk_id}
    assert top_ids == {"c1", "c2"}
    assert pytest.approx(fused[0].score, rel=1e-5) == expected_top_score
    assert pytest.approx(fused[1].score, rel=1e-5) == expected_top_score

    # c3 and c4 appear only once at rank 3:
    # expected score = 1/(60+3) = 1/63
    expected_lower_score = 1.0 / 63.0
    lower_ids = {fused[2].chunk_id, fused[3].chunk_id}
    assert lower_ids == {"c3", "c4"}
    assert pytest.approx(fused[2].score, rel=1e-5) == expected_lower_score
    assert pytest.approx(fused[3].score, rel=1e-5) == expected_lower_score


def test_bm25_index_search() -> None:
    chunks = [
        _make_chunk("c1", "Quy định về thời gian thi và hoãn thi kết thúc học phần"),
        _make_chunk("c2", "Điều kiện làm khóa luận tốt nghiệp cho sinh viên chính quy"),
        _make_chunk("c3", "Chuẩn đầu ra chứng chỉ ngoại ngữ tiếng Anh tương đương TOEIC VNU-EPT"),
    ]

    index = BM25Index("test_set", chunks, tokenizer=_tokenize_simple)

    # Query matching chunk 1
    results_thi = index.search("hoãn thi học phần", top_k=2)
    assert len(results_thi) > 0
    assert results_thi[0].chunk_id == "c1"
    assert results_thi[0].score > 0.0

    # Query matching chunk 3
    results_nn = index.search("ngoại ngữ TOEIC", top_k=2)
    assert len(results_nn) > 0
    assert results_nn[0].chunk_id == "c3"
    assert results_nn[0].score > 0.0


def test_vector_to_sql_literal() -> None:
    vec = [0.123456789, -0.987654321]
    literal = _vector_to_sql_literal(vec)
    assert literal.startswith("[0.12345679,-0.98765432")
    assert literal.endswith("]")


class _FakeCursor:
    def __init__(self) -> None:
        self.last_query = ""
        self.last_params = None

    def execute(self, query: str, params=None) -> None:
        self.last_query = query
        self.last_params = params

    def fetchall(self) -> list:
        return [
            ("c1", "file.pdf", 1, 2, "Chương I", "Điều 5", "Hoãn thi", "Nội dung hoãn thi", 0.88),
        ]

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _FakeConn:
    def __init__(self) -> None:
        self.cur = _FakeCursor()

    def cursor(self) -> _FakeCursor:
        return self.cur


class _FakeModel:
    def encode(self, query: str, **kwargs) -> list[float]:
        return [0.1, 0.2, 0.3]


def test_dense_retrieval_uses_index_and_parameters() -> None:
    conn = _FakeConn()
    model = _FakeModel()

    results = retrieve_dense(
        "quy định hoãn thi",
        top_k=5,
        chunk_set="fixed_800_150",
        model_name="BAAI/bge-m3",
        conn=conn,
        model=model,
    )

    query_str = conn.cur.last_query
    assert "ORDER BY e.embedding <=> %s::vector" in query_str
    assert "LIMIT %s" in query_str
    assert "WHERE c.chunk_set = %s" in query_str

    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert results[0].article == "Điều 5"
    assert results[0].score == 0.88


class _FakeReranker:
    def predict(self, pairs: Sequence[list[str]], **kwargs) -> list[float]:
        # Return descending score for whatever has "khóa luận"
        return [0.95 if "khóa luận" in p[1] else 0.15 for p in pairs]


def test_reranker_reordering() -> None:
    chunks = [
        _make_chunk("c1", "Quy định về thời gian thi học kỳ"),
        _make_chunk("c2", "Tiêu chuẩn đăng ký thực hiện khóa luận tốt nghiệp"),
    ]
    mock_reranker = _FakeReranker()

    reranked = rerank("điều kiện làm khóa luận", chunks, model=mock_reranker)
    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].score == 0.95
    assert reranked[1].chunk_id == "c1"
    assert reranked[1].score == 0.15


def test_reranker_disabled_preserves_order() -> None:
    chunks = [
        _make_chunk("c1", "First chunk", score=0.8),
        _make_chunk("c2", "Second chunk", score=0.7),
    ]
    unranked = rerank("query", chunks, enabled=False)
    assert [c.chunk_id for c in unranked] == ["c1", "c2"]
