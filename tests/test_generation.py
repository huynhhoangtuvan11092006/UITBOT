"""Tests for Generation, Router, Guardrails, and Citations Validator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from app.generation.answer import _extractive_fallback_answer, format_context, generate_answer
from app.generation.citations import (
    format_citations_markdown,
    normalize_source_name,
    validate_citations,
)
from app.generation.guardrails import check_evidence_gate
from app.generation.models import Citation, Intent
from app.generation.router import _rule_based_fallback, route_and_rewrite
from app.main import app
from app.retrieval.models import RetrievedChunk
from fastapi.testclient import TestClient


def _sample_chunk(
    chunk_id: int = 1,
    source_file: str = "quy_che_dao_tao.pdf",
    page_start: int = 19,
    page_end: int = 19,
    article: str = "Điều 22",
    score: float = 0.85,
    text: str = (
        "Điều 22. Không hoàn tất học phần. Sinh viên ốm đau tai nạn được hoãn thi và nhận điểm I."
    ),
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_file=source_file,
        page_start=page_start,
        page_end=page_end,
        chapter="Chương IV",
        article=article,
        article_title="Không hoàn tất học phần",
        text=text,
        score=score,
    )


# ---------------------------------------------------------
# 1. Router Tests
# ---------------------------------------------------------
def test_router_rule_based_greetings():
    res = _rule_based_fallback("Xin chào")
    assert res.intent == Intent.SMALL_TALK
    assert "UITBOT" in (res.direct_response or "")

    res_hi = _rule_based_fallback("hello")
    assert res_hi.intent == Intent.SMALL_TALK


def test_router_rule_based_ambiguous():
    res = _rule_based_fallback("ừ")
    assert res.intent == Intent.AMBIGUOUS


def test_router_rule_based_regulation_default():
    res = _rule_based_fallback("điều kiện xét tốt nghiệp đại học")
    assert res.intent == Intent.UIT_REGULATION
    assert res.rewritten_query == "điều kiện xét tốt nghiệp đại học"


def test_router_with_mock_llm():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = (
        '{"intent": "UIT_REGULATION", "rewritten_query": "quy định hoãn thi ốm đau tai nạn điểm I",'
        ' "explanation": "Hỏi về hoãn thi", "direct_response": null}'
    )
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    res = route_and_rewrite(
        "em bị ốm nằm viện ko thi được làm sao ạ",
        client=mock_client,
        model_name="dummy-model",
    )
    assert res.intent == Intent.UIT_REGULATION
    assert "hoãn thi" in res.rewritten_query


# ---------------------------------------------------------
# 2. Citations Validator Tests
# ---------------------------------------------------------
def test_normalize_source_name():
    assert normalize_source_name("data/raw/quy_che.pdf") == "quy_che.pdf"
    assert normalize_source_name("QUY_CHE.PDF") == "quy_che.pdf"


def test_validate_citations_success():
    chunk = _sample_chunk()
    raw = [
        Citation(
            source_file="quy_che_dao_tao.pdf",
            article="Điều 22",
            page=19,
            quote="Sinh viên ốm đau tai nạn được hoãn thi",
        )
    ]
    verified = validate_citations(raw, [chunk])
    assert len(verified) == 1
    assert verified[0].source_file == "quy_che_dao_tao.pdf"
    assert verified[0].article == "Điều 22"
    assert verified[0].page == 19


def test_validate_citations_prunes_hallucinations():
    chunk = _sample_chunk()
    fake = [
        # Hallucinated file
        Citation(source_file="fake_document.pdf", article="Điều 1", page=1),
        # Real file but completely wrong page
        Citation(source_file="quy_che_dao_tao.pdf", article="Điều 22", page=999),
    ]
    verified = validate_citations(fake, [chunk])
    assert len(verified) == 0


def test_format_citations_markdown():
    c = Citation(source_file="quy_che.pdf", article="Điều 10", page=5)
    md = format_citations_markdown([c])
    assert "**Tài liệu trích dẫn:**" in md
    assert "[1] quy_che.pdf – Điều 10 – Trang 5" in md


# ---------------------------------------------------------
# 3. Guardrails & Evidence Gate Tests
# ---------------------------------------------------------
def test_evidence_gate_empty():
    ok, refusal = check_evidence_gate([])
    assert ok is False
    assert "không có văn bản nào" in (refusal or "")


def test_evidence_gate_below_threshold():
    weak_chunk = _sample_chunk(score=0.005)
    ok, refusal = check_evidence_gate([weak_chunk], min_score=0.02)
    assert ok is False
    assert "không đề cập rõ ràng" in (refusal or "")


def test_evidence_gate_above_threshold():
    good_chunk = _sample_chunk(score=0.055)
    ok, refusal = check_evidence_gate([good_chunk], min_score=0.02)
    assert ok is True
    assert refusal is None


# ---------------------------------------------------------
# 4. Answer Generation Tests
# ---------------------------------------------------------
def test_format_context():
    chunk = _sample_chunk()
    ctx = format_context([chunk])
    assert "File: quy_che_dao_tao.pdf" in ctx
    assert "Mục: Điều 22" in ctx
    assert "Trang 19" in ctx


def test_extractive_fallback_answer():
    chunk = _sample_chunk()
    ans, citations, ok = _extractive_fallback_answer([chunk])
    assert "Điều 22" in ans
    assert len(citations) == 1
    assert ok is True


def test_generate_answer_mock_llm():
    chunk = _sample_chunk()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = (
        '{"answer": "Theo Điều 22, sinh viên ốm đau được hoãn thi.", '
        '"citations": [{"source_file": "quy_che_dao_tao.pdf", "article": "Điều 22", "page": 19}], '
        '"sufficient": true}'
    )
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    ans, citations, ok = generate_answer(
        "quy chế hoãn thi",
        [chunk],
        client=mock_client,
        model_name="dummy-model",
    )
    assert "hoãn thi" in ans
    assert len(citations) == 1
    assert ok is True


# ---------------------------------------------------------
# 5. FastAPI Endpoints Integration Tests
# ---------------------------------------------------------
@pytest.fixture
def client():
    return TestClient(app)


def test_chat_endpoint_empty_query(client):
    res = client.post("/api/v1/chat", json={"question": ""})
    assert res.status_code == 400


def test_chat_endpoint_small_talk(client, monkeypatch):
    # Ensure router classifies greeting
    res = client.post("/api/v1/chat", json={"question": "Xin chào UITBOT"})
    assert res.status_code == 200
    data = res.json()
    assert data["intent"] == "SMALL_TALK"
    assert "UITBOT" in data["answer"]
    assert len(data["citations"]) == 0


def test_feedback_endpoint(client):
    res = client.post(
        "/api/v1/feedback",
        json={"rating": 1, "comment": "Câu trả lời rất chính xác!"},
    )
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
