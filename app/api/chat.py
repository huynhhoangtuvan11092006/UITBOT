"""Chat API endpoints for UITBOT."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.generation.models import ChatRequest, ChatResponse, FeedbackRequest
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Synchronous chat endpoint compatible with UIT web UI."""
    query = request.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
    if len(query) > 1000:
        raise HTTPException(status_code=400, detail="Câu hỏi quá dài (tối đa 1000 ký tự).")

    try:
        response = rag_service.answer_query(
            query,
            session_id=request.session_id,
        )
        return response
    except Exception as exc:
        logger.exception("Error processing chat query: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý hệ thống: {exc}") from exc


async def _stream_chat_generator(query: str, session_id: str | None) -> AsyncIterator[str]:
    """Stream response events using Server-Sent Events (SSE)."""
    try:
        # Step 1: Send processing event
        status_payload = json.dumps({"message": "Đang phân tích câu hỏi..."}, ensure_ascii=False)
        yield f"event: status\ndata: {status_payload}\n\n"

        # Generate answer through RAG pipeline
        response = rag_service.answer_query(query, session_id=session_id)

        # Step 2: Send routing and sources event
        meta_payload = json.dumps(
            {
                "intent": response.intent,
                "sources": response.sources,
                "rewritten_query": response.rewritten_query,
            },
            ensure_ascii=False,
        )
        yield f"event: meta\ndata: {meta_payload}\n\n"

        # Step 3: Stream answer content
        words = response.answer.split(" ")
        chunk_size = 4
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size]) + " "
            yield f"event: token\ndata: {json.dumps({'token': chunk}, ensure_ascii=False)}\n\n"

        # Step 4: Done event
        yield f"event: done\ndata: {json.dumps(response.model_dump(), ensure_ascii=False)}\n\n"

    except Exception as exc:
        logger.exception("Streaming error: %s", exc)
        yield f"event: error\ndata: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, req: Request) -> StreamingResponse:
    """Server-Sent Events (SSE) streaming chat endpoint."""
    query = request.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
    if len(query) > 1000:
        raise HTTPException(status_code=400, detail="Câu hỏi quá dài (tối đa 1000 ký tự).")

    return StreamingResponse(
        _stream_chat_generator(query, request.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/feedback")
async def feedback_endpoint(feedback: FeedbackRequest) -> dict[str, str]:
    """Capture user feedback for continuous improvement."""
    logger.info(
        "User feedback received: rating=%d session=%s comment=%s",
        feedback.rating,
        feedback.session_id,
        feedback.comment,
    )
    return {"status": "ok", "message": "Cảm ơn bạn đã gửi phản hồi!"}
