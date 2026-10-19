
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.rag_service import RAGService

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Câu hỏi về quy chế UIT",
    )


class Source(BaseModel):
    file: str
    page: int
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@lru_cache(maxsize=1)
def get_rag_service():
    return RAGService()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Câu hỏi không được để trống.",
        )

    try:
        service = get_rag_service()
        return service.answer(question)

    except Exception:
        logger.exception("UITBOT chat failed")

        raise HTTPException(
            status_code=502,
            detail="Không thể xử lý câu hỏi hiện tại.",
        )
