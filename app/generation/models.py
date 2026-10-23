"""Pydantic data models for generation, routing, and guardrails."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Intent(str, Enum):
    """User query classification intent."""

    SMALL_TALK = "SMALL_TALK"
    UIT_REGULATION = "UIT_REGULATION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    AMBIGUOUS = "AMBIGUOUS"


class RouterDecision(BaseModel):
    """Output of the router and query rewriter step."""

    intent: Intent
    rewritten_query: str = ""
    explanation: str = ""
    direct_response: str | None = None


class Citation(BaseModel):
    """Structured citation referencing a specific regulation source."""

    source_file: str
    article: str | None = None
    page: int | None = None
    quote: str | None = None


class ChatRequest(BaseModel):
    """Request payload for chat endpoints."""

    model_config = ConfigDict(populate_by_name=True)

    message: str = Field(..., validation_alias="question", description="User query or message")
    session_id: str | None = None
    history: list[dict[str, str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Response payload for chat endpoints."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Simplified sources list for legacy UI compatibility",
    )
    intent: str
    rewritten_query: str | None = None
    sufficient: bool = True
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    """User feedback for a generated response."""

    session_id: str | None = None
    message_id: str | None = None
    rating: int = Field(..., description="1 for positive, -1 for negative")
    comment: str | None = None
