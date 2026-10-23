"""Orchestrates the complete RAG pipeline.
Flow: Routing -> Retrieval -> Evidence Gate -> Generation -> Citation Validation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.config import settings
from app.generation.answer import generate_answer
from app.generation.citations import format_citations_markdown, validate_citations
from app.generation.guardrails import check_evidence_gate
from app.generation.models import ChatResponse, Citation, Intent
from app.generation.router import route_and_rewrite
from app.retrieval.bm25 import retrieve_bm25
from app.retrieval.dense import retrieve_dense
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import rerank

logger = logging.getLogger(__name__)

_LANGFUSE_CLIENT = None


def get_langfuse_client():
    """Lazy initialize Langfuse client if enabled."""
    global _LANGFUSE_CLIENT
    if not settings.langfuse_enabled or not settings.langfuse_secret_key:
        return None
    if _LANGFUSE_CLIENT is None:
        try:
            from langfuse import Langfuse

            _LANGFUSE_CLIENT = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_base_url,
            )
            logger.info("Langfuse tracing client initialized.")
        except Exception as e:
            logger.warning("Could not initialize Langfuse: %s", e)
            return None
    return _LANGFUSE_CLIENT


def format_ui_sources(citations: Sequence[Citation]) -> list[dict[str, Any]]:
    """Format verified citations for web UI compatibility."""
    return [
        {
            "file": c.source_file,
            "article": c.article or "",
            "page": c.page or 1,
            "quote": c.quote or "",
        }
        for c in citations
    ]


class RAGService:
    """End-to-end RAG service with routing and evidence guardrails."""

    def __init__(self, *, conn=None) -> None:
        self.conn = conn

    def answer_query(
        self,
        query: str,
        *,
        session_id: str | None = None,
        top_k: int | None = None,
        evidence_min_score: float | None = None,
    ) -> ChatResponse:
        """Process user query through the full pipeline."""
        clean_query = query.strip()
        top_k = top_k or settings.rag_top_k
        min_score = (
            evidence_min_score if evidence_min_score is not None else settings.evidence_min_score
        )

        langfuse = get_langfuse_client()
        trace = None
        if langfuse is not None:
            try:
                trace = langfuse.trace(
                    name="uitbot_rag_pipeline",
                    session_id=session_id,
                    input={"query": clean_query},
                )
            except Exception as e:
                logger.debug("Failed to start Langfuse trace: %s", e)

        # Step 1: Routing & Query Rewriting
        decision = route_and_rewrite(clean_query)
        logger.info(
            "Query '%s' routed to intent=%s (rewritten: '%s')",
            clean_query,
            decision.intent.value,
            decision.rewritten_query,
        )

        if trace:
            try:
                trace.span(
                    name="routing",
                    input={"query": clean_query},
                    output=decision.model_dump(),
                )
            except Exception:
                pass

        # Handle non-regulation intents immediately
        if decision.intent != Intent.UIT_REGULATION:
            direct = decision.direct_response or "Chào bạn, mình có thể giúp gì về quy chế UIT?"
            if trace:
                try:
                    trace.update(output={"answer": direct, "intent": decision.intent.value})
                except Exception:
                    pass
            return ChatResponse(
                answer=direct,
                citations=[],
                sources=[],
                intent=decision.intent.value,
                rewritten_query=decision.rewritten_query,
                sufficient=True,
                retrieved_chunks=[],
            )

        # Step 2: Retrieval with rewritten query
        search_query = decision.rewritten_query or clean_query
        candidate_count = settings.retrieval_candidate_count

        dense_candidates = retrieve_dense(
            search_query,
            top_k=candidate_count,
            conn=self.conn,
        )
        bm25_candidates = retrieve_bm25(
            search_query,
            top_k=candidate_count,
            conn=self.conn,
        )

        fused_candidates = reciprocal_rank_fusion(
            [dense_candidates, bm25_candidates],
            k=settings.rrf_k,
            top_k=candidate_count,
        )

        reranked_chunks: list[RetrievedChunk] = rerank(
            search_query,
            fused_candidates,
            top_k=top_k,
        )

        # Step 3: Evidence Gate
        is_sufficient, refusal_reason = check_evidence_gate(reranked_chunks, min_score=min_score)

        if not is_sufficient and refusal_reason:
            refusal_answer = refusal_reason
            if trace:
                try:
                    trace.update(
                        output={"answer": refusal_answer, "sufficient": False, "gated": True}
                    )
                except Exception:
                    pass
            return ChatResponse(
                answer=refusal_answer,
                citations=[],
                sources=[],
                intent=decision.intent.value,
                rewritten_query=decision.rewritten_query,
                sufficient=False,
                retrieved_chunks=[
                    {
                        "chunk_id": c.chunk_id,
                        "source_file": Path(c.source_file).name,
                        "article": c.article,
                        "score": round(c.score, 4),
                        "page": c.page_start,
                    }
                    for c in reranked_chunks
                ],
            )

        # Step 4: Answer Generation
        answer_text, raw_citations, gen_sufficient = generate_answer(
            clean_query,
            reranked_chunks,
        )

        # Step 5: Citation Validation
        verified_citations = validate_citations(raw_citations, reranked_chunks)

        # If LLM didn't attach citation markdown, append verified citations footnote
        if verified_citations and "**Tài liệu trích dẫn:**" not in answer_text:
            answer_text += format_citations_markdown(verified_citations)

        ui_sources = format_ui_sources(verified_citations)

        chunk_summaries = [
            {
                "chunk_id": c.chunk_id,
                "source_file": Path(c.source_file).name,
                "article": c.article,
                "score": round(c.score, 4),
                "page": c.page_start,
            }
            for c in reranked_chunks
        ]

        if trace:
            try:
                trace.update(
                    output={
                        "answer": answer_text,
                        "citations": [c.model_dump() for c in verified_citations],
                        "sufficient": gen_sufficient,
                    }
                )
            except Exception:
                pass

        return ChatResponse(
            answer=answer_text,
            citations=verified_citations,
            sources=ui_sources,
            intent=decision.intent.value,
            rewritten_query=decision.rewritten_query,
            sufficient=gen_sufficient,
            retrieved_chunks=chunk_summaries,
        )


rag_service = RAGService()
