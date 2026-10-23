"""Retrieval pipeline for UITBOT."""

from __future__ import annotations

from app.retrieval.bm25 import get_bm25_index, rebuild, retrieve_bm25
from app.retrieval.dense import retrieve_dense
from app.retrieval.hybrid import reciprocal_rank_fusion, retrieve_hybrid
from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import load_reranker_model, rerank

__all__ = [
    "RetrievedChunk",
    "get_bm25_index",
    "load_reranker_model",
    "rebuild",
    "reciprocal_rank_fusion",
    "rerank",
    "retrieve_bm25",
    "retrieve_dense",
    "retrieve_hybrid",
]
