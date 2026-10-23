"""CLI tool to inspect retrieval modes: Dense, BM25, Hybrid, Rerank."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from app.config import settings
from app.ingestion.chunkers import default_chunk_set
from app.logging_setup import setup_logging
from app.retrieval.bm25 import retrieve_bm25
from app.retrieval.dense import retrieve_dense
from app.retrieval.hybrid import retrieve_hybrid
from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import rerank


def _format_chunk(rank: int, chunk: RetrievedChunk) -> str:
    article_part = (
        f"{chunk.article}: {chunk.article_title}" if chunk.article else "Chưa định danh Điều"
    )
    page_part = (
        f"Trang {chunk.page_start}"
        if chunk.page_start == chunk.page_end
        else f"Trang {chunk.page_start}-{chunk.page_end}"
    )
    snippet = chunk.text.replace("\n", " ").strip()
    if len(snippet) > 180:
        snippet = snippet[:180] + "..."
    return (
        f"  [{rank}] Score: {chunk.score:.4f} | {chunk.source_file} ({page_part})\n"
        f"      Mục: {article_part}\n"
        f"      Đoạn trích: {snippet}"
    )


def _print_section(title: str, chunks: Sequence[RetrievedChunk]) -> None:
    print(f"\n{'=' * 75}\n{title}\n{'=' * 75}")
    if not chunks:
        print("  (Không có kết quả)")
        return
    for rank, chunk in enumerate(chunks, start=1):
        print(_format_chunk(rank, chunk))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect UITBOT retrieval across Dense, BM25, Hybrid, and Rerank modes."
    )
    parser.add_argument("query", type=str, help="Câu hỏi tra cứu quy chế")
    parser.add_argument("--top-k", type=int, default=5, help="Số lượng kết quả in ra mỗi chế độ")
    parser.add_argument(
        "--candidates",
        type=int,
        default=settings.retrieval_candidate_count,
        help="Số lượng ứng viên trước khi fusion/rerank",
    )
    parser.add_argument(
        "--chunk-set",
        type=str,
        default=default_chunk_set(),
        help="Chunk set cần tra cứu",
    )
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        help="Bỏ qua bước CrossEncoder rerank",
    )

    args = parser.parse_args()
    setup_logging(debug=False)

    print(f"\n[QUERY]: '{args.query}'")
    print(f"[CONFIG]: chunk_set={args.chunk_set}, top_k={args.top_k}, candidates={args.candidates}")

    # 1. Dense retrieval
    dense_results = retrieve_dense(
        args.query,
        top_k=args.candidates,
        chunk_set=args.chunk_set,
    )
    _print_section("1. DENSE RETRIEVAL (BAAI/bge-m3 + pgvector HNSW)", dense_results[: args.top_k])

    # 2. BM25 retrieval
    bm25_results = retrieve_bm25(
        args.query,
        top_k=args.candidates,
        chunk_set=args.chunk_set,
    )
    _print_section("2. BM25 SPARSE RETRIEVAL (rank-bm25 + underthesea)", bm25_results[: args.top_k])

    # 3. Hybrid RRF
    hybrid_results = retrieve_hybrid(
        args.query,
        top_k=args.candidates,
        dense_results=dense_results,
        bm25_results=bm25_results,
        chunk_set=args.chunk_set,
    )
    _print_section(
        "3. HYBRID RETRIEVAL (Reciprocal Rank Fusion k=60)",
        hybrid_results[: args.top_k],
    )

    # 4. Rerank
    if not args.skip_rerank and settings.reranker_enabled:
        reranked_results = rerank(
            args.query,
            hybrid_results,
            top_k=args.top_k,
        )
        _print_section(
            f"4. HYBRID + RERANK ({settings.reranker_model_name})",
            reranked_results,
        )
    print("\n")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print('Usage: python -m app.retrieval.cli "câu hỏi"')
        sys.exit(1)
    main()
