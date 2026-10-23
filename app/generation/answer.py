"""Answer generation module with anti-hallucination guardrails."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.config import settings
from app.generation.models import Citation
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Bạn là trợ lý giải đáp quy chế đào tạo UITBOT
của Trường Đại học Công nghệ Thông tin (ĐHQG-HCM).

Nhiệm vụ của bạn:
1. Trả lời câu hỏi dựa HOÀN TOÀN và DUY NHẤT vào các đoạn trích quy chế (Ngữ cảnh) được cung cấp.
2. TUYỆT ĐỐI KHÔNG tự suy diễn, bịa đặt hoặc sử dụng thông tin ngoài ngữ cảnh được cung cấp.
3. Nếu ngữ cảnh KHÔNG đủ thông tin để trả lời câu hỏi hoặc thông tin không chắc chắn:
   - Đặt "sufficient": false.
   - Trong "answer", nêu rõ thông tin trong quy chế hiện tại không đề cập đầy đủ vấn đề này và
     hướng dẫn sinh viên liên hệ Phòng Đào tạo (phongdaotao@uit.edu.vn) hoặc OEP để được hỗ trợ.
4. Trích dẫn chính xác: Mỗi ý trong câu trả lời cần liên kết với tài liệu tham chiếu.
5. Trình bày tiếng Việt chuẩn mực, mạch lạc, chia gạch đầu dòng rõ ràng để sinh viên dễ đọc.

BẮT BUỘC trả về định dạng JSON object:
{
  "answer": "Nội dung trả lời chi tiết bằng tiếng Việt...",
  "citations": [
    {
      "source_file": "tên_file.pdf",
      "article": "Điều X (nếu có)",
      "page": 1,
      "quote": "trích đoạn ngắn <= 200 ký tự nguyên văn từ ngữ cảnh"
    }
  ],
  "sufficient": true | false
}
"""


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Format retrieved chunks into numbered context blocks for the LLM prompt."""
    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, 1):
        filename = Path(chunk.source_file).name
        article_info = chunk.article or "Chưa định danh Điều"
        if chunk.article_title:
            article_info += f" ({chunk.article_title})"
        page_info = (
            f"Trang {chunk.page_start}"
            if chunk.page_start == chunk.page_end
            else f"Trang {chunk.page_start}-{chunk.page_end}"
        )

        block = (
            f"--- [TÀI LIỆU {idx}] ---\n"
            f"File: {filename}\n"
            f"Mục: {article_info}\n"
            f"Vị trí: {page_info}\n"
            f"Nội dung:\n{chunk.text.strip()}\n"
        )
        blocks.append(block)

    return "\n".join(blocks)


def _extractive_fallback_answer(
    chunks: Sequence[RetrievedChunk],
) -> tuple[str, list[Citation], bool]:
    """Extractive grounded fallback when LLM is offline or fails."""
    if not chunks:
        return (
            "Rất tiếc, hệ thống không tìm thấy quy định phù hợp trong cơ sở dữ liệu. "
            "Bạn vui lòng liên hệ Phòng Đào tạo (phongdaotao@uit.edu.vn) để được hỗ trợ.",
            [],
            False,
        )

    top = chunks[0]
    filename = Path(top.source_file).name
    article = top.article or "Quy định liên quan"

    answer = (
        f"Theo {filename} ({article}, Trang {top.page_start}):\n\n"
        f"{top.text[:600]}...\n\n"
        "*(Lưu ý: Đây là đoạn trích trực tiếp từ văn bản quy chế UIT)*"
    )

    citation = Citation(
        source_file=filename,
        article=top.article,
        page=top.page_start,
        quote=top.text[:150],
    )
    return answer, [citation], True


def generate_answer(
    query: str,
    chunks: Sequence[RetrievedChunk],
    *,
    client: OpenAI | None = None,
    model_name: str | None = None,
) -> tuple[str, list[Citation], bool]:
    """Generate a verified answer grounded strictly in retrieved chunks.

    Returns:
        tuple of (answer_text, raw_citations, sufficient)
    """
    if not chunks:
        return _extractive_fallback_answer(chunks)

    api_key = settings.openai_api_key
    base_url = settings.openai_base_url
    model = model_name or settings.openai_model

    if not api_key and client is None:
        logger.warning("No OpenAI API key; using extractive fallback.")
        return _extractive_fallback_answer(chunks)

    context_str = format_context(chunks)
    user_prompt = (
        f"Câu hỏi của sinh viên:\n{query}\n\n"
        f"Ngữ cảnh các điều khoản quy chế UIT trích xuất được:\n{context_str}\n\n"
        "Hãy đọc kỹ ngữ cảnh trên và trả lời câu hỏi theo đúng định dạng JSON yêu cầu."
    )

    try:
        llm_client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=settings.llm_timeout,
        )

        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.llm_temperature,
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content or "{}"
        data: dict[str, Any] = json.loads(raw_content)

        answer = data.get("answer", "").strip()
        sufficient = bool(data.get("sufficient", True))

        raw_citations_data = data.get("citations", [])
        citations: list[Citation] = []
        for c in raw_citations_data:
            if isinstance(c, dict):
                citations.append(
                    Citation(
                        source_file=c.get("source_file", ""),
                        article=c.get("article"),
                        page=c.get("page"),
                        quote=c.get("quote"),
                    )
                )

        if not answer:
            return _extractive_fallback_answer(chunks)

        return answer, citations, sufficient

    except Exception as exc:
        logger.warning("Answer generation failed (%s); using extractive fallback.", exc)
        return _extractive_fallback_answer(chunks)
