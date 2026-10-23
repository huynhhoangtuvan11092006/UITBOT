"""Router and Query Rewriter component."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

from app.config import settings
from app.generation.models import Intent, RouterDecision

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """Bạn là bộ phân loại ý định (Router) và chuẩn hóa truy vấn (Query Rewriter)
cho UITBOT - trợ lý giải đáp quy chế đào tạo Trường Đại học Công nghệ Thông tin (ĐHQG-HCM - UIT).

Nhiệm vụ của bạn:
1. Phân loại câu hỏi của người dùng vào 1 trong 4 nhóm ý định (intent):
   - "SMALL_TALK": Chào hỏi, cảm ơn, hỏi bạn là ai, tạm biệt.
   - "UIT_REGULATION": Hỏi về quy chế đào tạo, thi cử, điểm số, học vụ, học bổng, tốt nghiệp,
     kỷ luật, bảo lưu, chuyển ngành, học cùng lúc hai chương trình... của UIT hoặc ĐHQG-HCM.
   - "OUT_OF_SCOPE": Các câu hỏi ngoài phạm vi học vụ UIT (thời tiết, code, tin tức showbiz...).
   - "AMBIGUOUS": Câu hỏi quá ngắn hoặc tối nghĩa (ví dụ: "cái đó", "điều 5", "sao vậy").

2. Nếu intent là "UIT_REGULATION":
   - Viết lại câu hỏi (rewritten_query) từ khẩu ngữ sinh viên sang thuật ngữ pháp quy UIT
     (bỏ "em", "mình", mở rộng từ đồng nghĩa: hoãn thi, vắng thi, điểm I, bảo lưu, kỷ luật...).
   - rewritten_query phải ngắn gọn, súc tích (10-25 từ), tối ưu cho BM25 và Vector search.

3. Nếu intent là "SMALL_TALK", "OUT_OF_SCOPE", hoặc "AMBIGUOUS":
   - direct_response: Cung cấp câu trả lời/phản hồi trực tiếp, thân thiện, lịch sự (tiếng Việt).
   - rewritten_query: để trống ("").

Định dạng trả về BẮT BUỘC là một JSON object với 4 trường:
{
  "intent": "SMALL_TALK" | "UIT_REGULATION" | "OUT_OF_SCOPE" | "AMBIGUOUS",
  "rewritten_query": "chuỗi truy vấn chuẩn hóa nếu là UIT_REGULATION, ngược lại để rỗng",
  "explanation": "giải thích ngắn gọn lý do phân loại",
  "direct_response": "câu trả lời trực tiếp nếu không phải UIT_REGULATION, ngược lại null"
}
"""

_GREETING_PATTERNS = re.compile(
    r"^(xin chào|chào|hello|hi|hey|cảm ơn|thank you|thanks|bạn là ai|tạm biệt)(\s.*)?$",
    re.IGNORECASE,
)


def _rule_based_fallback(query: str) -> RouterDecision:
    """Fallback classifier when LLM call is unavailable or fails."""
    normalized = query.strip()
    if _GREETING_PATTERNS.match(normalized):
        return RouterDecision(
            intent=Intent.SMALL_TALK,
            rewritten_query="",
            explanation="Rule-based matched greeting pattern",
            direct_response=(
                "Xin chào bạn! Mình là UITBOT, trợ lý giải đáp quy chế, quy định học vụ "
                "tại Trường Đại học Công nghệ Thông tin (ĐHQG-HCM). Bạn cần tìm hiểu thông tin gì?"
            ),
        )

    if len(normalized) < 4:
        return RouterDecision(
            intent=Intent.AMBIGUOUS,
            rewritten_query="",
            explanation="Rule-based query is too short",
            direct_response=(
                "Câu hỏi của bạn quá ngắn hoặc chưa rõ ràng. Bạn vui lòng mô tả chi tiết hơn nhé!"
            ),
        )

    # Default to UIT_REGULATION with original query
    return RouterDecision(
        intent=Intent.UIT_REGULATION,
        rewritten_query=normalized,
        explanation="Rule-based default regulation fallback",
        direct_response=None,
    )


def route_and_rewrite(
    query: str,
    *,
    client: OpenAI | None = None,
    model_name: str | None = None,
) -> RouterDecision:
    """Classify user intent and rewrite the query using LLM."""
    query_clean = query.strip()
    if not query_clean:
        return _rule_based_fallback(query_clean)

    api_key = settings.openai_api_key
    base_url = settings.openai_base_url
    model = model_name or settings.openai_model

    if not api_key and client is None:
        logger.warning("No OpenAI API key configured; falling back to rule-based routing.")
        return _rule_based_fallback(query_clean)

    try:
        llm_client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=settings.llm_timeout,
        )

        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query_clean},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        data: dict[str, Any] = json.loads(content)

        raw_intent = data.get("intent", "UIT_REGULATION").upper()
        try:
            intent = Intent(raw_intent)
        except ValueError:
            intent = Intent.UIT_REGULATION

        rewritten = data.get("rewritten_query", "").strip()
        explanation = data.get("explanation", "").strip()
        direct_response = data.get("direct_response")

        if intent == Intent.UIT_REGULATION and not rewritten:
            rewritten = query_clean

        return RouterDecision(
            intent=intent,
            rewritten_query=rewritten,
            explanation=explanation,
            direct_response=direct_response,
        )

    except Exception as exc:
        logger.warning("LLM routing failed (%s); using rule-based fallback.", exc)
        return _rule_based_fallback(query_clean)
