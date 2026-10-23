"""Evidence Gate and Guardrails for UITBOT RAG system."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from app.config import settings
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


def check_evidence_gate(
    chunks: Sequence[RetrievedChunk],
    *,
    min_score: float | None = None,
) -> tuple[bool, str | None]:
    """Check if retrieved chunks satisfy the minimum evidence threshold.

    Returns:
        (is_sufficient, refusal_reason)
    """
    threshold = settings.evidence_min_score if min_score is None else min_score

    if not chunks:
        logger.info("Evidence gate: No candidate chunks retrieved.")
        return False, (
            "Rất tiếc, cơ sở dữ liệu quy chế hiện tại không có văn bản nào liên quan đến "
            "câu hỏi này. Bạn vui lòng kiểm tra lại câu hỏi hoặc liên hệ trực tiếp "
            "Phòng Đào tạo (phongdaotao@uit.edu.vn)."
        )

    top_chunk = chunks[0]
    if top_chunk.score < threshold:
        logger.info(
            "Evidence gate: Top chunk score %.4f is below minimum threshold %.4f",
            top_chunk.score,
            threshold,
        )
        return False, (
            "Hiện tại các quy chế, quy định học vụ đã ban hành của UIT không đề cập rõ ràng "
            "về trường hợp cụ thể này (điểm tin cậy đối chiếu thấp). "
            "Để tránh nhầm lẫn về quyền lợi học tập, bạn hãy liên hệ Văn phòng Khoa hoặc "
            "Phòng Đào tạo (phongdaotao@uit.edu.vn / OEP) để được giải đáp chính thức nhé."
        )

    return True, None
