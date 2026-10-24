"""Evaluate Intent Classifier, Evidence Gate, and Guardrails."""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.generation.guardrails import check_evidence_gate  # noqa: E402
from app.generation.router import route_and_rewrite  # noqa: E402
from app.retrieval import retrieve_hybrid  # noqa: E402
from app.retrieval.reranker import rerank  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("uitbot.eval.guardrail")

# Labeled intent test dataset (40 diverse queries)
INTENT_TEST_SET: list[dict[str, str]] = [
    # SMALL_TALK (10)
    {"query": "Xin chào bạn", "expected_intent": "SMALL_TALK"},
    {"query": "Chào bot nhé", "expected_intent": "SMALL_TALK"},
    {"query": "Hello UITBOT", "expected_intent": "SMALL_TALK"},
    {"query": "Bạn là ai vậy?", "expected_intent": "SMALL_TALK"},
    {"query": "Cảm ơn bạn nhiều nha", "expected_intent": "SMALL_TALK"},
    {"query": "Thanks bot", "expected_intent": "SMALL_TALK"},
    {"query": "Tạm biệt nhé", "expected_intent": "SMALL_TALK"},
    {"query": "Chúc một ngày tốt lành", "expected_intent": "SMALL_TALK"},
    {"query": "Chào buổi sáng", "expected_intent": "SMALL_TALK"},
    {"query": "Rất vui được gặp bạn", "expected_intent": "SMALL_TALK"},
    # OUT_OF_SCOPE (10)
    {"query": "Thời tiết hôm nay ở Thủ Đức thế nào?", "expected_intent": "OUT_OF_SCOPE"},
    {"query": "Chỉ mình cách nấu món bún bò Huế ngon đi", "expected_intent": "OUT_OF_SCOPE"},
    {
        "query": "Viết giúp mình một đoạn script Python tính số Fibonacci",
        "expected_intent": "OUT_OF_SCOPE",
    },
    {"query": "Ai là người phát minh ra máy tính?", "expected_intent": "OUT_OF_SCOPE"},
    {"query": "Giá vàng hôm nay tăng hay giảm?", "expected_intent": "OUT_OF_SCOPE"},
    {"query": "Bộ phim điện ảnh nào đang hot ngoài rạp?", "expected_intent": "OUT_OF_SCOPE"},
    {
        "query": "Cách sửa lỗi crash màn hình xanh trên Windows 11",
        "expected_intent": "OUT_OF_SCOPE",
    },
    {"query": "Quán cà phê nào gần trường UIT view đẹp?", "expected_intent": "OUT_OF_SCOPE"},
    {"query": "Đội bóng nào vừa vô địch Cúp C1 Châu Âu?", "expected_intent": "OUT_OF_SCOPE"},
    {"query": "Giải thích định luật vạn vật hấp dẫn của Newton", "expected_intent": "OUT_OF_SCOPE"},
    # AMBIGUOUS (10)
    {"query": "sao vậy", "expected_intent": "AMBIGUOUS"},
    {"query": "cái đó", "expected_intent": "AMBIGUOUS"},
    {"query": "điều 5", "expected_intent": "AMBIGUOUS"},
    {"query": "thế còn cái kia?", "expected_intent": "AMBIGUOUS"},
    {"query": "hả", "expected_intent": "AMBIGUOUS"},
    {"query": "là sao ta", "expected_intent": "AMBIGUOUS"},
    {"query": "rồi sao nữa", "expected_intent": "AMBIGUOUS"},
    {"query": "vậy hả", "expected_intent": "AMBIGUOUS"},
    {"query": "thật không", "expected_intent": "AMBIGUOUS"},
    {"query": "ủa", "expected_intent": "AMBIGUOUS"},
    # UIT_REGULATION (10)
    {
        "query": "Một kỳ chính em được đăng ký tối đa bao nhiêu tín chỉ?",
        "expected_intent": "UIT_REGULATION",
    },
    {
        "query": "Điều kiện để được xét tốt nghiệp ra trường là gì?",
        "expected_intent": "UIT_REGULATION",
    },
    {"query": "Bị cảnh báo học vụ mức 2 thì xử lý thế nào?", "expected_intent": "UIT_REGULATION"},
    {
        "query": "Quy định xin hoãn thi kết thúc học phần vì lý do ốm đau?",
        "expected_intent": "UIT_REGULATION",
    },
    {
        "query": "Học phí một tín chỉ chương trình chuẩn là bao nhiêu?",
        "expected_intent": "UIT_REGULATION",
    },
    {
        "query": "Quy chế công nhận chứng chỉ tiếng Anh IELTS để xét tốt nghiệp",
        "expected_intent": "UIT_REGULATION",
    },
    {
        "query": "Thời gian tối đa để hoàn thành chương trình đào tạo chính quy?",
        "expected_intent": "UIT_REGULATION",
    },
    {
        "query": "Cách tính điểm trung bình tích lũy theo thang điểm 4",
        "expected_intent": "UIT_REGULATION",
    },
    {
        "query": "Thủ tục xin nghỉ học tạm thời bảo lưu kết quả học tập",
        "expected_intent": "UIT_REGULATION",
    },
    {
        "query": "Sinh viên bị điểm F có được thi lại không hay phải học lại?",
        "expected_intent": "UIT_REGULATION",
    },
]

# Evidence Gate Test Queries:
# 20 Answerable questions (should PASS evidence gate)
ANSWERABLE_QUERIES: list[str] = [
    "Sinh viên hệ chính quy được đăng ký tối đa bao nhiêu tín chỉ trong một học kỳ chính?",
    "Số tín chỉ tối thiểu mà sinh viên phải đăng ký trong một học kỳ chính là bao nhiêu?",
    "Học kỳ hè sinh viên được đăng ký tối đa bao nhiêu tín chỉ?",
    "Thời hạn xin rút bớt học phần trong học kỳ chính là khi nào?",
    "Học cải thiện điểm thì điểm cũ có bị xóa không hay lấy điểm cao hơn?",
    "Nếu bị điểm F môn bắt buộc thì sinh viên phải làm gì?",
    "Một tín chỉ lý thuyết tương đương bao nhiêu tiết học trên lớp?",
    "Điều kiện để sinh viên được công nhận tốt nghiệp đại học UIT là gì?",
    "Thang điểm đánh giá học phần gồm những thang điểm nào?",
    "Điểm chữ D tương ứng với thang điểm 4 là bao nhiêu?",
    "Sinh viên bị bao nhiêu lần cảnh báo học vụ thì bị buộc thôi học?",
    "Điều kiện để được xem xét chuyển ngành đào tạo là gì?",
    "Quy định về việc vắng thi có phép và không phép?",
    "Điểm I là điểm gì và thời hạn hoàn thành điểm I là bao lâu?",
    "Sinh viên năm nhất có được tạm dừng học tập không?",
    "Thời gian tối đa được phép tạm dừng học tập trong toàn khóa học là bao lâu?",
    "Quy định xếp loại tốt nghiệp loại Giỏi cần GPA bao nhiêu?",
    "Nếu bị kỷ luật từ mức nào thì sinh viên bị giảm một bậc xếp loại tốt nghiệp?",
    "Quy định chuyển điểm chứng chỉ tiếng Anh quốc tế sang điểm học phần Anh văn?",
    "Quy định về chuẩn đầu ra ngoại ngữ tốt nghiệp đối với sinh viên UIT?",
]

# 15 Reasonable-sounding Unanswerable / Out-of-corpus queries (should be REFUSED by evidence gate)
UNANSWERABLE_QUERIES: list[str] = [
    "Trường UIT có chính sách hỗ trợ vay vốn mua xe máy điện cho sinh viên không?",
    "Quy định về việc nuôi thú cưng chó mèo trong phòng ký túc xá khu B?",
    "Thực đơn cơm trưa căn tin trường UIT hôm nay có món gì ngon?",
    "Quy định lắp đặt điều hòa cá nhân trong giảng đường khu A?",
    "Sinh viên có được phép đi trực thăng riêng đến sân bóng rổ trường không?",
    "Mức phí gửi xe ô tô 4 chỗ qua đêm tại tầng hầm tòa E là bao nhiêu?",
    "Chính sách bảo hiểm y tế nha khoa bọc răng sứ cho sinh viên năm 3?",
    "Thủ tục đăng ký kết hôn tại phòng công tác sinh viên UIT?",
    "Quy định về trang phục hóa trang cosplay vào ngày lễ Halloween tại trường?",
    "Quy chế cấp học bổng du học sao Hỏa cho sinh viên ngành Khoa học máy tính?",
    "Quy định cho phép sinh viên mang bếp gas mini vào phòng tự học?",
    "Chính sách miễn học phí cho sinh viên có người yêu cùng trường?",
    "Quy định chuyển đổi bằng lái xe máy hạng A1 sang tín chỉ thể dục?",
    "Thủ tục mở sạp bán trà sữa khởi nghiệp tại sảnh tòa C?",
    "Quy định về thời gian đóng cổng trường không cho người ngoài vào đá cầu ban đêm?",
]


@dataclass
class IntentEvalSummary:
    total: int
    correct: int
    accuracy: float
    per_class_accuracy: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]


@dataclass
class EvidenceGateSweepPoint:
    threshold: float
    true_refusals: int
    false_accepts: int
    true_accepts: int
    false_refusals: int
    refusal_precision: float
    refusal_recall: float
    refusal_f1: float
    pass_accuracy: float


def evaluate_intents() -> IntentEvalSummary:
    """Evaluate router classification across test queries."""
    total = len(INTENT_TEST_SET)
    correct = 0
    categories = ["SMALL_TALK", "OUT_OF_SCOPE", "AMBIGUOUS", "UIT_REGULATION"]
    confusion: dict[str, dict[str, int]] = {
        cat: {pred: 0 for pred in categories} for cat in categories
    }
    class_total = {cat: 0 for cat in categories}
    class_correct = {cat: 0 for cat in categories}

    for item in INTENT_TEST_SET:
        query = item["query"]
        expected = item["expected_intent"]
        class_total[expected] += 1
        decision = route_and_rewrite(query)
        actual = decision.intent.value

        # Fallback to category if unexpected value
        if actual not in categories:
            actual = "UIT_REGULATION"

        confusion[expected][actual] += 1
        if actual == expected:
            correct += 1
            class_correct[expected] += 1

    per_class = {cat: round(class_correct[cat] / max(class_total[cat], 1), 4) for cat in categories}

    return IntentEvalSummary(
        total=total,
        correct=correct,
        accuracy=round(correct / total, 4),
        per_class_accuracy=per_class,
        confusion_matrix=confusion,
    )


def evaluate_evidence_gate(
    threshold_list: list[float] | None = None,
) -> tuple[dict[str, Any], list[EvidenceGateSweepPoint]]:
    """Evaluate Evidence Gate with answerable vs unanswerable queries."""
    thresholds = threshold_list or [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]

    # Pre-retrieve top chunks and rerank scores for all test queries
    logger.info("Pre-retrieving candidate chunks for answerable queries...")
    answerable_chunks: list[list[Any]] = []
    for q in ANSWERABLE_QUERIES:
        candidates = retrieve_hybrid(q, top_k=5)
        reranked = rerank(q, candidates, top_k=5)
        answerable_chunks.append(reranked)

    logger.info("Pre-retrieving candidate chunks for unanswerable queries...")
    unanswerable_chunks: list[list[Any]] = []
    for q in UNANSWERABLE_QUERIES:
        candidates = retrieve_hybrid(q, top_k=5)
        reranked = rerank(q, candidates, top_k=5)
        unanswerable_chunks.append(reranked)

    sweep_results: list[EvidenceGateSweepPoint] = []

    for thresh in thresholds:
        # Evaluate unanswerable: expectation is REFUSAL (is_sufficient == False)
        true_refusals = 0
        false_accepts = 0
        for chunks in unanswerable_chunks:
            is_sufficient, _ = check_evidence_gate(chunks, min_score=thresh)
            if not is_sufficient:
                true_refusals += 1
            else:
                false_accepts += 1

        # Evaluate answerable: expectation is ACCEPT (is_sufficient == True)
        true_accepts = 0
        false_refusals = 0
        for chunks in answerable_chunks:
            is_sufficient, _ = check_evidence_gate(chunks, min_score=thresh)
            if is_sufficient:
                true_accepts += 1
            else:
                false_refusals += 1

        # Refusal metrics (Refusal = Positive)
        prec = true_refusals / max(true_refusals + false_refusals, 1)
        rec = true_refusals / max(true_refusals + false_accepts, 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-9)
        total_q = len(ANSWERABLE_QUERIES) + len(UNANSWERABLE_QUERIES)
        pass_acc = (true_accepts + true_refusals) / total_q

        sweep_results.append(
            EvidenceGateSweepPoint(
                threshold=thresh,
                true_refusals=true_refusals,
                false_accepts=false_accepts,
                true_accepts=true_accepts,
                false_refusals=false_refusals,
                refusal_precision=round(prec, 4),
                refusal_recall=round(rec, 4),
                refusal_f1=round(f1, 4),
                pass_accuracy=round(pass_acc, 4),
            )
        )

    # Optimal threshold selection: maximize F1 and accuracy
    best_point = max(sweep_results, key=lambda p: (p.refusal_f1, p.pass_accuracy))
    current_setting = settings.evidence_min_score

    summary = {
        "configured_threshold": current_setting,
        "optimal_threshold": best_point.threshold,
        "num_answerable": len(ANSWERABLE_QUERIES),
        "num_unanswerable": len(UNANSWERABLE_QUERIES),
        "best_refusal_f1": best_point.refusal_f1,
        "best_pass_accuracy": best_point.pass_accuracy,
    }

    return summary, sweep_results


def run_all_guardrail_evals() -> dict[str, Any]:
    """Run intent eval + evidence gate sweep and return comprehensive dict."""
    logger.info("Evaluating Router Intent Classification (%d queries)...", len(INTENT_TEST_SET))
    intent_summary = evaluate_intents()

    total_gate_q = len(ANSWERABLE_QUERIES) + len(UNANSWERABLE_QUERIES)
    logger.info("Evaluating Evidence Gate Guardrails (%d queries)...", total_gate_q)
    gate_summary, sweep_points = evaluate_evidence_gate()

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "intent_evaluation": asdict(intent_summary),
        "evidence_gate_summary": gate_summary,
        "evidence_gate_sweep": [asdict(p) for p in sweep_points],
    }

    out_file = Path("evaluation/guardrail_results.json")
    out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Guardrail evaluation complete. Saved to %s", out_file)
    return output


if __name__ == "__main__":
    run_all_guardrail_evals()
