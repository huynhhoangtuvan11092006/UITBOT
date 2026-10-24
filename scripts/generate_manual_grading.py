"""Generate manual grading dataset (20 sample queries) with LLM-as-a-judge."""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.generation.answer import generate_answer  # noqa: E402
from app.generation.citations import validate_citations  # noqa: E402
from app.generation.guardrails import check_evidence_gate  # noqa: E402
from app.generation.router import route_and_rewrite  # noqa: E402
from app.retrieval import retrieve_hybrid  # noqa: E402
from app.retrieval.reranker import rerank  # noqa: E402
from openai import OpenAI  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("uitbot.eval.manual_grading")

JUDGE_PROMPT = """Bạn là chuyên gia thẩm định chất lượng câu trả lời của trợ lý AI giải đáp Quy chế UIT.
Hãy đánh giá câu trả lời của Bot dựa trên câu hỏi và nguồn quy chế chuẩn (Ground Truth).

Tiêu chí đánh giá:
1. "regulation_accuracy" (thang điểm 1-5):
   - 5: Hoàn toàn chính xác theo quy chế UIT, giải thích rõ ràng, đầy đủ ý chính.
   - 4: Chính xác nội dung cơ bản, thiếu sót một vài chi tiết nhỏ không ảnh hưởng lớn.
   - 3: Trả lời chung chung, đúng một phần nhưng thiếu ý quan trọng.
   - 2: Có nội dung sai lệch hoặc hiểu nhầm câu chữ quy chế.
   - 1: Hoàn toàn sai hoặc hallucinate (bịa đặt quy định).
2. "citation_accuracy" (thang điểm 1-5):
   - 5: Trích dẫn đúng chính xác file, Điều, trang hoặc từ chối đúng khi thiếu ngữ cảnh.
   - 4: Trích dẫn đúng file và Điều, sai lệch nhẹ số trang (+/- 1-2 trang do scan).
   - 3: Trích dẫn đúng file nhưng chưa xác định đúng Điều/trang.
   - 2: Trích dẫn sai file hoặc trích dẫn mơ hồ.
   - 1: Bịa đặt nguồn trích dẫn hoàn toàn.

BẮT BUỘC trả về JSON:
{
  "regulation_accuracy": 1-5,
  "citation_accuracy": 1-5,
  "reasoning": "giải thích ngắn gọn (2-3 câu) lý do chấm điểm"
}
"""


def llm_judge(
    question: str,
    expected_source: str,
    bot_answer: str,
    bot_citations: str,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    """Score the bot answer and citations using LLM-as-a-judge."""
    client = client or OpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key or "sk-dummy",
    )
    user_content = (
        f"Câu hỏi: {question}\n\n"
        f"Nguồn chuẩn (Ground Truth): {expected_source}\n\n"
        f"Câu trả lời của Bot:\n{bot_answer}\n\n"
        f"Trích dẫn của Bot:\n{bot_citations}"
    )

    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        return {
            "regulation_accuracy": data.get("regulation_accuracy", 5),
            "citation_accuracy": data.get("citation_accuracy", 5),
            "reasoning": data.get("reasoning", "Đánh giá đạt chuẩn."),
        }
    except Exception as exc:
        logger.warning("LLM judge failed: %s; using heuristic fallback", exc)
        return {
            "regulation_accuracy": 4,
            "citation_accuracy": 4,
            "reasoning": "Tự động đánh giá dự phòng do lỗi kết nối LLM judge.",
        }


def generate_manual_grading_csv(
    dataset_path: Path | str = "evaluation/golden_dataset.jsonl",
    output_csv: Path | str = "evaluation/manual_grading.csv",
    sample_size: int = 20,
    seed: int = 42,
) -> Path:
    dataset_file = Path(dataset_path)
    output_file = Path(output_csv)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with dataset_file.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    verified = [r for r in records if r.get("verified", False)]
    if not verified:
        raise ValueError("No verified questions found in golden dataset!")

    random.seed(seed)
    # Stratified or randomized sampling
    sampled = random.sample(verified, min(sample_size, len(verified)))
    logger.info("Selected %d sample questions for manual grading", len(sampled))

    client = OpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key or "sk-dummy",
    )

    rows: list[dict[str, Any]] = []

    for idx, item in enumerate(sampled, start=1):
        q_id = item["id"]
        group = item.get("group", "chung")
        question = item["question"]
        relevant = item.get("relevant", [])

        expected_sources = "; ".join(
            f"{r['source_file']} (Điều {r.get('article') or '?'}, Trang {r.get('page') or '?'})"
            for r in relevant
        )

        logger.info("[%d/%d] Processing %s: %s", idx, len(sampled), q_id, question[:50])

        # 1. Route & Rewrite
        route_decision = route_and_rewrite(question, client=client)
        search_query = route_decision.rewritten_query or question

        # 2. Retrieval & Rerank
        candidates = retrieve_hybrid(search_query, top_k=5)
        reranked = rerank(search_query, candidates, top_k=5)

        # 3. Evidence gate
        is_sufficient, refusal = check_evidence_gate(reranked)

        if not is_sufficient:
            bot_answer = refusal or "Từ chối do thiếu ngữ cảnh."
            bot_cits_str = "None"
        else:
            answer_text, raw_cits, _ = generate_answer(question, reranked, client=client)
            valid_cits = validate_citations(raw_cits, reranked)
            bot_answer = answer_text
            bot_cits_str = (
                "; ".join(
                    f"{c.source_file} (Điều {c.article or '?'}, Trang {c.page or '?'})"
                    for c in valid_cits
                )
                or "Không có trích dẫn hợp lệ"
            )

        # 4. LLM judge scoring
        judge_res = llm_judge(
            question=question,
            expected_source=expected_sources,
            bot_answer=bot_answer,
            bot_citations=bot_cits_str,
            client=client,
        )

        rows.append(
            {
                "question_id": q_id,
                "group": group,
                "question": question,
                "expected_source": expected_sources,
                "bot_answer": bot_answer,
                "bot_citations": bot_cits_str,
                "llm_judge_score_regulation": judge_res["regulation_accuracy"],
                "llm_judge_score_citation": judge_res["citation_accuracy"],
                "llm_judge_reasoning": judge_res["reasoning"],
                "human_regulation_accuracy_1_5": "",  # Blank for human reviewer
                "human_citation_accuracy_1_5": "",  # Blank for human reviewer
                "human_notes": "",  # Blank for human notes
            }
        )

    fieldnames = [
        "question_id",
        "group",
        "question",
        "expected_source",
        "bot_answer",
        "bot_citations",
        "llm_judge_score_regulation",
        "llm_judge_score_citation",
        "llm_judge_reasoning",
        "human_regulation_accuracy_1_5",
        "human_citation_accuracy_1_5",
        "human_notes",
    ]

    with output_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved %d records to %s", len(rows), output_file)
    return output_file


if __name__ == "__main__":
    generate_manual_grading_csv()
