"""Pre-generate query rewrites using local 9Router to warm the cache."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

DATASET = Path("evaluation/golden_dataset.jsonl")
CACHE_FILE = Path("evaluation/rewritten_queries_cache.json")

SYSTEM_PROMPT = """Bạn là bộ phân loại ý định (Router) và chuẩn hóa truy vấn (Query Rewriter)
cho UITBOT - trợ lý giải đáp quy chế đào tạo Trường Đại học Công nghệ Thông tin (ĐHQG-HCM - UIT).

Nhiệm vụ: Viết lại câu hỏi (rewritten_query) từ khẩu ngữ sinh viên sang thuật ngữ pháp quy UIT
(bỏ "em", "mình", mở rộng từ đồng nghĩa: hoãn thi, vắng thi, điểm I, bảo lưu, kỷ luật...).
rewritten_query phải ngắn gọn, súc tích (10-25 từ), tối ưu cho BM25 và Vector search.

Định dạng trả về BẮT BUỘC là JSON object:
{
  "intent": "UIT_REGULATION",
  "rewritten_query": "chuỗi truy vấn chuẩn hóa",
  "explanation": "lý do"
}
"""


def main():
    with DATASET.open("r", encoding="utf-8") as f:
        questions = [json.loads(line)["question"] for line in f if line.strip()]

    cache = {}
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    to_process = [q for q in questions if q not in cache]
    print(f"Total questions: {len(questions)}, Cached: {len(cache)}, Remaining: {len(to_process)}")

    if not to_process:
        print("All questions already cached!")
        return

    client = OpenAI(
        base_url="http://localhost:20128/v1",
        api_key="sk-46cdcf81e036987d-wt9lbw-51719286",
    )

    def process_q(q):
        try:
            res = client.chat.completions.create(
                model="ag/gemini-3.8-flash",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = res.choices[0].message.content or "{}"
            data = json.loads(content)
            rw = data.get("rewritten_query") or q
            return q, rw
        except Exception as e:
            return q, q

    # Run with 10 threads concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_q, q): q for q in to_process}
        for i, fut in enumerate(as_completed(futures), 1):
            q, rw = fut.result()
            cache[q] = rw
            if i % 20 == 0 or i == len(to_process):
                print(f"Processed {i}/{len(to_process)}")

    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Successfully cached {len(cache)} rewrites to {CACHE_FILE}")


if __name__ == "__main__":
    main()
