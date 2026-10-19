import json
from pathlib import Path

from app.services.rag_service import RAGService

DATASET = Path("evaluation/golden_dataset.jsonl")


def main():
    with DATASET.open("r", encoding="utf-8") as file:
        questions = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not questions:
        raise ValueError("Golden dataset đang rỗng.")

    rag = RAGService()

    hits = {3: 0, 5: 0}

    # MRR
    reciprocal_ranks = []

    for number, item in enumerate(questions, start=1):
        question = item["question"]

        expected = {
            (source["source_file"], int(source["page"]))
            for source in item["relevant_pages"]
        }

        # Lấy 5 kết quả từ Hybrid Search
        results = rag.retrieve(question, top_k=5)

        retrieved_pages = [
            (row[1], int(row[2]))
            for row in results
        ]

        # Recall@3 / Recall@5
        for k in (3, 5):
            top_k_pages = set(
                retrieved_pages[:k]
            )

            if top_k_pages & expected:
                hits[k] += 1

        # =========================
        # MRR@5
        # =========================
        rr = 0.0

        for rank, page_info in enumerate(
            retrieved_pages,
            start=1,
        ):
            if page_info in expected:
                rr = 1.0 / rank
                break

        reciprocal_ranks.append(rr)

        print(f"[{number}/{len(questions)}] {question}")
        print("Expected:", sorted(expected))
        print("Hybrid top 5:", retrieved_pages)
        print(f"Reciprocal Rank: {rr:.4f}")
        print("-" * 60)

    total = len(questions)

    # Tính MRR
    mrr = sum(reciprocal_ranks) / total

    print("\n===== HYBRID SEARCH EVALUATION =====")
    print(f"Total questions: {total}")
    print(f"Recall@3 (page-level): {hits[3] / total:.2%}")
    print(f"Recall@5 (page-level): {hits[5] / total:.2%}")
    print(f"MRR@5: {mrr:.4f}")


if __name__ == "__main__":
    main()