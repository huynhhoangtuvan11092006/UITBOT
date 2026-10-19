import json
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DATASET = Path("evaluation/golden_dataset.jsonl")
MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Không tìm thấy {DATASET}")

    with DATASET.open("r", encoding="utf-8") as file:
        questions = [json.loads(line) for line in file if line.strip()]

    if not questions:
        raise ValueError("Golden dataset đang rỗng.")

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    with psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    ) as connection:
        with connection.cursor() as cursor:

            hits = {3: 0, 5: 0}

         
            # MRR
       
            reciprocal_ranks = []

            for item in questions:
                question = item["question"]

                expected = {
                    (p["source_file"], int(p["page"]))
                    for p in item["relevant_pages"]
                }

                vector = embeddings.embed_query(question)

                vector_string = "[" + ",".join(
                    str(value) for value in vector
                ) + "]"

                cursor.execute(
                    """
                    SELECT source_file, page
                    FROM document_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT 5;
                    """,
                    (vector_string,),
                )

                results = cursor.fetchall()

            
                # RECALL@3 / RECALL@5
         
                for k in (3, 5):
                    retrieved = {
                        (source, page)
                        for source, page in results[:k]
                    }

                    if retrieved & expected:
                        hits[k] += 1

 
                # MRR@5
  
                rr = 0.0

                for rank, (source, page) in enumerate(
                    results,
                    start=1,
                ):
                    if (source, page) in expected:
                        rr = 1.0 / rank
                        break

                reciprocal_ranks.append(rr)


                print(f"\nQuestion: {question}")
                print("Expected pages:", sorted(expected))
                print("Top 5 retrieved:", results)
                print(f"Reciprocal Rank: {rr:.4f}")

    total = len(questions)


    # TÍNH MRR
    mrr = sum(reciprocal_ranks) / total

    print("\n===== RETRIEVAL BASELINE =====")
    print(f"Total questions: {total}")
    print(f"Recall@3 (page-level): {hits[3] / total:.2%}")
    print(f"Recall@5 (page-level): {hits[5] / total:.2%}")
    print(f"MRR@5: {mrr:.4f}")


if __name__ == "__main__":
    main()