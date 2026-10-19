from pathlib import Path
import json
import os

import psycopg2
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


INPUT_FILE = Path("data/processed/chunks.jsonl")


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def main():
    if not INPUT_FILE.exists():
        print("Không tìm thấy file chunks.jsonl")
        return

    print("Đang tải model embedding...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Đang đọc chunks...")

    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    print(f"Tổng số chunks: {len(records)}")

    texts = [record["text"] for record in records]

    print("Đang tạo embeddings...")

    vectors = embeddings.embed_documents(texts)

    print(f"Đã tạo {len(vectors)} embeddings")
    print(f"Số chiều vector: {len(vectors[0])}")

    connection = get_connection()
    cursor = connection.cursor()

    print("Đang lưu vào PostgreSQL...")

    for record, vector in zip(records, vectors):

        vector_string = "[" + ",".join(map(str, vector)) + "]"

        cursor.execute(
            """
            INSERT INTO document_chunks (
                chunk_id,
                source_file,
                page,
                text,
                embedding
            )
            VALUES (%s, %s, %s, %s, %s::vector)
            ON CONFLICT (chunk_id)
            DO UPDATE SET
                source_file = EXCLUDED.source_file,
                page = EXCLUDED.page,
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding;
            """,
            (
                record["chunk_id"],
                record["source_file"],
                record["page"],
                record["text"],
                vector_string,
            ),
        )

    connection.commit()

    cursor.close()
    connection.close()

    print("Đã lưu toàn bộ chunks vào PostgreSQL.")


if __name__ == "__main__":
    main()