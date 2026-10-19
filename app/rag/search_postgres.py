import os

import psycopg2
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def main():
    print("Đang tải model embedding...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    connection = get_connection()
    cursor = connection.cursor()

    print("PostgreSQL vector search đã sẵn sàng.")
    print("Nhập exit để thoát.")

    while True:
        query = input("\nCâu hỏi: ")

        if query.lower() == "exit":
            break

        query_vector = embeddings.embed_query(query)

        vector_string = "[" + ",".join(map(str, query_vector)) + "]"

        cursor.execute(
            """
            SELECT
                chunk_id,
                source_file,
                page,
                text,
                embedding <=> %s::vector AS distance
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT 3;
            """,
            (vector_string, vector_string),
        )

        results = cursor.fetchall()

        print("\n===== CÁC ĐOẠN LIÊN QUAN NHẤT =====")

        for index, row in enumerate(results, start=1):
            chunk_id, source_file, page, text, distance = row

            print(f"\n===== KẾT QUẢ {index} =====")
            print("Distance:", distance)
            print("Chunk ID:", chunk_id)
            print("File:", source_file)
            print("Trang:", page)
            print("Nội dung:")
            print(text)

    cursor.close()
    connection.close()


if __name__ == "__main__":
    main()