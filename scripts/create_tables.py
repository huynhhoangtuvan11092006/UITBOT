import os

import psycopg2
from dotenv import load_dotenv


load_dotenv()


def main():
    connection = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )

    cursor = connection.cursor()

    cursor.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            chunk_id TEXT UNIQUE NOT NULL,
            source_file TEXT NOT NULL,
            page INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding VECTOR(384) NOT NULL
        );
    """)

    connection.commit()

    cursor.close()
    connection.close()

    print("Đã tạo bảng document_chunks thành công.")


if __name__ == "__main__":
    main()