from pathlib import Path
import json

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


INPUT_FILE = Path("data/processed/chunks.jsonl")
INDEX_DIR = Path("data/processed/faiss_index")


def load_documents():
    documents = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            documents.append(
                Document(
                    page_content=record["text"],
                    metadata={
                        "chunk_id": record["chunk_id"],
                        "source_file": record["source_file"],
                        "page": record["page"],
                    },
                )
            )

    return documents


def main():
    if not INPUT_FILE.exists():
        print("Không tìm thấy file chunks.jsonl")
        return

    print("Đang đọc chunks...")
    documents = load_documents()
    print(f"Đã đọc {len(documents)} chunks")

    print("Đang tải model embedding tiếng Việt...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Đang tạo embeddings...")

    vectorstore = FAISS.from_documents(
        documents,
        embeddings,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    print("Đã tạo FAISS index")
    print(f"Vị trí lưu: {INDEX_DIR}")


if __name__ == "__main__":
    main()