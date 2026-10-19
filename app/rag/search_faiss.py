from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


INDEX_DIR = Path("data/processed/faiss_index")


def main():
    if not INDEX_DIR.exists():
        print("Không tìm thấy thư mục faiss_index")
        return

    print("Đang tải model embedding...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Đang tải FAISS index...")

    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print("Đã tải FAISS index")
    print("Nhập exit để thoát")

    while True:
        query = input("\nCâu hỏi: ")

        if query.lower() == "exit":
            break

        results = vectorstore.similarity_search_with_score(
            query,
            k=3,
        )

        print("\nCác đoạn liên quan nhất:")

        for index, (document, score) in enumerate(results, start=1):
            print(f"\n===== KẾT QUẢ {index} =====")
            print("Điểm khoảng cách:", score)
            print("File:", document.metadata["source_file"])
            print("Trang:", document.metadata["page"])
            print("Chunk ID:", document.metadata["chunk_id"])
            print("Nội dung:")
            print(document.page_content)


if __name__ == "__main__":
    main()