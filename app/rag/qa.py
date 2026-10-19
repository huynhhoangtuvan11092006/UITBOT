from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI


load_dotenv()


INDEX_DIR = Path("data/processed/faiss_index")


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_context(results):
    context_parts = []

    for document, score in results:
        source_file = document.metadata["source_file"]
        page = document.metadata["page"]
        content = document.page_content

        context_parts.append(
            f"[Nguồn: {source_file}, trang {page}]\n{content}"
        )

    return "\n\n".join(context_parts)


def main():
    if not INDEX_DIR.exists():
        print("Không tìm thấy FAISS index")
        return

    vectorstore = load_vectorstore()

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
    )

    print("UniPolicy AI đã sẵn sàng.")
    print("Gõ exit để thoát.")

    while True:
        question = input("\nCâu hỏi: ")

        if question.lower() == "exit":
            break

        results = vectorstore.similarity_search_with_score(
            question,
            k=3,
        )

        context = build_context(results)

        prompt = f"""
Bạn là UniPolicy AI, trợ lý trả lời quy chế đào tạo của UIT.

Hãy trả lời câu hỏi dựa CHỈ trên phần ngữ cảnh bên dưới.

Quy tắc:
- Đọc toàn bộ các đoạn ngữ cảnh trước khi trả lời.
- Nếu có một đoạn chứa câu trả lời trực tiếp, phải ưu tiên sử dụng đoạn đó.
- Không được nói "không tìm thấy" nếu ngữ cảnh đã chứa câu trả lời.
- Không tự thêm thông tin không có trong ngữ cảnh.
- Trả lời ngắn gọn bằng tiếng Việt.
- Luôn ghi nguồn theo dạng [Nguồn: tên file, trang số].

Ngữ cảnh:
{context}

Câu hỏi:
{question}
"""

        response = llm.invoke(prompt)

        print("\n===== TRẢ LỜI =====")
        print(response.content)


if __name__ == "__main__":
    main()