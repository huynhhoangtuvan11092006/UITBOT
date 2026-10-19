from pathlib import Path
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter


INPUT_FILE = Path("data/processed/documents.jsonl")
OUTPUT_FILE = Path("data/processed/chunks.jsonl")


def main():
    if not INPUT_FILE.exists():
        print("Không tìm thấy file documents.jsonl")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    total_chunks = 0

    with (
        INPUT_FILE.open("r", encoding="utf-8") as input_file,
        OUTPUT_FILE.open("w", encoding="utf-8") as output_file,
    ):
        for line in input_file:
            document = json.loads(line)

            text = document["text"]

            chunks = splitter.split_text(text)

            for chunk_index, chunk in enumerate(chunks):
                record = {
                    "chunk_id": (
                        f"{document['source_file']}_"
                        f"page_{document['page']}_"
                        f"chunk_{chunk_index}"
                    ),
                    "source_file": document["source_file"],
                    "page": document["page"],
                    "text": chunk,
                }

                output_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                total_chunks += 1

    print("Đã chia nhỏ tài liệu")
    print(f"Tổng số chunks: {total_chunks}")
    print(f"File kết quả: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()