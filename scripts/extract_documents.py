from pathlib import Path
import json
import re

from pypdf import PdfReader


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "documents.jsonl"


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u00a0", " ")

    # Gộp nhiều khoảng trắng thành một
    text = re.sub(r"[ \t]+", " ", text)

    # Giảm số dòng trống liên tiếp
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def main():
    # Tạo thư mục processed nếu chưa có
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print("Không tìm thấy file PDF trong data/raw")
        return

    total_pages = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as output_file:
        for pdf_path in pdf_files:
            print(f"Đang xử lý: {pdf_path.name}")

            reader = PdfReader(str(pdf_path))

            for page_number, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text() or ""
                text = clean_text(raw_text)

                if not text:
                    continue

                record = {
                    "source_file": pdf_path.name,
                    "page": page_number,
                    "text": text
                }

                output_file.write(
                    json.dumps(record, ensure_ascii=False)
                    + "\n"
                )

                total_pages += 1

    print("\nĐã xử lý xong")
    print(f"Tổng số trang có nội dung: {total_pages}")
    print(f"File kết quả: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()