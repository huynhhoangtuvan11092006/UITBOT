from pathlib import Path
from pypdf import PdfReader


RAW_DIR = Path("data/raw")


def main():
    pdf_files = list(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print("Không tìm thấy file PDF trong data/raw")
        return

    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))

        total_pages = len(reader.pages)
        pages_with_text = 0
        total_characters = 0

        for page in reader.pages:
            text = page.extract_text() or ""

            if text.strip():
                pages_with_text += 1
                total_characters += len(text)

        print(f"File: {pdf_path.name}")
        print(f"Số trang: {total_pages}")
        print(f"Trang đọc được chữ: {pages_with_text}")
        print(f"Tổng số ký tự: {total_characters}")

        if pages_with_text == 0:
            print("Cảnh báo: PDF có thể là bản scan")


if __name__ == "__main__":
    main()