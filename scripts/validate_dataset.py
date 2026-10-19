from pathlib import Path
import json


INPUT_FILE = Path("data/processed/documents.jsonl")


def main():
    if not INPUT_FILE.exists():
        print("Không tìm thấy file documents.jsonl")
        return

    total_records = 0
    invalid_records = 0
    empty_text_records = 0

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Lỗi JSON tại dòng {line_number}")
                invalid_records += 1
                continue

            required_fields = [
                "source_file",
                "page",
                "text"
            ]

            for field in required_fields:
                if field not in record:
                    print(
                        f"Dòng {line_number} thiếu trường: {field}"
                    )
                    invalid_records += 1

            if not record.get("text", "").strip():
                empty_text_records += 1

            total_records += 1

    print("\nKẾT QUẢ KIỂM TRA")
    print(f"Tổng bản ghi: {total_records}")
    print(f"Bản ghi lỗi: {invalid_records}")
    print(f"Bản ghi rỗng: {empty_text_records}")

    if invalid_records == 0 and empty_text_records == 0:
        print("Dataset hợp lệ")
    else:
        print("Dataset còn lỗi, cần kiểm tra lại")


if __name__ == "__main__":
    main()