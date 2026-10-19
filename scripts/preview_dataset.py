from pathlib import Path
import json


INPUT_FILE = Path("data/processed/documents.jsonl")


def main():
    if not INPUT_FILE.exists():
        print("Không tìm thấy file documents.jsonl")
        return

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        first_record = json.loads(file.readline())

    print("Tên file:", first_record["source_file"])
    print("Số trang:", first_record["page"])
    print("\nNội dung 1000 ký tự đầu tiên:")
    print(first_record["text"][:1000])


if __name__ == "__main__":
    main()