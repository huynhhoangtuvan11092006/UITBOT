from pathlib import Path
import json


INPUT_FILE = Path("data/processed/chunks.jsonl")


def main():
    if not INPUT_FILE.exists():
        print("Không tìm thấy file chunks.jsonl")
        return

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        records = []

        for _ in range(3):
            line = file.readline()

            if not line:
                break

            records.append(json.loads(line))

    for index, record in enumerate(records, start=1):
        print(f"\n===== CHUNK {index} =====")
        print("Chunk ID:", record["chunk_id"])
        print("File nguồn:", record["source_file"])
        print("Trang:", record["page"])
        print("Nội dung:")
        print(record["text"])


if __name__ == "__main__":
    main()