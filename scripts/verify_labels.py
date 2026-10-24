"""Interactive CLI to review, edit, and verify labels in evaluation/golden_dataset.jsonl.

Usage:
    python scripts/verify_labels.py             # Interactive terminal review
    python scripts/verify_labels.py --status     # Show verification summary
    python scripts/verify_labels.py --auto-all   # Verify all pending questions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_FILE = ROOT / "evaluation" / "golden_dataset.jsonl"


def load_dataset() -> list[dict]:
    if not EVAL_FILE.exists():
        print(f"❌ Không tìm thấy file: {EVAL_FILE}")
        sys.exit(1)
    items = []
    with EVAL_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_dataset(items: list[dict]) -> None:
    temp_file = EVAL_FILE.with_suffix(".tmp")
    with temp_file.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    temp_file.replace(EVAL_FILE)


def show_status(items: list[dict]) -> None:
    total = len(items)
    verified = sum(1 for it in items if it.get("verified", False))
    pending = total - verified
    print("\n" + "=" * 55)
    print("📊 BÁO CÁO TIẾN ĐỘ XÁC THỰC NHÃN (GOLDEN DATASET)")
    print("=" * 55)
    print(f"Tổng số câu hỏi:     {total}")
    print(f"Đã duyệt (verified): {verified} ({(verified / total * 100) if total else 0:.1f}%)")
    print(f"Chưa duyệt:          {pending}")

    groups: dict[str, tuple[int, int]] = {}
    for it in items:
        g = it.get("group", "khác")
        v, t = groups.get(g, (0, 0))
        groups[g] = (v + (1 if it.get("verified", False) else 0), t + 1)

    print("\nPhân bổ theo nhóm:")
    for g, (v, t) in sorted(groups.items()):
        print(f"  • {g.ljust(12)}: {v}/{t} verified")
    print("=" * 55 + "\n")


def interactive_review(items: list[dict]) -> None:
    print("\n" + "=" * 65)
    print("🎯 BẮT ĐẦU DUYỆT NHÃN CÂU HỎI (INTERACTIVE CLI)")
    print("Phím tắt:")
    print("  [y/Enter] Chấp nhận | [e] Sửa | [d] Xóa | [s] Bỏ qua | [a] Duyệt hết | [q] Thoát")
    print("=" * 65 + "\n")

    modified = False
    idx = 0
    while idx < len(items):
        item = items[idx]
        if item.get("verified", False):
            idx += 1
            continue

        q_id = item.get("id", f"q_{idx+1}")
        question = item.get("question", "")
        group = item.get("group", "")
        relevant = item.get("relevant", [])

        print(f"\n[{idx + 1}/{len(items)}] ID: {q_id} | Nhóm: {group}")
        print(f"❓ Câu hỏi: {question}")
        print("📄 Tài liệu liên quan:")
        for r in relevant:
            sf, pg, art = r.get("source_file"), r.get("page"), r.get("article")
            print(f"   - File: {sf} | Trang: {pg} | Điều: {art}")

        try:
            choice = input("\nLựa chọn [y/e/d/s/a/q] (Mặc định: y): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nĐã tạm dừng duyệt.")
            break

        if choice in ("", "y", "yes"):
            item["verified"] = True
            modified = True
            print("✅ Đã xác nhận.")
            idx += 1
        elif choice == "e":
            new_q = input(f"Sửa câu hỏi [{question}]: ").strip()
            if new_q:
                item["question"] = new_q
            item["verified"] = True
            modified = True
            print("✏️ Đã cập nhật và xác nhận.")
            idx += 1
        elif choice == "d":
            items.pop(idx)
            modified = True
            print("🗑️ Đã xóa câu hỏi.")
        elif choice == "s":
            print("⏩ Đã bỏ qua.")
            idx += 1
        elif choice == "a":
            # Chấp nhận toàn bộ các câu còn lại
            for rem in items[idx:]:
                rem["verified"] = True
            modified = True
            print("🚀 Đã chấp thuận toàn bộ các câu hỏi còn lại!")
            break
        elif choice == "q":
            print("Lưu và thoát.")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng chọn y, e, d, s, a, hoặc q.")

    if modified:
        save_dataset(items)
        print("💾 Đã lưu thay đổi vào evaluation/golden_dataset.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="UITBOT Golden Dataset Label Verifier CLI")
    parser.add_argument("--status", action="store_true", help="Hiển thị thống kê nhãn")
    parser.add_argument(
        "--auto-all", action="store_true", help="Tự động xác nhận tất cả câu hỏi"
    )
    args = parser.parse_args()

    items = load_dataset()

    if args.status:
        show_status(items)
    elif args.auto_all:
        for it in items:
            it["verified"] = True
        save_dataset(items)
        print(f"✅ Đã duyệt xác thực thành công toàn bộ {len(items)} câu hỏi!")
        show_status(items)
    else:
        interactive_review(items)
        show_status(items)


if __name__ == "__main__":
    main()
