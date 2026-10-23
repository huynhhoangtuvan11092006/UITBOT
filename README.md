# UITBOT — Vietnamese University Regulation RAG Assistant

UITBOT answers questions about official academic regulations of the **University of Information Technology, VNU-HCM (UIT)**. Answers must be grounded in retrieved passages and cite file, article (Điều), and page.

This repository is being rebuilt in measured phases. **Phase 1 (current)** covers configuration, PostgreSQL + pgvector, PDF ingest (text/OCR), structure parsing, and chunking. Chat answering is not available yet.

## Tech stack (Phase 1)

- Python 3.12, FastAPI, Pydantic v2
- PostgreSQL 16 + pgvector (Docker)
- psycopg 3 + connection pool
- Embedding at ingest time: `BAAI/bge-m3` (local sentence-transformers)
- OCR: PyMuPDF + Tesseract `vie` (optional; ingest still runs if Tesseract is missing)

## Quick start (Phase 1)

1. Copy `.env.example` to `.env` and set `POSTGRES_PASSWORD` (do not commit `.env`).
2. Place official PDFs in `data/raw/`.
3. Start the database:

```bash
docker compose up -d db
```

4. Create a virtualenv, install dependencies, then ingest:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/ingest.py
```

5. Run the API (health check + static UI):

```bash
uvicorn app.main:app --reload --port 8000
```

- App UI: http://localhost:8000
- Health: http://localhost:8000/health
- API docs: http://localhost:8000/docs

### Tesseract on Windows (for scanned pages)

Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) and the Vietnamese language pack (`vie`). Confirm with `tesseract --version`. If the binary is missing, set `OCR_ENGINE=none` or leave the default: ingest logs a warning and keeps the extracted text.

### Tests

```bash
pytest
```

## Ingest output

- Chunks and embeddings in PostgreSQL (`chunks` plus `chunk_embeddings_<model_slug>`)
- `data/processed/ocr_review.csv` — pages that went through OCR (file, page, confidence, first 300 characters) for manual review

Re-running `python scripts/ingest.py` upserts by `chunk_id` and does not duplicate rows for the same `chunk_set`.

## Phase 4 — UI Manual Testing Checklist (Checklist kiểm thử giao diện)

Mở trình duyệt vào **`http://localhost:8000`** và thực hiện các bước kiểm tra sau:

- [ ] **1. Kiểm thử Streaming thời gian thực (SSE)**:
  - Bấm vào một chip gợi ý (ví dụ: *"Điều kiện để xét tốt nghiệp?"*) hoặc gõ câu hỏi vào ô input.
  - **Kỳ vọng:** Xuất hiện trạng thái phân tích, sau đó câu trả lời tuôn ra từng cụm từ kèm con trỏ nhấp nháy liên tục, giao diện tự động cuộn xuống dưới cùng mượt mà.
- [ ] **2. Kiểm thử Thẻ nguồn trích dẫn (File – Điều – Trang)**:
  - Khi câu trả lời hoàn tất, kiểm tra phần *"Căn cứ quy chế trích dẫn:"*.
  - **Kỳ vọng:** Hiển thị thẻ dạng pill màu xanh UIT với định dạng `📄 [Tên_file.pdf] – [Điều X] – Trang [Y]`. Có trích đoạn văn bản `quote` đi kèm trong hộp viền xanh.
- [ ] **3. Kiểm thử Đánh giá phản hồi 👍 / 👎 (`/api/v1/feedback`)**:
  - Bấm vào nút `👍` hoặc `👎` ở góc dưới câu trả lời của bot.
  - **Kỳ vọng:** Nút chuyển sang trạng thái active, xuất hiện dòng thông báo `"✓ Cảm ơn bạn!"`. Kiểm tra tab Network của trình duyệt thấy gửi thành công `POST /api/v1/feedback` với payload `{ "rating": 1, "session_id": "..." }` trả về mã `200 OK`.
- [ ] **4. Kiểm thử Lịch sử phía Client (localStorage)**:
  - Đặt từ 2 - 3 câu hỏi liên tiếp để tạo đoạn hội thoại dài.
  - Bấm `F5` hoặc đóng tab mở lại `http://localhost:8000`.
  - **Kỳ vọng:** Toàn bộ lịch sử tin nhắn của người dùng và bot cùng các thẻ trích dẫn vẫn được phục hồi nguyên vẹn.
  - Bấm nút **"Làm mới hội thoại"** trên thanh Header → Chọn OK → Lịch sử trong `localStorage` được dọn sạch và màn hình quay lại tin nhắn chào mặc định.
- [ ] **5. Kiểm thử Trạng thái lỗi & Thử lại (Error State & Retry)**:
  - Tạm dừng container app bằng lệnh: `docker compose stop app`.
  - Nhập một câu hỏi và bấm gửi.
  - **Kỳ vọng:** Giao diện không bị treo, hiển thị thông báo lỗi màu đỏ lịch sự kèm nút `🔄 Thử lại`. Bật lại app (`docker compose start app`) và bấm `Thử lại` → Câu hỏi được gửi lại thành công.
- [ ] **6. Kiểm thử Hiển thị an toàn chống tấn công XSS**:
  - Nhập vào ô input đoạn mã độc: `<script>alert('xss')</script>` hoặc `<img src=x onerror=alert(1)>`.
  - **Kỳ vọng:** Toàn bộ chuỗi ký tự được hiển thị an toàn dưới dạng chuỗi văn bản thuần túy (`textContent`), không có bất kỳ cửa sổ popup độc hại nào được kích hoạt.


## Project structure

```text
app/            FastAPI app, config, db, ingestion
scripts/        ingest.py
tests/          unit tests
docs/           ARCHITECTURE.md, DECISIONS.md
data/raw/       official PDFs (you add these)
evaluation/     filled in Phase 5
```

## Known limits (Phase 1)

- No chat / retrieval / generation API yet.
- Retrieval quality numbers are not published here; they will come from `evaluation/results.md` after a real eval run.
- Header/footer stripping is heuristic and can over-cut; check OCR review and a few chunks after ingest.
- `BAAI/bge-m3` is large; the first ingest downloads the model.

## Author

**Huỳnh Hoàng Tứ Văn** — University of Information Technology, VNU-HCM

## Disclaimer

UITBOT is an educational project. Answers (once enabled) assist with information retrieval and do not replace official UIT announcements.
