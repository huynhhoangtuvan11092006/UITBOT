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
