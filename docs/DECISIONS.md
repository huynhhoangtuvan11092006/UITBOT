# Technical decisions

## Phase 1

### Replace the previous LangChain stack
Chose a from-scratch layout (FastAPI + psycopg 3 + sentence-transformers) and removed LangChain, FAISS, and Chroma.
Reason: the old schema and retrieval path cannot support per-model embedding tables, article-level metadata, or LangChain-free generation.
Rejected: keeping the old `rag_service.py` beside the new ingest pipeline (two sources of truth).

### Embedding tables named from the model slug
Chunks live in `chunks`; vectors live in `chunk_embeddings_<slug>` with HNSW cosine and a dimension taken from the loaded model.
Reason: Phase 5 must compare embedding models and chunk sets without rewriting one `VECTOR(384)` column.
Rejected: a single embedding column on `chunks`.

### OCR is per page, behind an `OcrEngine` protocol
Pages with too little text, replacement characters, or abnormal Vietnamese letter ratio are routed to OCR. Default engine is Tesseract (`vie`); if the binary is missing, the factory falls back to `none` and logs a warning.
Reason: mixed text/scan PDFs are common; later a VLM engine can implement the same protocol.
Rejected: OCR for the whole file, or a hard dependency that crashes ingest when Tesseract is absent.
Deskew is not implemented in Phase 1 (would need extra CV libraries); only grayscale is applied.

### Sentence chunking uses underthesea, with regex fallback
`underthesea.sent_tokenize` is the primary splitter for the `sentence` strategy.
Reason: it handles Vietnamese abbreviations better than splitting on `. `.
Rejected: `pyvi` for this step (it is word segmentation, not sentence splitting). BM25 tokenization is deferred to Phase 2.

### Recursive character splitter is in-repo
The `fixed` strategy copies the usual recursive-separator behavior without LangChain.
Reason: the stack forbids LangChain; eval must pin chunking behavior in this repository.

### Chat API removed until generation exists
`POST /api/v1/chat` from the previous demo is deleted in Phase 1.
Reason: it targeted a schema and pipeline that no longer exist. The static UIT UI remains but cannot answer yet.
