# UITBOT System Architecture

UITBOT is a production-grade, grounded Vietnamese Retrieval-Augmented Generation (RAG) assistant designed for student academic regulations at the University of Information Technology (VNU-HCM - UIT).

---

## 1. High-Level Architecture Overview

The system is organized into a modular 4-tier architecture decoupling data ingestion, hybrid retrieval, guardrailed generation, and client delivery:

```mermaid
flowchart TD
    subgraph Ingestion["1. Ingestion & Storage Pipeline"]
        pdf["Official Regulation PDFs<br/>(data/raw/*.pdf)"] --> pymupdf["PDF Text Extraction<br/>(PyMuPDF)"]
        pymupdf --> quality_gate{"Quality Gate<br/>(length, , VN ratio)"}
        quality_gate -->|"Clean Text"| nfc["Unicode NFC Normalization"]
        quality_gate -->|"Scanned / Degraded"| ocr["Tesseract OCR Engine<br/>(tesseract-ocr-vie, 300 DPI)"]
        ocr --> nfc
        nfc --> structure["Structure Parser<br/>(Regex Chương / Điều / Khoản)"]
        structure --> chunker["Recursive Text Splitter<br/>(fixed 800/150 chars)"]
        chunker --> embed["Dense Embedder<br/>(BAAI/bge-m3, 1024-dim)"]
        embed --> pgvector[("PostgreSQL 16 + pgvector<br/>(HNSW Cosine Index)")]
        chunker --> bm25_store[("In-Memory BM25 Okapi Index<br/>(underthesea word tokenizer)")]
    end

    subgraph Retrieval["2. Hybrid Retrieval & Reranking"]
        user_query["User Question"] --> router["Query Router & Rewriter<br/>(Formal Regulation Query)"]
        router --> dense_search["Dense Vector Search<br/>(HNSW top-20 candidates)"]
        router --> bm25_search["BM25 Lexical Search<br/>(underthesea top-20 candidates)"]
        dense_search --> rrf["Reciprocal Rank Fusion<br/>(RRF k=60, top-20 fused)"]
        bm25_search --> rrf
        rrf --> reranker["Cross-Encoder Reranker<br/>(bge-reranker-v2-m3, top-5)"]
    end

    subgraph Generation["3. Guardrails & Answer Generation"]
        reranker --> evidence_gate{"Evidence Gate<br/>(Top score >= 0.02?)"}
        evidence_gate -->|"No (score < 0.02)"| refusal["Refusal & Academic Affairs Notice<br/>(phongdaotao@uit.edu.vn)"]
        evidence_gate -->|"Yes"| llm["LLM Grounded Synthesis<br/>(Strict JSON Schema)"]
        llm --> citation_validator{"Citation Validator<br/>(Grounding & Page Match)"}
        citation_validator -->|"Prune ungrounded"| response["Verified Answer + Citations"]
        llm -.->|"Offline / Timeout"| extractive_fallback["Deterministic Extractive Fallback"]
    end

    subgraph Delivery["4. Delivery Layer & Frontend"]
        response --> sse["Server-Sent Events (SSE)<br/>(/api/v1/chat/stream)"]
        refusal --> sse
        extractive_fallback --> sse
        sse --> web_ui["Vanilla JS/CSS Responsive UI<br/>(Zero external build tools)"]
        web_ui --> feedback_api["User Feedback Loop<br/>(/api/v1/feedback)"]
    end
```

---

## 2. Ingestion & Quality Gate Pipeline

1. **Extraction**: Reads raw PDF regulations from `data/raw/` using `PyMuPDF` (`fitz`).
2. **Quality Heuristics**: Automatically inspects every page:
   - Character count $\ge 50$.
   - Replacement character (`\ufffd`) ratio $\le 0.02$.
   - Vietnamese accented vowel ratio $\ge 0.15$.
   Pages failing any heuristic are transparently routed to `Tesseract OCR` (`vie`, 300 DPI).
3. **Normalization**: Strips duplicate blank lines, unifies hyphens, and converts all strings to standard Unicode NFC (`unicodedata.normalize("NFC", text)`).
4. **Structural Extraction**: Identifies organizational units (`Chương`, `Mục`, `Điều`) and preserves them alongside metadata (`source_file`, `page_start`, `page_end`, `chapter`, `article`, `article_title`).
5. **Multi-Model Storage**:
   - `chunks`: Master table storing chunks, source files, page spans, and text.
   - `chunk_embeddings_<model_slug>`: Dynamic vector tables dedicated to each embedding model slug, indexed with `HNSW (m=16, ef_construction=64, vector_cosine_ops)`.

---

## 3. Retrieval & Reranker Architecture

Empirical evaluation demonstrated that relying exclusively on Dense or BM25 search leaves severe blind spots in university regulation search:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Router as Router & Rewriter
    participant DB as PostgreSQL (pgvector)
    participant BM25 as In-Memory BM25
    participant RRF as Reciprocal Rank Fusion
    participant CE as Cross-Encoder Reranker

    User->>Router: "em bị ốm nằm viện ko thi được làm sao ạ"
    Router->>Router: Rewrite into formal query: "quy định hoãn thi vắng thi ốm đau tai nạn điểm I"
    par Dense Search
        Router->>DB: Cosine KNN (ORDER BY embedding <=> query_vec LIMIT 20)
        DB-->>RRF: Top 20 dense candidates
    and Sparse Search
        Router->>BM25: BM25Okapi search with underthesea word tokens
        BM25-->>RRF: Top 20 BM25 candidates
    end
    RRF->>RRF: Compute RRF score: Sum(1 / (60 + rank))
    RRF->>CE: Top 20 merged candidates
    CE->>CE: Cross-attention score query-chunk pairs (bge-reranker-v2-m3)
    CE-->>User: Top 5 reranked chunks with confidence scores
```

---

## 4. Guardrails & Anti-Hallucination Architecture

UITBOT enforces 4 defense-in-depth safety layers before returning an answer:

```mermaid
flowchart TD
    input["Incoming Request"] --> L1["Layer 1: Input Validation & Sanitization<br/>- max_length=1000 chars<br/>- Prompt injection & jailbreak regex<br/>- Delimiter neutralization"]
    L1 --> L2["Layer 2: Intent Classification<br/>- SMALL_TALK: Greeting responses<br/>- OUT_OF_SCOPE / INJECTION: Safe polite rejection<br/>- AMBIGUOUS: Clarification prompt<br/>- UIT_REGULATION: Proceed to retrieval"]
    L2 --> L3["Layer 3: Evidence Gate<br/>- Evaluates top-1 Cross-Encoder score<br/>- If score < 0.02: Short-circuit refusal with Office contact"]
    L3 --> L4["Layer 4: Code-Level Citation Validator<br/>- Matches (source_file, article, page) against candidates<br/>- Prunes any hallucinated or ungrounded citations<br/>- Extractive fallback if LLM times out"]
    L4 --> output["Safe, Grounded Output"]
```

---

## 5. Security & Isolation

- **Non-Root Execution**: Runs under unprivileged user `uitbot` (UID 1000, GID 1000) inside Docker.
- **Credential Hygiene**: Zero secrets in source code or git history; all configuration loaded via environment variables and `.env`.
- **Input Sanitization**: Control characters, delimiter spoofing (`--- [TÀI LIỆU ... ---`), and XML tag injections are stripped before prompt assembly.
- **Safe Prompt Delimiting**: User queries are isolated within `<user_query>` tags accompanied by strict system instructions forbidding prompt overriding.
- **XSS Immunity**: The web interface builds DOM nodes exclusively via `textContent` and `createElement`, completely avoiding `innerHTML` execution.
