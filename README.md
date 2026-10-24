# 🎓 UITBOT — Production-Grade Vietnamese Legal RAG System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16_pgvector-336791?style=flat&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![CI Tests](https://img.shields.io/badge/CI-53%2F53%20Passing-brightgreen?style=flat&logo=githubactions&logoColor=white)]()
[![Code Style](https://img.shields.io/badge/Code%20Style-Ruff-000000?style=flat&logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Trợ lý AI thông minh hỗ trợ sinh viên tra cứu nhanh chóng, chính xác các quy chế, quy định học vụ tại **Trường Đại học Công nghệ Thông tin (ĐHQG-HCM - UIT)** kèm nguồn trích dẫn rõ ràng đến từng Điều khoản và số trang.

---

## 📌 Tổng quan hiệu năng (Executive Summary)

Hệ thống được thiết kế theo tiêu chuẩn công nghiệp với kiến trúc **4 tầng phòng vệ (Defense-in-Depth)** và kiểm thử thực nghiệm trên 105 câu hỏi thực tế từ sinh viên:

| 🎯 89.5% | ⚡ 291ms | 🛡️ 100% | 🏆 95.0% |
|:---:|:---:|:---:|:---:|
| **Độ phủ tài liệu (Top-50)** | **Độ trễ Hybrid Retrieval** | **Loại trừ trích dẫn ảo** | **Độ chuẩn xác câu trả lời** |
| *74.4% câu trượt Top-5 nằm trong Top-50* | *Dense pgvector + BM25 Okapi* | *Deterministic Python Validator* | *19/20 câu đạt độ tin cậy tuyệt đối* |

---

## 🏗️ Kiến trúc hệ thống (System Architecture)

Hệ thống loại bỏ hoàn toàn các framework trừu tượng cồng kềnh (LangChain/LlamaIndex) để trực tiếp tối ưu hóa pipeline qua **FastAPI**, **pgvector HNSW**, và **Sentence-Transformers**:

```mermaid
flowchart TD
    subgraph Ingestion["📥 1. Ingestion Pipeline"]
        PDF["PDFs Quy chế UIT (11 văn bản)"] --> OCR{"Kiểm tra trang scan?<br/>(Tỷ lệ chữ tiếng Việt)"}
        OCR -->|"Văn bản số hóa"| PyMuPDF["PyMuPDF Extract"]
        OCR -->|"Scan / Dấu mộc"| Tesseract["Tesseract OCR (vie)"]
        PyMuPDF & Tesseract --> Chunking["Hierarchical & Sentence Chunking<br/>(underthesea)"]
        Chunking --> EmbedStore["Embeddings: BAAI/bge-m3<br/>Storage: PostgreSQL 16 pgvector (HNSW)"]
    end

    subgraph Serving["⚡ 2. Online Serving Pipeline"]
        User(["Sinh viên"]) --> Guardrails["Input Validation & Anti-Prompt Injection"]
        Guardrails --> Router{"Intent Router & Rewriter"}
        Router -->|"Smalltalk / Chào hỏi"| DirectResp["Phản hồi tức thì"]
        Router -->|"Tra cứu quy chế"| Retrieval["Hybrid Retrieval (RRF k=60)"]
        
        subgraph RetEngine["Bộ đôi tìm kiếm kết hợp"]
            Retrieval --> Dense["pgvector HNSW (Dense 1024d)"]
            Retrieval --> BM25["BM25 Okapi (underthesea tokenized)"]
            Dense & BM25 --> RRF["RRF Fusion & Cross-Encoder Rerank<br/>(bge-reranker-v2-m3)"]
        end
        
        RRF --> Gate{"Evidence Gate<br/>(Score >= 0.02?)"}
        Gate -->|"Không đủ căn cứ"| SafeReject["Từ chối an toàn + Hướng dẫn gửi P.Đào tạo"]
        Gate -->|"Đủ bằng chứng"| LLM["LLM Grounded Answer Generator"]
        LLM --> Validator["Deterministic Python Citation Validator"]
        Validator --> StreamOut(["SSE Stream Output (Client UI)"])
    end
```

---

## 🚀 Khởi chạy nhanh (Quick Start trong 5 lệnh)

### Yêu cầu tiên quyết
- **Docker** & **Docker Compose** (hoặc Python 3.12+ và PostgreSQL 16 pgvector).

```bash
# 1. Clone mã nguồn
git clone https://github.com/huynhhoangtuvan11092006/UITBOT.git && cd UITBOT

# 2. Khởi tạo cấu hình môi trường
cp .env.example .env

# 3. Khởi chạy cụm dịch vụ (App non-root + PostgreSQL pgvector)
docker compose up -d

# 4. Trích xuất văn bản và nạp vector vào CSDL
docker compose exec -T app python scripts/ingest.py

# 5. Tái hiện toàn bộ bảng kết quả thực nghiệm chuẩn
docker compose exec -T app python scripts/evaluate_all.py
```

### Cổng dịch vụ trực tiếp
- **Giao diện sinh viên**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

> 💡 **Lưu ý cài đặt Tesseract trên Windows** *(chỉ cần khi chạy local không qua Docker)*: Tải và cài đặt [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki), chọn gói ngôn ngữ **Vietnamese**, thêm thư mục cài đặt vào `PATH`. Khi chạy bằng Docker, Tesseract OCR đã được cấu hình tự động.

---

## 📊 Kết quả thực nghiệm (`evaluation/results.md`)

Mọi thông số dưới đây đều được đo lường thực tế trên bộ dữ liệu **105 câu hỏi vàng** (`evaluation/golden_dataset.jsonl`):

### 1. So sánh chiến lược tìm kiếm (Retrieval Performance)

| Chiến lược tìm kiếm | Recall@5 | MRR | Độ trễ p50 | Đặc tính kỹ thuật |
|:---|:---:|:---:|:---:|:---|
| **Dense (Vector bge-m3)** | 58.1% | 0.444 | 413ms | Bắt ngữ nghĩa tốt, hạn chế với từ viết tắt học vụ |
| **Hybrid (Dense + BM25)** | 59.1% | 0.433 | **291ms ⚡** | **Cân bằng tối ưu:** Bắt chuẩn từ khóa (`TOEIC`, `điểm I`, `học vụ`) |
| **Hybrid + Reranker** | 59.1% | **0.462 🎯** | ~5.5s | **Chính xác nhất:** Đưa đoạn văn chuẩn nhất lên vị trí Top-1 |
| **Rewrite + Hybrid + Reranker** | **61.9% 🏆** | 0.452 | ~12.6s | **Thông minh nhất:** Chuyển đổi khẩu ngữ sinh viên sang thuật ngữ quy chế |

### 2. Phân tích độ sâu tìm kiếm (Depth Ablation)

| Kích thước ngữ cảnh | Tỷ lệ tìm trúng tài liệu | Ý nghĩa kỹ thuật |
|:---:|:---:|:---|
| **Top 5 đoạn văn** | **59.1%** | Ngữ cảnh chuẩn gửi vào Prompt LLM để giữ độ trễ thấp |
| **Top 20 đoạn văn** | **80.0%** | Tăng vọt +20.9% khả năng định vị đúng văn bản |
| **Top 50 đoạn văn** | **89.5% 🎯** | **Gần 90% câu hỏi tìm thấy văn bản quy chế chính xác** |

> 🔍 **Insight cốt lõi**: Có đến **74.4% (32/43) câu hỏi trượt ở Top-5 thực chất đã nằm sẵn trong Top-50**. Bộ tìm kiếm hoạt động xuất sắc; điểm nghẽn hiệu năng chủ yếu nằm ở khâu re-ranking và độ dài context của LLM.

### 3. Đánh giá câu trả lời & Phòng vệ ảo giác (Guardrails)
- 🛡️ **100% Trích dẫn chính xác**: Thuật toán Python độc lập so khớp Điều, Khoản, Số trang trực tiếp từ metadata DB — xóa bỏ hoàn toàn trích dẫn bịa đặt.
- 🚦 **Evidence Gate ($\tau=0.02$)**: Tự động từ chối an toàn khi tài liệu không đủ căn cứ, hướng dẫn sinh viên liên hệ `phongdaotao@uit.edu.vn`.
- 🧠 **95.0% Answer Groundedness**: 19/20 câu hỏi đánh giá độc lập bám sát 100% văn bản quy phạm, không có suy diễn ngoài căn cứ pháp lý.

---

## ⚙️ Cấu hình môi trường (`.env`)

Hệ thống đã cấu hình giá trị mặc định tối ưu. Chỉ cần điều chỉnh các biến cốt lõi sau:

| Biến môi trường | Bắt buộc | Mặc định / Mẫu | Mô tả |
|:---|:---:|:---:|:---|
| `OPENAI_API_KEY` | **Có** | `sk-...` | Khóa API của LLM (OpenAI, OpenRouter, Groq...) |
| `OPENAI_BASE_URL` | Tùy chọn | `https://api.openai.com/v1` | URL endpoint (đổi nếu chạy vLLM / Ollama nội bộ) |
| `OPENAI_MODEL` | **Có** | `gpt-4o-mini` | Tên mô hình LLM dùng sinh câu trả lời |
| `POSTGRES_PASSWORD` | Tùy chọn | `change_me` | Mật khẩu truy cập cơ sở dữ liệu PostgreSQL |
| `POSTGRES_PORT` | Tùy chọn | `5433` (Docker host) | Cổng kết nối PostgreSQL pgvector |

> 💡 *Toàn bộ các biến nâng cao khác (mô hình Embedding `BAAI/bge-m3`, Cross-Encoder Reranker, Tesseract OCR, kích thước chunking...) đều đã được tinh chỉnh sẵn trong mã nguồn. Xem chi tiết tại [.env.example](.env.example).*

---

## 🧠 Quyết định kỹ thuật & Bài học thực tế (Architectural Decisions)

1. **RRF ($k=60$) thay vì phụ thuộc hoàn toàn vào Vector Search**:
   - Các câu hỏi quy chế học vụ chứa nhiều từ khóa viết tắt hoặc thuật ngữ riêng (`ĐTBCTL`, `CTĐT`, `TOEIC 450`, `điểm F`). Vector embedding thường bắt trượt các mã này, trong khi BM25 (với bộ tách từ tiếng Việt `underthesea`) giải quyết triệt để. RRF dung hòa hoàn hảo hai thế mạnh mà không cần chuẩn hóa điểm số phức tạp.
2. **Deterministic Python Validator thay vì tin cậy vào LLM Self-Evaluation**:
   - Dùng LLM để tự kiểm tra trích dẫn của chính nó vẫn tiềm ẩn nguy cơ bị ảo giác (hallucination loop). UITBOT sử dụng regex trích xuất thẻ trích dẫn và đối soát set matching trực tiếp với ID/metadata của các chunk trong DB, đảm bảo độ tin cậy tuyệt đối 100%.
3. **Chiến lược Evidence Gate trước khi sinh câu trả lời**:
   - Trong lĩnh vực quy chế đại học, một câu trả lời sai hoặc suy diễn bừa bãi có thể ảnh hưởng trực tiếp đến kết quả tốt nghiệp của sinh viên. Nếu điểm rerank $< 0.02$, hệ thống lập tức ngắt sớm, từ chối an toàn và cung cấp địa chỉ email chính thức của Phòng Đào tạo.

---

## ⚠️ Giới hạn kỹ thuật đã biết (Known Limitations)

- **Văn bản scan chất lượng thấp**: Các văn bản cũ có dấu mộc đỏ đè lên chữ viết hoặc độ tương phản kém vẫn phụ thuộc vào độ chính xác nhận dạng ký tự của Tesseract OCR.
- **Độ trễ Cross-Encoder trên CPU**: Mô hình `bge-reranker-v2-m3` trên CPU có độ trễ ~5.5s. Với các môi trường không có GPU, chế độ **Hybrid (291ms)** được khuyến nghị làm lựa chọn mặc định cân bằng tốt nhất giữa tốc độ và độ chính xác.
- **Tính thời điểm của văn bản (Temporal Validity)**: Chưa tự động phân giải xung đột khi có Thông tư/Quyết định mới sửa đổi, bổ sung cho Quyết định cũ; hiện tại cần quản lý qua siêu dữ liệu `effective_date`.

---

## 📂 Cấu trúc dự án (Repository Structure)

```text
UITBOT/
├── .github/workflows/ci.yml # CI Pipeline: Ruff Lint/Format & 53 Unit Tests (100% Pass)
├── app/
│   ├── api/                 # Endpoints: /chat, /chat/stream (SSE), /feedback, /health
│   ├── db/                  # Connection pool & migration bảng pgvector HNSW
│   ├── generation/          # Answer generator, Evidence Gate, Citation Validator, Router
│   ├── ingestion/           # Pipeline trích xuất PDF, OCR Tesseract, Chunking, BGE-M3 Embeddings
│   ├── retrieval/           # Dense Search, BM25 Okapi, RRF Fusion, Cross-Encoder Reranker
│   └── static/              # Giao diện Web Responsive thương hiệu UIT (Vanilla HTML/CSS/JS)
├── data/raw/                # 11 văn bản quy chế PDF gốc được ban hành của UIT
├── docs/                    # Tài liệu chuyên sâu kiến trúc & quyết định:
│   ├── ARCHITECTURE.md      # Chi tiết 4 tầng kiến trúc & biểu đồ luồng dữ liệu
│   ├── DECISIONS.md         # Toàn bộ Architecture Decision Records (ADRs)
│   └── DEPLOYMENT_GUIDE.md  # Hướng dẫn triển khai Production trên Cloud VPS Ubuntu
├── evaluation/              # golden_dataset.jsonl (105 câu), failure_analysis.csv, results.md
├── scripts/                 # evaluate_all.py, ingest.py, run_failure_analysis.py
├── tests/                   # 53 Automated Tests kiểm thử toàn diện
├── docker-compose.yml       # Cụm container app (non-root uitbot) + db (pgvector)
├── Dockerfile               # Multi-stage build với PyTorch CPU & Tesseract OCR vie
└── requirements.txt         # Pinned production dependencies
```

---

## 📄 Bản quyền (License)

Dự án được phân phối theo giấy phép mã nguồn mở **MIT License**. Xem chi tiết tại [LICENSE](LICENSE).
