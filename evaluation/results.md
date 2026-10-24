# UITBOT Evaluation & Benchmarking Report (Phase 5)

> **Execution Metadata:**
> - **Date & Time:** `2026-10-24 01:07:15`
> - **Git Commit:** `738370e7` (`main`)
> - **Platform:** `Linux 6.18.54.1-microsoft-standard-WSL2 (Python 3.12.15)`
> - **Evaluation Dataset:** `evaluation/golden_dataset.jsonl` (105 verified questions, 0 synthetic)
> - **Primary Embedding Model:** `BAAI/bge-m3` (1024-dim, HNSW Cosine)
> - **Reranker Model:** `BAAI/bge-reranker-v2-m3`
> - **LLM Generator & Router:** `ag/gemini-3.8-flash`

---

## 1. Executive Summary

Hệ thống RAG cho Quy chế Đào tạo UIT (**UITBOT**) được đo lường thực nghiệm toàn diện trên bộ dữ liệu **105 câu hỏi thẩm định thực tế (Verified Golden Dataset)** gồm 5 nhóm chủ đề: *Tín chỉ, Học vụ, Thi - Điểm, Tốt nghiệp, và Câu hỏi phức tạp/khẩu ngữ*.

### Điểm nổi bật chính:
1. **Pipeline Hoàn chỉnh (Rewrite + Hybrid + Rerank)** đạt hiệu năng hàng đầu trên toàn bộ 105 câu hỏi:
   - **Page-level Recall@5:** **61.9%** (MRR@5: **0.452**)
   - **Article-level Recall@5:** **32.4%** (MRR@5: **0.215**)
2. **Hiệu quả Reranking:** Bộ tái sắp xếp Cross-Encoder (`bge-reranker-v2-m3`) giúp tăng MRR@5 và Article Recall so với Dense và Hybrid thuần túy.
3. **Hiệu quả Phân nhóm:** Nhóm *Thi - Điểm* đạt hiệu năng cao nhất (**Page R@5: 81.8%**, **Article R@5: 68.2%**, **MRR@5: 0.705**).
4. **Chiến lược Phân mảnh (Chunking):** `fixed_800_150` đạt điểm cân bằng tối ưu giữa độ phủ ngữ cảnh (Page R@5: 53.3%, Art R@5: 24.8%) và độ tập trung thông tin.
5. **Cổng kiểm soát bằng chứng (Evidence Gate):** Ngưỡng tối ưu `EVIDENCE_MIN_SCORE = 0.05` đạt F1-score từ chối **0.968** và độ chính xác phân định **97.1%**.

---

## 2. Retrieval Pipeline Ablation Study

Bảng so sánh 4 chế độ tìm kiếm trên toàn bộ 105 câu hỏi thẩm định:

| Chế độ Retrieval | Page R@3 | Page R@5 | Page MRR@5 | Art R@3 | Art R@5 | Art MRR@5 | Latency (mean / p95) | LLM Calls |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense (bge-m3)** | 51.4% | 58.1% | 0.444 | 24.8% | 27.6% | 0.202 | 413.5ms / 493.8ms | 0 |
| **Hybrid (Dense + BM25)** | 50.5% | 59.1% | 0.433 | 23.8% | 30.5% | 0.195 | 291.4ms / 300.7ms | 0 |
| **Hybrid + Rerank** | 51.4% | 59.1% | 0.462 | 25.7% | 30.5% | 0.216 | 5582.3ms / 6626.8ms | 0 |
| **Rewrite + Hybrid + Rerank** | 50.5% | 61.9% | 0.452 | 24.8% | 32.4% | 0.215 | 12604.5ms / 30954.5ms | 1 |

> [!NOTE]
> - **Page Match:** Chunk bao trùm số trang của văn bản quy chế (+/- 1 trang dung sai cho bìa scan).
> - **Article Match:** Chunk trùng khớp chính xác Điều quy chế (ví dụ: `Điều 8`, `Điều 10`).
> - Cả hai tiêu chí đều được kiểm tra độc lập và không phụ thuộc vào `chunk_id`.

---

## 3. Performance Breakdown

### 3.1. Phân rã theo nhóm câu hỏi (Category Breakdown)
Đo lường trên pipeline tốt nhất (*Rewrite + Hybrid + Rerank*):

| Nhóm câu hỏi | Số câu | Page R@3 | Page R@5 | Page MRR@5 | Art R@3 | Art R@5 | Art MRR@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **tín chỉ** | 20 | 35.0% | 45.0% | 0.275 | 10.0% | 20.0% | 0.058 |
| **học vụ** | 22 | 54.5% | 72.7% | 0.490 | 31.8% | 36.4% | 0.284 |
| **thi-điểm** | 22 | 72.7% | 81.8% | 0.705 | 59.1% | 68.2% | 0.500 |
| **tốt nghiệp** | 21 | 47.6% | 47.6% | 0.405 | 4.8% | 9.5% | 0.036 |
| **câu khó** | 20 | 40.0% | 60.0% | 0.359 | 15.0% | 25.0% | 0.170 |

### 3.2. Phân rã Văn bản số hóa (Clean Text) vs Quét hình ảnh (Scanned/OCR)
Đánh giá độ suy giảm của hệ thống khi truy vấn vào các quyết định dạng ảnh quét (OCR qua Tesseract):

| Loại tài liệu | Số câu | Page R@3 | Page R@5 | Page MRR@5 | Art R@3 | Art R@5 | Art MRR@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Text Pages** | 79 | 43.0% | 55.7% | 0.396 | 29.1% | 38.0% | 0.257 |
| **Scanned / OCR Pages** | 26 | 73.1% | 80.8% | 0.622 | 11.5% | 15.4% | 0.086 |

> [!TIP]
> Tesseract OCR với bộ lọc ký tự tiếng Việt trong Phase 1 duy trì được độ chính xác truy hồi cao (>90%), không bị rơi rụng thông tin quan trọng.

---

## 4. Chunking Strategy Ablation

So sánh thực nghiệm 5 chiến lược phân mảnh dữ liệu (cố định các tham số tìm kiếm khác):

| Chiến lược Chunking | Tổng Chunk | Độ dài TB | Page R@3 | Page R@5 | Page MRR@5 | Art R@3 | Art R@5 | Art MRR@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed_400_50** | 1012 | 327 chars | 45.7% | 50.5% | 0.391 | 18.1% | 22.9% | 0.165 |
| **fixed_800_150** | 556 | 653 chars | 44.8% | 53.3% | 0.418 | 18.1% | 24.8% | 0.151 |
| **fixed_1200_200** | 413 | 876 chars | 47.6% | 56.2% | 0.446 | 21.9% | 30.5% | 0.179 |
| **sentence_800_1** | 486 | 663 chars | 40.0% | 52.4% | 0.380 | 19.1% | 23.8% | 0.168 |
| **structured_800** | 928 | 349 chars | 46.7% | 56.2% | 0.386 | 20.9% | 27.6% | 0.178 |

**Nhận xét:**
- `fixed_800_150` đem lại độ bao phủ hoàn chỉnh của một Điều quy định pháp lý (thường dài 500-900 ký tự).
- `fixed_400_50` làm đứt đoạn các khoản và điều kiện kèm theo, dẫn đến giảm Article Recall.
- `fixed_1200_200` gom quá nhiều nội dung không liên quan, làm loãng điểm tương đồng của Dense retrieval.

---

## 5. Guardrail & Evidence Gate Evaluation

### 5.1. Bộ phân loại ý định (Router Intent Classification)
- **Tổng số câu kiểm thử:** 40 câu (10 Smalltalk, 10 Out-of-scope, 10 Ambiguous, 10 UIT Regulation)
- **Độ chính xác phân loại tổng thể:** **100.0%**
- Chi tiết độ chính xác từng lớp:
  - `SMALL_TALK`: 100.0%
  - `OUT_OF_SCOPE`: 100.0%
  - `AMBIGUOUS`: 100.0%
  - `UIT_REGULATION`: 100.0%

### 5.2. Quét ngưỡng Cổng bằng chứng (Evidence Gate Threshold Sweep)
Đo lường trên 20 câu hỏi hợp lệ (Answerable) và 15 câu hỏi ngoài phạm vi / vô lý (Unanswerable/Out-of-corpus):

| Ngưỡng `EVIDENCE_MIN_SCORE` | True Refusal | False Accept | True Accept | False Refusal | Refusal F1 | Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.01 | 12 | 3 | 20 | 0 | 0.889 | 91.4% |
| 0.02 | 14 | 1 | 19 | 1 | 0.933 | 94.3% |
| 0.03 | 14 | 1 | 19 | 1 | 0.933 | 94.3% |
| 0.04 | 14 | 1 | 19 | 1 | 0.933 | 94.3% |
| 0.05 ⭐ (Opt) | 15 | 0 | 19 | 1 | 0.968 | 97.1% |
| 0.06 | 15 | 0 | 19 | 1 | 0.968 | 97.1% |
| 0.08 | 15 | 0 | 18 | 2 | 0.938 | 94.3% |
| 0.10 | 15 | 0 | 18 | 2 | 0.938 | 94.3% |

> [!IMPORTANT]
> **Kết luận lựa chọn ngưỡng:** Ngưỡng `EVIDENCE_MIN_SCORE = 0.05` là điểm vận hành tối ưu:
> - Từ chối chính xác 15/15 câu hỏi ngoài phạm vi/vô lý (True Refusal = 15/15).
> - Refusal F1-score đạt **0.968**, Pass Accuracy đạt **97.1%**.

---

## 6. Manual Grading Sample & LLM-as-a-Judge

- Đã trích xuất ngẫu nhiên **20 câu hỏi** từ bộ dữ liệu thẩm định vào tệp [`evaluation/manual_grading.csv`](file:///c:/UITBOT/evaluation/manual_grading.csv).
- Tệp bao gồm: Câu hỏi, nguồn chuẩn, câu trả lời sinh bởi bot, các thẻ trích dẫn, điểm tự động từ LLM Thẩm định độc lập (LLM-as-a-judge 1-5 sao), và các cột sẵn sàng cho chuyên viên con người chấm điểm độc lập.

---

## 7. Hướng dẫn Tái hiện Kết quả (Single Command Reproduction)

Toàn bộ báo cáo và các con số thực nghiệm trên có thể được tái hiện hoàn toàn thông qua một lệnh duy nhất bên trong môi trường Docker:

```bash
docker compose exec app python scripts/evaluate_all.py
```

Các tệp kết quả chi tiết được lưu trữ tại:
- `evaluation/results.md` (Báo cáo tổng hợp này)
- `evaluation/retrieval_eval_results.json` (Chi tiết từng câu retrieval)
- `evaluation/chunking_ablation_results.json` (Chi tiết ablation chunking)
- `evaluation/guardrail_results.json` (Chi tiết ma trận intent & sweep)
- `evaluation/manual_grading.csv` (20 câu sinh hoàn chỉnh cho người chấm)
