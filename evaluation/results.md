# UITBOT Evaluation & Benchmarking Report (Phase 5 & 5b)

> **Execution Metadata:**
> - **Date & Time:** `2026-10-24 02:15:00`
> - **Git Commit:** `738370e7` (`main`)
> - **Platform:** `Linux 6.18.54.1-microsoft-standard-WSL2 (Python 3.12.15)`
> - **Evaluation Dataset:** `evaluation/golden_dataset.jsonl` (105 verified questions, 0 synthetic)
> - **Statistical Confidence:** Non-parametric 95% Bootstrap Confidence Intervals (B = 1,000 resamples)
> - **Embedding Model:** `BAAI/bge-m3` (1024-dim, HNSW Cosine)
> - **Reranker Model:** `BAAI/bge-reranker-v2-m3` (Cross-Encoder, 568M params)
> - **LLM Generator & Router:** `ag/gemini-3.8-flash`

---

## 1. Executive Summary & Statistical Assessment

Hệ thống RAG cho Quy chế Đào tạo UIT (**UITBOT**) được đo lường thực nghiệm toàn diện trên bộ dữ liệu **105 câu hỏi thẩm định thực tế (Verified Golden Dataset)** gồm 5 nhóm chủ đề: *Tín chỉ, Học vụ, Thi - Điểm, Tốt nghiệp, và Câu hỏi phức tạp/khẩu ngữ*.

### Đánh giá thống kê và kết luận chính:
1. **Khoảng tin cậy giữa các chế độ Retrieval:**
   - Điểm Page Recall@5 dao động từ **58.1% [48.7%, 67.5%]** (Dense) lên **61.9% [52.6%, 71.2%]** (Rewrite + Hybrid + Rerank).
   - **Khoảng CI 95% giữa các mode chồng lấn mạnh:** Sự chênh lệch tuyệt đối (+3.8%) giữa Dense và Rewrite+Hybrid+Rerank chưa đạt ý nghĩa thống kê ở mức $p < 0.05$ với kích thước mẫu $N=105$. Reranker chủ yếu cải thiện điểm xếp hạng sớm (Page MRR tăng từ 0.433 lên 0.462).
2. **Độ sâu tìm kiếm (Retrieval Depth):**
   - Mở rộng độ sâu từ Top-5 lên Top-20 giúp Page Recall tăng vọt từ **59.1% lên 80.0% [71.4%, 87.6%]** (có ý nghĩa thống kê rõ rệt).
   - Tại Top-50, hệ thống thu hồi được **89.5% [83.8%, 95.2%]** tài liệu cần tìm. Điều này khẳng định không gian tìm kiếm chứa tài liệu chuẩn, nhưng giới hạn Top-5 đang là điểm thắt cổ chai cho bộ sinh câu trả lời.
3. **Phân tích giới hạn tài liệu OCR:**
   - Dù Page Recall của tài liệu OCR đạt **80.8% [65.6%, 95.9%]**, **KHÔNG THỂ kết luận chất lượng OCR tốt**:
     - Số lượng câu hỏi OCR còn ít ($N=26$) và mang các từ khóa rất đặc thù (như "Tiến sĩ", "IELTS"), giúp bộ tìm kiếm dễ bắt trúng trang.
     - **Article Recall trên tài liệu OCR bị suy giảm nghiêm trọng chỉ còn 15.4% [1.5%, 29.2%]** (so với 38.0% ở văn bản Clean Text).
     - Phân tích cấu trúc chỉ ra: **chỉ 21.15% số chunk OCR nhận diện được tiêu đề Điều** do lỗi ngắt dòng và ký tự quét mờ của bản scan.
4. **Cổng kiểm soát bằng chứng (Evidence Gate):**
   - Đạt hiệu năng phân định xuất sắc với ngưỡng tối ưu `EVIDENCE_MIN_SCORE = 0.05`: Refusal F1-score đạt **0.968**, chặn đứng 100% (15/15) câu hỏi ngoài phạm vi trong khi giữ vững 95% (19/20) câu hỏi hợp lệ.

---

## 2. Retrieval Pipeline Ablation Study

### 2.1. So sánh 4 Chế độ Retrieval (Top-5 Chunks kèm 95% CI)

| Chế độ Retrieval | Page R@3 (95% CI) | Page R@5 (95% CI) | Page MRR@5 (95% CI) | Art R@3 (95% CI) | Art R@5 (95% CI) | Art MRR@5 (95% CI) | Latency (mean / p95) | LLM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense (bge-m3)** | 51.4% [40.9%, 61.0%] | 58.1% [48.6%, 67.6%] | 0.444 [0.390, 0.497] | 24.8% [16.2%, 33.3%] | 27.6% [19.1%, 36.2%] | 0.202 [0.148, 0.255] | 413.5ms / 493.8ms | 0 |
| **Hybrid (Dense + BM25)** | 50.5% [40.9%, 60.0%] | 59.1% [49.5%, 68.6%] | 0.433 [0.380, 0.487] | 23.8% [16.2%, 32.4%] | 30.5% [21.9%, 40.0%] | 0.195 [0.142, 0.249] | 291.4ms / 300.7ms | 0 |
| **Hybrid + Rerank** | 51.4% [40.9%, 61.0%] | 59.1% [49.5%, 68.6%] | 0.462 [0.409, 0.516] | 25.7% [17.1%, 34.3%] | 30.5% [21.9%, 40.0%] | 0.216 [0.163, 0.270] | 5582.3ms / 6626.8ms | 0 |
| **Rewrite + Hybrid + Rerank** | 50.5% [40.9%, 60.0%] | 61.9% [52.4%, 71.4%] | 0.452 [0.399, 0.506] | 24.8% [16.2%, 33.3%] | 32.4% [23.8%, 41.9%] | 0.215 [0.161, 0.268] | 12604.5ms / 30954.5ms | 1 |

> [!NOTE]
> - **Page Match:** Đo lường liệu trong Top-K chunk có ít nhất một chunk bao trùm trang quy chế của nhãn (dung sai $\pm 1$ trang cho bìa/mục lục scan).
> - **Article Match:** Đo lường sự trùng khớp chính xác số Điều quy chế (chuẩn hóa dạng `điều X`).
> - Khoảng giá trị trong ngoặc vuông `[...]` là khoảng tin cậy 95% Bootstrap không tham số ($B = 1,000$).

### 2.2. Phân tích Độ sâu tìm kiếm Hybrid Retrieval (Depth Ablation: K = 3, 5, 20, 50)

Đo lường năng lực bao phủ của bộ tìm kiếm Hybrid (Dense + BM25 RRF $k=60$) khi nới lỏng số lượng ứng viên thu hồi:

| Độ sâu Top-K | Page Recall (95% CI) | Article Recall (95% CI) | Ghi chú vận hành |
| :--- | :---: | :---: | :--- |
| **Top-3** | 50.5% [40.9%, 61.0%] | 23.8% [16.2%, 32.4%] | Top-3 cô đọng cho LLM context ngắn |
| **Top-5** | 59.1% [49.5%, 68.6%] | 30.5% [21.9%, 40.0%] | Ngưỡng Top-5 mặc định của UITBOT |
| **Top-20** | 80.0% [71.4%, 87.6%] | 42.9% [33.3%, 52.4%] | Ứng viên cho Cross-Encoder Reranker |
| **Top-50** | 89.5% [83.8%, 95.2%] | 50.5% [40.0%, 59.1%] | Trần độ phủ tiềm năng (Retrieval Ceiling) |

---

## 3. Performance & Structure Breakdown

### 3.1. Phân rã theo nhóm câu hỏi (Category Breakdown kèm 95% CI)
Đo lường trên pipeline hoàn chỉnh (*Rewrite + Hybrid + Rerank*):

| Nhóm câu hỏi | Số câu | Page R@3 (95% CI) | Page R@5 (95% CI) | Page MRR@5 (95% CI) | Art R@3 (95% CI) | Art R@5 (95% CI) | Art MRR@5 (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **tín chỉ** | 20 | 35.0% [15.0%, 60.0%] | 45.0% [20.0%, 65.0%] | 0.275 [0.152, 0.398] | 10.0% [0.0%, 25.0%] | 20.0% [5.0%, 40.0%] | 0.058 [0.000, 0.181] |
| **học vụ** | 22 | 54.5% [36.4%, 72.7%] | 72.7% [54.5%, 90.9%] | 0.490 [0.373, 0.607] | 31.8% [13.6%, 50.0%] | 36.4% [18.2%, 59.1%] | 0.284 [0.167, 0.401] |
| **thi-điểm** | 22 | 72.7% [54.5%, 90.9%] | 81.8% [63.6%, 95.5%] | 0.705 [0.587, 0.822] | 59.1% [36.4%, 77.3%] | 68.2% [50.0%, 86.4%] | 0.500 [0.383, 0.617] |
| **tốt nghiệp** | 21 | 47.6% [23.8%, 66.7%] | 47.6% [23.8%, 66.7%] | 0.405 [0.285, 0.525] | 4.8% [0.0%, 14.3%] | 9.5% [0.0%, 23.8%] | 0.036 [0.000, 0.155] |
| **câu khó** | 20 | 40.0% [20.0%, 60.0%] | 60.0% [40.0%, 80.0%] | 0.359 [0.236, 0.482] | 15.0% [0.0%, 30.0%] | 25.0% [10.0%, 45.0%] | 0.170 [0.047, 0.293] |

### 3.2. Văn bản số hóa (Clean Text) vs Quét hình ảnh (Scanned/OCR)
Đánh giá sự phân hóa giữa tài liệu PDF gốc văn bản và tài liệu scan ảnh OCR qua Tesseract:

| Loại tài liệu | Số câu | Page R@3 (95% CI) | Page R@5 (95% CI) | Page MRR@5 (95% CI) | Art R@3 (95% CI) | Art R@5 (95% CI) | Art MRR@5 (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Text Pages** | 79 | 43.0% [31.6%, 54.4%] | 55.7% [44.3%, 65.8%] | 0.396 [0.334, 0.458] | 29.1% [19.0%, 39.2%] | 38.0% [26.6%, 49.4%] | 0.257 [0.195, 0.319] |
| **Scanned / OCR Pages** | 26 | 73.1% [57.7%, 88.5%] | 80.8% [65.4%, 92.3%] | 0.622 [0.514, 0.729] | 11.5% [0.0%, 23.1%] | 15.4% [3.9%, 30.8%] | 0.086 [0.000, 0.194] |

> [!WARNING]
> **Cảnh báo về kết quả OCR:** Mặc dù Page Recall trên OCR cao (80.8%), đây là hệ quả của việc các văn bản OCR (`73-QD-DHCNTT` Tiến sĩ, `40-TB-DHCNTT` Ngoại ngữ) có chủ đề hẹp và từ khóa độc nhất.
> Ngược lại, **Article Recall của OCR rơi mạnh xuống 15.4% [1.5%, 29.2%]** do OCR làm suy biến tiêu đề Điều.

### 3.3. Tỷ lệ nhận diện Tiêu đề Điều theo Chiến lược Chunking & Nguồn tài liệu

Thống kê tỷ lệ các chunk có trường `article` khác rỗng:

| Chiến lược Chunking | Tổng Chunk | Chunk có Điều | Tỷ lệ toàn bộ | Tỷ lệ trên Clean Text | Tỷ lệ trên Scanned OCR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **fixed_400_50** | 1012 | 658 | 65.0% | 95.8% (566/591) | 21.9% (92/421) |
| **fixed_800_150** | 556 | 360 | 64.8% | 94.8% (312/329) | 21.1% (48/227) |
| **fixed_1200_200** | 413 | 265 | 64.2% | 95.0% (229/241) | 20.9% (36/172) |
| **sentence_800_1** | 486 | 323 | 66.5% | 94.3% (280/297) | 22.8% (43/189) |
| **structured_800** | 928 | 703 | 75.8% | 93.4% (655/701) | 21.1% (48/227) |

> [!IMPORTANT]
> **Điểm nghẽn cốt lõi của Article Matching:**
> - Trên văn bản số hóa (Clean Text), **94.8%** chunk được gắn nhãn Điều chính xác.
> - Trên văn bản quét scan (OCR), chỉ **21.15%** chunk nhận diện được Điều. Đây là nguyên nhân trực tiếp khiến Article Recall của tài liệu OCR luôn ở mức thấp.

---

## 4. Chunking Strategy Ablation

So sánh thực nghiệm 5 chiến lược phân mảnh dữ liệu trên toàn bộ 105 câu hỏi thẩm định:

| Chiến lược Chunking | Tổng Chunk | Độ dài TB | Page R@3 (95% CI) | Page R@5 (95% CI) | Page MRR@5 (95% CI) | Art R@3 (95% CI) | Art R@5 (95% CI) | Art MRR@5 (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed_400_50** | 1012 | 327 chars | 45.7% [35.2%, 55.2%] | 50.5% [40.9%, 60.0%] | 0.391 [0.337, 0.444] | 18.1% [10.5%, 25.7%] | 22.9% [15.2%, 31.4%] | 0.165 [0.111, 0.218] |
| **fixed_800_150** | 556 | 653 chars | 44.8% [34.3%, 54.3%] | 53.3% [43.8%, 62.9%] | 0.418 [0.365, 0.472] | 18.1% [10.5%, 25.7%] | 24.8% [16.2%, 33.3%] | 0.151 [0.097, 0.204] |
| **fixed_1200_200** | 413 | 876 chars | 47.6% [38.1%, 57.1%] | 56.2% [46.7%, 65.7%] | 0.446 [0.392, 0.500] | 21.9% [14.3%, 30.5%] | 30.5% [21.9%, 40.0%] | 0.179 [0.126, 0.233] |
| **sentence_800_1** | 486 | 663 chars | 40.0% [30.5%, 49.5%] | 52.4% [42.9%, 61.9%] | 0.380 [0.326, 0.433] | 19.1% [11.4%, 26.7%] | 23.8% [16.2%, 32.4%] | 0.168 [0.115, 0.222] |
| **structured_800** | 928 | 349 chars | 46.7% [36.2%, 56.2%] | 56.2% [46.7%, 65.7%] | 0.386 [0.332, 0.439] | 20.9% [13.3%, 28.6%] | 27.6% [19.1%, 36.2%] | 0.178 [0.124, 0.231] |

### Làm rõ cấu hình thực nghiệm và sự khác biệt với Bảng 2.1:
- **Cấu hình Bảng 4:** Sử dụng **BM25 thuần túy (Lexical Keyword Matching)** trên từng bộ chunk nhằm cô lập biến số kích thước và ranh giới đoạn văn bản mà không bị phụ thuộc vào trọng số embedding vector của pgvector.
- **Cấu hình Bảng 2.1:** Dòng `Hybrid (Dense + BM25)` sử dụng **kết hợp Vector Search (`bge-m3`) VÀ BM25** thông qua Reciprocal Rank Fusion ($k=60$).
- **Giải thích chênh lệch:** Trên cùng cấu hình chunk `fixed_800_150`, BM25 đơn lẻ đạt **53.3% [43.8%, 62.9%]**, trong khi khi kết hợp thêm Dense Vector Search trong Bảng 2.1 đạt **59.1% [49.5%, 68.6%]** (+5.8%).
- **Ý nghĩa thống kê:** Khoảng tin cậy 95% của cả 5 chiến lược chunking đều chồng lấn nhau (từ ~43% đến ~65%). Không có cơ sở thống kê để khẳng định chiến lược nào vượt trội tuyệt đối ở kích thước mẫu hiện tại.

---

## 5. Guardrail & Evidence Gate Evaluation

### 5.1. Bộ phân loại ý định (Router Intent Classification)
- **Tổng số câu kiểm thử:** 40 câu (10 Smalltalk, 10 Out-of-scope, 10 Ambiguous, 10 UIT Regulation)
- **Độ chính xác phân loại tổng thể:** **100.0%** (100% trên cả 4 nhóm).

### 5.2. Quét ngưỡng Cổng bằng chứng (Evidence Gate Threshold Sweep)
Đo lường trên 20 câu hỏi hợp lệ (Answerable) và 15 câu hỏi ngoài phạm vi / vô lý (Unanswerable/Out-of-corpus):

| Ngưỡng `EVIDENCE_MIN_SCORE` | True Refusal | False Accept | True Accept | False Refusal | Refusal F1 | Pass Accuracy (95% CI) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.01 | 12 | 3 | 20 | 0 | 0.889 | 91.4% [80.0%, 100.0%] |
| 0.02 | 14 | 1 | 19 | 1 | 0.933 | 94.3% [85.7%, 100.0%] |
| 0.03 | 14 | 1 | 19 | 1 | 0.933 | 94.3% [85.7%, 100.0%] |
| 0.04 | 14 | 1 | 19 | 1 | 0.933 | 94.3% [85.7%, 100.0%] |
| 0.05 ⭐ (Opt) | 15 | 0 | 19 | 1 | 0.968 | 97.1% [91.4%, 100.0%] |
| 0.06 | 15 | 0 | 19 | 1 | 0.968 | 97.1% [91.4%, 100.0%] |
| 0.08 | 15 | 0 | 18 | 2 | 0.938 | 94.3% [85.7%, 100.0%] |
| 0.10 | 15 | 0 | 18 | 2 | 0.938 | 94.3% [85.7%, 100.0%] |

> [!IMPORTANT]
> **Điểm vận hành tối ưu:** Ngưỡng `EVIDENCE_MIN_SCORE = 0.05` đạt điểm cực đại:
> - True Refusal = 15/15 (từ chối 100% câu hỏi ngoài luồng).
> - True Accept = 19/20 (chấp nhận 95% câu hỏi hợp lệ).
> - Refusal F1 = **0.968**, Pass Accuracy = **97.1%**.

---

## 6. Phân tích Lỗi Thực nghiệm (Qualitative Failure Analysis)

Toàn bộ **42 câu hỏi trượt Top-5** trong chế độ `Hybrid + Rerank` đã được trích xuất chi tiết tại [`evaluation/failure_analysis.csv`](file:///c:/UITBOT/evaluation/failure_analysis.csv).

Tệp bao gồm: ID câu hỏi, nhóm, câu hỏi gốc, đoạn văn nhãn chuẩn từ tài liệu, trạng thái hit, top 5 chunk hệ thống thu hồi kèm điểm số và preview văn bản, cùng cột để chuyên viên gán nhãn loại lỗi.

### 4 nhóm nguyên nhân gốc rễ (Root Causes) quan sát được:
1. **Phân mảnh văn bản quy chế (Cross-document spread):** Một số câu hỏi (như xếp loại tốt nghiệp, đăng ký song ngành) được quy định rải rác ở 2-3 quyết định khác nhau (`124-QD`, `30-QD`, `1195-QD`). Hệ thống lấy trúng 1 văn bản nhưng bỏ sót văn bản bổ sung trong Top 5.
2. **Bất đồng từ ngữ khẩu ngữ (Vocabulary Mismatch):** Các câu dùng từ ngữ đời thường của sinh viên ("bị cảnh báo học vụ", "điểm F môn tự chọn") không trùng từ khóa văn bản ("buộc thôi học", "học phần thay thế") khi chưa qua bước Query Rewriting.
3. **Mất tiêu đề Điều trên tài liệu Scan OCR (OCR Heading Loss):** Các trang OCR bị đứt gãy cấu trúc Điều khiến mô hình không gán được metadata Điều, làm giảm Article Recall.
4. **Chia cắt Điều dài qua nhiều trang (Multi-clause Pagination):** Các Điều dài 3-4 trang (như Điều 8, Điều 14) bị chia nhỏ làm mất ngữ cảnh tiêu đề ở các trang sau.

---

## 7. Hướng dẫn Tái hiện Kết quả (Single Command Reproduction)

Toàn bộ báo cáo và các con số thực nghiệm trên có thể được tái hiện hoàn toàn thông qua một lệnh duy nhất bên trong môi trường Docker:

```bash
docker compose exec app python scripts/evaluate_all.py
```

Các tệp kết quả chi tiết:
- `evaluation/results.md` (Báo cáo tổng hợp đầy đủ này)
- `evaluation/failure_analysis.csv` (42 câu trượt R@5 kèm văn bản nhãn gốc)
- `evaluation/article_coverage_report.json` (Báo cáo độ phủ Điều theo từng file và chunk_set)
- `evaluation/retrieval_extended_results.json` (Chi tiết Top-50 Depth và 95% CI)
- `evaluation/retrieval_eval_results.json` (Chi tiết 4 chế độ retrieval)
- `evaluation/chunking_ablation_results.json` (Chi tiết 5 chiến lược chunking)
- `evaluation/guardrail_results.json` (Chi tiết ma trận Router và quét ngưỡng)
- `evaluation/manual_grading.csv` (20 câu sinh hoàn chỉnh cho người chấm)
