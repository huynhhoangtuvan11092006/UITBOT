# Retrieval Evaluation Results

## Evaluation Setup

- Dataset: 50 manually curated questions
- Corpus: 3 official UIT regulation documents
- Evaluation level: Page-level retrieval
- Metrics:
  - Recall@3
  - Recall@5
  - MRR@5

## Results

| Retrieval Method | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|
| Dense Retrieval | 36.00% | 42.00% | 0.3140 |
| Hybrid Search | 48.00% | 60.00% | 0.4247 |
| Hybrid Search + Reranker | **96.00%** | **96.00%** | **0.8600** |

## Analysis

Dense retrieval provides the initial baseline performance.

Hybrid Search combines dense retrieval with BM25 keyword retrieval and improves retrieval performance:

- Recall@3: 36% → 48%
- Recall@5: 42% → 60%
- MRR@5: 0.3140 → 0.4247

Adding the cross-encoder reranker produces the strongest performance:

- Recall@3: 96%
- Recall@5: 96%
- MRR@5: 0.8600

The results show that hybrid retrieval improves candidate retrieval compared with dense retrieval alone, while reranking significantly improves the ordering of relevant documents.

## Final Retrieval Pipeline

User Query
→ Query Rewriting
→ Dense Retrieval + BM25
→ Reciprocal Rank Fusion
→ Candidate Retrieval
→ Cross-Encoder Reranking
→ Top-K Context
→ LLM Answer Generation

## Notes

- Evaluation uses exact `(source_file, page)` matching.
- A retrieval result is counted as correct only when the expected source file and page appear within the evaluated Top-K results.
- The benchmark is based on the current 3-document UIT regulation corpus.