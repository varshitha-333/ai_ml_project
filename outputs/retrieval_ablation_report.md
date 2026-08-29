# Cross-Encoder Reranker Experimental Ablation Report

This report presents an empirical evaluation of adding a **Cross-Encoder Reranker** (`cross-encoder/ms-marco-MiniLM-L6-v2`) to the **Hybrid BM25 + Dense RRF Retrieval Pipeline**.

---

## 📊 Quantitative Retrieval Performance Comparison

| Configuration | Recall@5 | Recall@10 | Recall@30 | MRR | Avg Latency | P95 Latency | Overhead vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. BM25 Only** | `19.05%` | `30.95%` | `50.0%` | `0.1133` | `1.16 ms` | `2.01 ms` | `-` |
| **B. Dense Only (`all-MiniLM-L6-v2`)** | `4.76%` | `4.76%` | `23.81%` | `0.1488` | `992.5 ms` | `4781.31 ms` | `-` |
| **C. BM25 + Dense RRF Baseline** | `19.05%` | `28.57%` | `50.0%` | `0.1248` | `1156.71 ms` | `5578.39 ms` | **Baseline (0.0 ms)** |
| **D. Hybrid RRF + CrossEncoder (Pool N=30)** | `4.76%` | `9.52%` | `50.0%` | `0.0949` | `572.3 ms` | `631.26 ms` | `+-584.41 ms` |
| **E. Hybrid RRF + CrossEncoder (Pool N=50)** | `4.76%` | `4.76%` | `30.95%` | `0.0817` | `937.38 ms` | `1017.96 ms` | `+-219.33 ms` |

---

## ⏱️ Sub-Component Latency Breakdown

- **Cross-Encoder Model Load Time (One-time startup)**: `5849.64 ms`
- **BM25 Search Latency**: `1.16 ms`
- **Dense Embedding Search Latency**: `992.5 ms`
- **RRF Rank Fusion Latency**: `1156.71 ms`
- **Cross-Encoder Candidate Inference Latency (Batch N=30)**: `550.77 ms`

---

## 📋 Per-Case Benchmark Ranking Error Table

| conversation_id | query | target_facet_id | baseline_rank | reranker_rank | baseline_retrieved | reranker_retrieved | improved | notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CONV_001 | I am taking a wild risk by going skydiving without a backup ... | FACET_005 | 8 | 10 | True | True | False | Reranked lower |
| CONV_002 | I submitted my resignation letter without having another job... | FACET_001 | 12 | 20 | True | True | False | Reranked lower |
| CONV_003 | I claim to be fearless and dauntless, yet my knees were knoc... | FACET_268 | >30 | >30 | False | False | False | Unchanged |
| CONV_005 | Oh sure, I'd just LOVE to stay past midnight fixing your typ... | FACET_337 | >30 | >30 | False | False | False | Unchanged |
| CONV_006 | I was feeling super tired and con flojera today, so I just s... | FACET_114 | >30 | >30 | False | False | False | Unchanged |
| CONV_007 | Could you please pass me the salt from across the table?... | FACET_337 | >30 | >30 | False | False | False | Unchanged |
| CONV_010 | I am feeling really blue and down in the dumps today because... | FACET_017 | 4 | 18 | True | True | False | Reranked lower |
