# Cross-Encoder Reranker Experimental Ablation Report

This report presents an empirical evaluation of adding a **Cross-Encoder Reranker** (`cross-encoder/ms-marco-MiniLM-L6-v2`) to the **Hybrid BM25 + Dense RRF Retrieval Pipeline**.

---

## 📊 Quantitative Retrieval Performance Comparison

| Configuration | Recall@5 | Recall@10 | Recall@30 | MRR | Avg Latency | P95 Latency | Overhead vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. BM25 Only** | `19.05%` | `30.95%` | `50.0%` | `0.1133` | `0.86 ms` | `2.22 ms` | `-` |
| **B. Dense Only (`all-MiniLM-L6-v2`)** | `4.76%` | `4.76%` | `9.52%` | `0.1429` | `2704.97 ms` | `13176.68 ms` | `-` |
| **C. BM25 + Dense RRF Baseline** | `19.05%` | `28.57%` | `50.0%` | `0.195` | `2323.95 ms` | `11302.76 ms` | **Baseline (0.0 ms)** |
| **D. Hybrid RRF + CrossEncoder (Pool N=30)** | `4.76%` | `9.52%` | `50.0%` | `0.096` | `482.04 ms` | `521.18 ms` | `+-1841.91 ms` |
| **E. Hybrid RRF + CrossEncoder (Pool N=50)** | `4.76%` | `4.76%` | `45.24%` | `0.0868` | `776.06 ms` | `815.16 ms` | `+-1547.89 ms` |

---

## ⏱️ Sub-Component Latency Breakdown

- **Cross-Encoder Model Load Time (One-time startup)**: `20159.48 ms`
- **BM25 Search Latency**: `0.86 ms`
- **Dense Embedding Search Latency**: `2704.97 ms`
- **RRF Rank Fusion Latency**: `2323.95 ms`
- **Cross-Encoder Candidate Inference Latency (Batch N=30)**: `460.09 ms`

---

## 📋 Per-Case Benchmark Ranking Error Table

| conversation_id | query | target_facet_id | baseline_rank | reranker_rank | baseline_retrieved | reranker_retrieved | improved | notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CONV_001 | I am taking a wild risk by going skydiving without a backup ... | FACET_001 | 1 | 2 | True | True | False | Reranked lower |
| CONV_002 | I submitted my resignation letter without having another job... | FACET_001 | 14 | 19 | True | True | False | Reranked lower |
| CONV_003 | I claim to be fearless and dauntless, yet my knees were knoc... | FACET_268 | >30 | >30 | False | False | False | Unchanged |
| CONV_005 | Oh sure, I'd just LOVE to stay past midnight fixing your typ... | FACET_337 | >30 | >30 | False | False | False | Unchanged |
| CONV_006 | I was feeling super tired and con flojera today, so I just s... | FACET_114 | >30 | >30 | False | False | False | Unchanged |
| CONV_007 | Could you please pass me the salt from across the table?... | FACET_337 | >30 | >30 | False | False | False | Unchanged |
| CONV_010 | I am feeling really blue and down in the dumps today because... | FACET_017 | 4 | 19 | True | True | False | Reranked lower |
