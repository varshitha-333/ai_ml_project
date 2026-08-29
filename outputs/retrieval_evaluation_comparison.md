# PHASE 3 — Retrieval Strategy Comparison Benchmark

**Total Test Queries**: `55`  
**Cross-Encoder Inference Latency Overhead**: `934.78 ms`

---

## 📊 Quantitative Strategy Performance Table

| Strategy | Recall@5 | Recall@10 | Recall@20 | Recall@30 | MRR | Top-1 Acc | Avg Latency | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 Lexical Only** | `7.88%` | `16.67%` | `39.09%` | `51.82%` | `0.0892` | `3.64%` | `0.71 ms` | `1.55 ms` |
| **Dense Vector Only** | `26.06%` | `29.7%` | `33.94%` | `41.21%` | `0.175` | `10.91%` | `575.21 ms` | `23.47 ms` |
| **Hybrid BM25 + Dense RRF** | **`29.7%`** | **`44.85%`** | **`57.27%`** | **`57.27%`** | **`0.1779`** | **`7.27%`** | `19.57 ms` | `22.73 ms` |
| **Hybrid RRF + Cross-Encoder** | `20.61%` | `26.67%` | `43.94%` | `50.0%` | `0.1619` | `9.09%` | `934.78 ms` | `506.64 ms` |

---

## 💡 Engineering Insights & Selection Rationale
- **Hybrid RRF Dominance**: Combined BM25 + Dense vector search via RRF provides the best trade-off, achieving the highest **MRR (`0.1779`)** and **Recall@10 (`44.85%`)**.
- **Cross-Encoder Rejection**: Off-the-shelf MS-MARCO Cross-Encoder weights degraded MRR from `0.1779` down to `0.1619` while adding **+934.78 ms latency per query**.
