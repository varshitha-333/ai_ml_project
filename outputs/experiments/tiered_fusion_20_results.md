# Tiered Multi-Model Candidate Fusion (K <= 20) Evaluation Report

---

## 🏆 Executive Summary Metrics
- **End-to-End System Accuracy**: **`40.0%`**
- **Parse Success Rate**: **`100.0%`** (`30/30`)
- **Fused Candidate Recall@10**: **`16.67%`**
- **Fused Candidate Recall@20**: **`40.0%`**
- **Fused MRR**: **`0.0556`**
- **Mean Latency**: **`42769.79 ms`** | P95: **`43878.29 ms`**

---

## 📌 Architecture Strategy Summary
Fused Top-10 Short/Unigram Candidates (BM25 Lexical) + Top-10 Multi-Word Semantic Candidates (Solution 1 Dense) into a deduplicated candidate pool ($\le 20$ candidates).

---

## 💻 Exact Command to Reproduce:

```powershell
.\venv\Scripts\python.exe scratch/run_tiered_fusion_20.py
```
