# Solution 1 Evaluation Report: Multi-Example Conversational Utterance Indexing

---

## 🏆 Executive Summary
- **Recall@1**: **`16.67%`** (Jumping from `3.33%` up to `16.67%`)
- **Recall@5**: **`50.0%`** (Jumping from `40.0%` up to `50.0%`)
- **Recall@10**: **`53.33%`** (Baseline: `56.67%`)
- **Recall@30**: **`60.0%`** (Baseline: `73.33%`)
- **MRR**: **`0.3078`** (Baseline: `0.1706`)
- **nDCG@10**: **`0.3602`** (Baseline: `0.2562`)
- **P95 Latency**: **`28.63 ms`**

---

## 📊 Performance Comparison Table

| Metric | Current Production Baseline | Solution 1 (Multi-Example Indexing) | Gain / Change |
| :--- | :---: | :---: | :---: |
| **Recall@1** | `3.33%` | **`16.67%`** | **+13.34%** |
| **Recall@5** | `40.00%` | **`50.0%`** | **+10.0%** |
| **Recall@10** | **`56.67%`** | `53.33%` | `-3.34%` |
| **Recall@30** | **`73.33%`** | `60.0%` | `-13.33%` |
| **MRR** | `0.1706` | **`0.3078`** | **+0.1372 (+80.4%)** |
| **nDCG@10** | `0.2562` | **`0.3602`** | **+0.104** |
| **P95 Latency** | `23.40 ms` | **`28.63 ms`** | Fast sub-40ms |

---

## 💻 Exact Reproduction Command:

```powershell
python scratch/run_solution1_experiment.py
```
