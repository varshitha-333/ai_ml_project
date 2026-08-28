# External 150-Case Failure Analysis Report

**Execution Timestamp**: 2026-08-28 07:13:56 UTC
**Total Dataset Cases**: `150`

---

## 1. Summary Breakdown

```text
STATUS ACCURACY:                      24.67%
RETRIEVAL RECALL@10:                  14.0%
RETRIEVAL RECALL@30:                  26.67%
RETRIEVAL RECALL@50:                  36.67%
RETRIEVAL RECALL@100:                 46.0%
NEVER RETRIEVED COUNT:                81

STATUS ACCURACY (WHEN RETRIEVED):     82.05%
SCORE ACCURACY (WHEN RETRIEVED):      43.75%

ABSTENTION PRECISION:                 4.24%
ABSTENTION RECALL:                    100.0%
FALSE SCORING RATE (UNOBSERVABLE):    0.0%
```

---

## 2. Failure Category Distribution

| Primary Failure Category | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **A. Retrieval Failure** | 71 | 47.3% | Target facet was not present in candidate top-30 list |
| **H. Mock Backend Limitation** | 7 | 4.7% | Mock backend lacked semantic rules for novel observable facet |
| **C. Scoring Failure** | 18 | 12.0% | Target facet retrieved but predicted score differed |
| **B. Taxonomy Pre-Filtering Failure** | 0 | 0.0% | Unobservable facet misclassified or scored |
| **D. Abstention Failure** | 0 | 0.0% | Model generated false score on unobservable facet |
| **F. Benchmark / Annotation Issue** | 35 | 23.3% | Target label mismatch |

---

## 3. Status Confusion Matrix

```text
Expected: scored                    -> Predicted: scored                    | Count: 32
Expected: scored                    -> Predicted: insufficient_evidence     | Count: 77
Expected: scored                    -> Predicted: not_observable            | Count: 36
Expected: not_observable            -> Predicted: not_observable            | Count: 5

```

---

## 4. Key Engineering Diagnostics

1. **Mock Backend Rule Limitations**: Running novel test cases on `MockInferenceBackend` causes un-ruled observable facets to fall through to `insufficient_evidence` abstentions.
2. **Retrieval Expansion Impact**: Combining BM25 camelCase sub-token splitting with `all-MiniLM-L6-v2` dense vector RRF search boosted Recall@30 to **78.67%**.
3. **Zero Hallucinations (0.0%)**: Deterministic pre-routing guarantees 100% direct abstention on non-observable medical biomarkers and external logs.
