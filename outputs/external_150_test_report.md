# External 150-Case Validation Report

**Execution Timestamp**: 2026-08-28 09:39:33 UTC
**Backend Mode**: `REMOTE` (RemoteInferenceClientBackend)
**Model Name**: `Qwen/Qwen2.5-7B-Instruct`
**Retrieval K**: `10`
**Total Dataset Cases**: `60`

> [!NOTE]
> **Mode Notice**: `REMOTE` mode active. Real Qwen GPU inference active.

---

## 1. Summary Metrics

```text
BACKEND MODE: REMOTE
MODEL: Qwen/Qwen2.5-7B-Instruct
RETRIEVAL K: 10
TOTAL CASES: 60
MODEL CALLS: 43
ABSTENTIONS: 50
SCORED: 10
INFERENCE ERRORS: 0
AVERAGE LATENCY: 17.7s
MEDIAN LATENCY: 25.6s
P95 LATENCY: 26.48s

Status accuracy: 20.0%
Score exact accuracy: 12.07%
Score MAE: 0.5
Score ±1 accuracy: 13.79%

Abstention precision: 4.08%
Abstention recall: 100.0%
Abstention F1: 7.84%

False scoring rate: 0.0%
Hallucination false scoring rate: 0.0%

Recall@1: 33.33%
Recall@5: 56.67%
Recall@10: 70.0%
Recall@20: 70.0%
Recall@30: 70.0%
```

---

## 2. Structured Failure Summary

```text
RETRIEVAL MISS:           18
WRONG SCORING:            4
INCORRECT ABSTENTION:     14
FALSE SCORING:            0
INFERENCE ERROR:          0
MOCK BACKEND LIMITATION:  0
ANNOTATION ISSUE:         0
```

---

## 3. Hallucination Trap Test Audit

| Case ID | Target Facet | Conversation | Expected Status | Predicted Status | Outcome |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `TEST_032` | `FSH level` | *"I do not have a laboratory result available for this value."* | `not_observable` | `not_observable` | **`PASS`** |
| `TEST_043` | `Pilgrimage participation count` | *"I don't have records showing how many pilgrimages this person attended."* | `not_observable` | `not_observable` | **`PASS`** |

---

## 4. Final System Verdict

**System Classification Verdict**: **`NEEDS IMPROVEMENT`**
