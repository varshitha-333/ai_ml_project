# External 150-Case Validation Report

**Execution Timestamp**: 2026-08-28 06:21:19 UTC
**Backend Mode**: `MOCK` (MockInferenceBackend)
**Model Name**: `Qwen/Qwen2.5-7B-Instruct (Mock)`
**Retrieval K**: `30`
**Total Dataset Cases**: `150`

> [!NOTE]
> **Mode Notice**: `MOCK` mode active. Mock results are NOT model-quality measurements.

---

## 1. Summary Metrics

```text
BACKEND MODE: MOCK
MODEL: Qwen/Qwen2.5-7B-Instruct (Mock)
RETRIEVAL K: 30
TOTAL CASES: 150
MODEL CALLS: 110
ABSTENTIONS: 112
SCORED: 38
INFERENCE ERRORS: 0
AVERAGE LATENCY: 0.02s
MEDIAN LATENCY: 0.02s
P95 LATENCY: 0.05s

Status accuracy: 28.67%
Score exact accuracy: 8.97%
Score MAE: 0.68
Score ±1 accuracy: 25.52%

Abstention precision: 4.46%
Abstention recall: 100.0%
Abstention F1: 8.54%

False scoring rate: 0.0%
Hallucination false scoring rate: 0.0%

Recall@1: 36.0%
Recall@5: 48.67%
Recall@10: 60.0%
Recall@20: 69.33%
Recall@30: 78.67%
```

---

## 2. Structured Failure Summary

```text
RETRIEVAL MISS:           32
WRONG SCORING:            26
INCORRECT ABSTENTION:     0
FALSE SCORING:            0
INFERENCE ERROR:          0
MOCK BACKEND LIMITATION:  39
ANNOTATION ISSUE:         0
```

---

## 3. Hallucination Trap Test Audit

| Case ID | Target Facet | Conversation | Expected Status | Predicted Status | Outcome |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `TEST_032` | `FSH level` | *"I do not have a laboratory result available for this value."* | `not_observable` | `not_observable` | **`PASS`** |
| `TEST_043` | `Pilgrimage participation count` | *"I don't have records showing how many pilgrimages this person attended."* | `not_observable` | `not_observable` | **`PASS`** |
| `TEST_083` | `Sleep-disorder diagnosis` | *"I have no clinical record that would justify diagnosing a sleep disorder from this conversation."* | `not_observable` | `not_observable` | **`PASS`** |
| `TEST_128` | `Intelligence Quotient (IQ)` | *"I don't have access to this person's IQ test result, so I cannot infer their IQ from this sentence."* | `not_observable` | `not_observable` | **`PASS`** |
| `TEST_129` | `Sleep Apnea` | *"I do not have medical records showing whether this person has sleep apnea."* | `not_observable` | `not_observable` | **`PASS`** |

---

## 4. Final System Verdict

**System Classification Verdict**: **`NEEDS IMPROVEMENT`**
