# 150-Case External Validation Report

**Execution Timestamp**: 2026-08-28 06:04:25 UTC
**Backend Mode**: `MOCK` (MockInferenceBackend)
**Model Name**: `Qwen/Qwen2.5-7B-Instruct (Mock)`
**Retrieval K**: `30`
**Total Dataset Cases**: `150`

---

## 1. Executive Summary

```text
BACKEND MODE: MOCK
MODEL: Qwen/Qwen2.5-7B-Instruct (Mock)
RETRIEVAL K: 30
TOTAL CASES: 150
MODEL CALLS: 110
ABSTENTIONS: 118
SCORED: 32
INFERENCE ERRORS: 0
AVERAGE LATENCY: 0.0s
MEDIAN LATENCY: 0.0s
P95 LATENCY: 0.0s

Status accuracy: 24.67%
Score exact accuracy: 9.66%
Score MAE: 0.59
Score ±1 accuracy: 21.38%

Abstention precision: 4.24%
Abstention recall: 100.0%
Abstention F1: 8.14%

False scoring rate: 0.0%
Hallucination false scoring rate: 0.0%

Recall@1: 36.0%
Recall@5: 37.33%
Recall@10: 40.67%
Recall@20: 48.0%
Recall@30: 52.67%
```

---

## 2. Retrieval Candidate Depth Ablation (K=10 vs K=20 vs K=30)

| Candidate Depth K | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@30 | Avg Latency |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| K=10 | 9.33% | 10.67% | 14.0% | 14.0% | 14.0% | 0.76ms |
| K=20 | 9.33% | 10.67% | 14.0% | 21.33% | 21.33% | 0.8ms |
| K=30 | 9.33% | 10.67% | 14.0% | 21.33% | 26.67% | 0.79ms |

---

## 3. Hallucination Trap Test Audit

| Test ID | Facet | Conversation | Expected Status | Predicted Status | Outcome |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `TEST_032` | `FSH level` | *"I do not have a laboratory result available for this value."* | `not_observable` | `not_observable` | **`PASS`** |
| `TEST_043` | `Pilgrimage participation count` | *"I don't have records showing how many pilgrimages this person attended."* | `not_observable` | `not_observable` | **`PASS`** |
| `TEST_083` | `Sleep-disorder diagnosis` | *"I have no clinical record that would justify diagnosing a sleep disorder from this conversation."* | `not_observable` | `not_observable` | **`PASS`** |
| `TEST_128` | `Intelligence Quotient (IQ)` | *"I don't have access to this person's IQ test result, so I cannot infer their IQ from this sentence."* | `not_observable` | `not_observable` | **`PASS`** |
| `TEST_129` | `Sleep Apnea` | *"I do not have medical records showing whether this person has sleep apnea."* | `not_observable` | `not_observable` | **`PASS`** |

---

## 4. Overall Assessment

**System Classification Verdict**: **`NEEDS IMPROVEMENT`**
