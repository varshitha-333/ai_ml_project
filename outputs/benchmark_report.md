# Benchmark Evaluation Report - Human Reference Set

## 1. Executive Summary & Benchmark Metrics
- **Total Reference Pairs Evaluated**: `21`
- **Overall Status Classification Accuracy**: `61.9%`
- **Abstention Precision**: `55.56%`
- **Abstention Recall**: `100.0%`
- **Abstention F1 Score**: `71.43%`
- **False Scoring Rate on Unsupported Facets**: `0` instances
- **Score Mean Absolute Error (MAE)**: `0.0` (on scored reference cases)
- **Score Exact Match Percentage**: `100.0%`

---

## 2. Category Performance Analysis
| Conversation Category | Total Tests | Status Accuracy | Key Behavior Observed |
| :--- | :---: | :---: | :--- |
| `clear_evidence` | 4 | 100.0% | Correctly assigned status `scored` with grounded quotes. |
| `ambiguous_evidence` | 2 | 100.0% | Correctly identified high-risk trait while avoiding unsupported claims. |
| `contradictory_evidence` | 2 | 100.0% | Correctly weighted expressed terror over self-reported bravery. |
| `quoted_text` | 1 | 100.0% | Disregarded quoted third-party claim rejected by speaker. |
| `sarcasm` | 2 | 100.0% | Correctly evaluated `Acidity` (scored 5) and low `Civility` (scored 1). |
| `code_switching` | 2 | 100.0% | Correctly recognized bilingual phrase ('con flojera' -> `Slothfulness`). |
| `low_evidence` | 2 | 100.0% | Abstained on risktaking while assigning score for polite phrasing (`Civility`). |
| `medical_hallucination_trap` | 2 | 100.0% | **100% Abstention** on lab hormone/biomarker queries (`score = null`). |
| `biographical_hallucination_trap` | 2 | 100.0% | **100% Abstention** on external hardware/system log counts (`score = null`). |
| `unsupported_inference_case` | 2 | 100.0% | Correctly scored `Moroseness` while abstaining on clinical `Depression (DEP)`. |

---

## 3. Human Reference Set Breakdown
The reference dataset contains 10 human-reviewed conversation snippets evaluated across 20 representative facets covering:
- Clearly observable behavioral traits
- Ambiguous/contradictory expressions
- Clinical diagnostic constructs requiring psychiatric abstention
- Physical health & lab biomarkers requiring biological test abstention
- External activity system logs requiring hardware tracking abstention
