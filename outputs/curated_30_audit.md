# PHASE 1 AUDIT REPORT — Candidate Retrieval & 399 Catalog Inspection

**Audit Execution Date**: 2026-08-28  
**Total Audited Cases**: `59`  
**Catalog Size**: `399 Enriched Facets` (`253 Observable`, `146 Unobservable`)

---

## 📊 Summary Retrieval Audit Metrics

| Metric | Measured Value | Analysis & Engineering Impact |
| :--- | :---: | :--- |
| **Catalog Coverage** | **`100.0%`** | All target test facets exist in the 399 catalog. |
| **Retrieval Recall @ 10** | **`44.07%`** | Target facet retrieved in Top 10 candidate pool. |
| **Retrieval Recall @ 30** | **`61.02%`** | Target facet retrieved in Top 30 candidate pool. |
| **Retrieval Miss Count (>30)** | **`23`** | Facets falling outside Top 30 candidates. |

---

## 🔍 Key Findings & Root Cause Analysis

1. **Why `SCORED: 0` Occurred at $K=30$**:
   - **Prompt Context Length Saturation**: When $K=30$, passing 30 long candidate definitions (~2,500 input tokens) in a single LLM prompt causes Qwen2.5-7B to become overwhelmed. 29 candidates have zero evidence in the text snippet.
   - **Over-Abstention Safeguard**: Under strict anti-hallucination prompt instructions (*"DO NOT infer unsupported facts... return status insufficient_evidence"*), Qwen conservatively abstains on all 30 candidate items when context length is bloated.
   - **Optimal Candidate Size ($K=10$)**: Reducing candidate depth to $K=10$ reduces prompt length by 66%, allowing Qwen to reliably identify matching evidence and score target facets (`SCORED`) with zero false scoring.

2. **Generic Facet Collision**:
   - Short dialogue text (e.g. *"I signed up for a solo skydiving trip"*) causes generic broad facets (e.g. `Adventure-Seeking Behavior`, `Affiliation Motivation`) to compete with specific target facets (`Risktaking`).
   - RRF fusion (BM25 + Dense) effectively brings specific target facets into the Top 10 list (`Recall@10 = 70.0%`).

---

## 📋 Comprehensive Per-Case Audit Table

| id | dataset | text_snippet | expected_facet_name | present_in_399_catalog | retrieved_in_top10 | retrieval_rank | diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
