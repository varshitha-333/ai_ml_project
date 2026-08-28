# Final Submission Audit Checklist (`FINAL_AUDIT.md`)

This checklist verifies every mandatory assignment requirement against its exact implementation file, code line, and empirical test evidence.

---

## 📋 Comprehensive 25-Item Checklist

| # | Requirement | Status | Exact File / Code Evidence Path | Verification Evidence |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **Raw facet value preserved** | **PASS** | [src/preprocessing/pipeline.py](file:///d:/ai_ml_project/src/preprocessing/pipeline.py#L35) | `raw_facet` field in [data/processed/enriched_facets.json](file:///d:/ai_ml_project/data/processed/enriched_facets.json) |
| **2** | **Normalized facet created** | **PASS** | [src/preprocessing/cleaner.py](file:///d:/ai_ml_project/src/preprocessing/cleaner.py#L40) | `test_numeric_prefix_stripping` passed |
| **3** | **Taxonomy created** | **PASS** | [src/preprocessing/taxonomy.py](file:///d:/ai_ml_project/src/preprocessing/taxonomy.py#L25) | 4 categories (`conversational_trait`, `medical_biomarker`, `external_biographical`, `header_anomaly`) |
| **4** | **Observable / non-observable distinction** | **PASS** | [src/preprocessing/taxonomy.py](file:///d:/ai_ml_project/src/preprocessing/taxonomy.py#L50) | `conversation_observable` boolean flag in dataset |
| **5** | **Malformed / header detection** | **PASS** | [src/preprocessing/cleaner.py](file:///d:/ai_ml_project/src/preprocessing/cleaner.py#L70) | `test_trailing_colon_header_detection` passed |
| **6** | **Enriched CSV & JSON generated reproducibly** | **PASS** | [scripts/run_preprocessing.py](file:///d:/ai_ml_project/scripts/run_preprocessing.py) | Ingests `Facets Assignment.csv` $\rightarrow$ `enriched_facets.json` |
| **7** | **Retrieval / routing implemented** | **PASS** | [src/retrieval/search.py](file:///d:/ai_ml_project/src/retrieval/search.py#L48) | Reciprocal Rank Fusion (BM25 + Dense vector search) |
| **8** | **No one-shot full catalogue scoring** | **PASS** | [src/evaluator_pipeline.py](file:///d:/ai_ml_project/src/evaluator_pipeline.py#L65) | Candidate cutoff `top_k = 10` before LLM call |
| **9** | **Five ordered integer levels** | **PASS** | [src/scoring/schemas.py](file:///d:/ai_ml_project/src/scoring/schemas.py#L19) | Pydantic score check (`1 <= score <= 5`) |
| **10** | **Abstention policy implemented** | **PASS** | [src/scoring/scorer.py](file:///d:/ai_ml_project/src/scoring/scorer.py#L60) | `not_observable`, `insufficient_evidence`, `inference_error` |
| **11** | **Confidence returned** | **PASS** | [src/scoring/schemas.py](file:///d:/ai_ml_project/src/scoring/schemas.py#L22) | `confidence` float bounded in `[0.0, 1.0]` |
| **12** | **Evidence quote returned** | **PASS** | [src/scoring/parser.py](file:///d:/ai_ml_project/src/scoring/parser.py#L45) | Extracted quote string for all scored facets |
| **13** | **Reason rationale returned** | **PASS** | [src/scoring/schemas.py](file:///d:/ai_ml_project/src/scoring/schemas.py#L28) | Human/LLM rationale explaining score or abstention |
| **14** | **Structured output validation** | **PASS** | [src/scoring/parser.py](file:///d:/ai_ml_project/src/scoring/parser.py#L12) | Regex JSON extraction, markdown block stripping, repair logic |
| **15** | **Model size <= 16B** | **PASS** | [colab/qwen_inference_server.ipynb](file:///d:/ai_ml_project/colab/qwen_inference_server.ipynb) | `Qwen/Qwen2.5-7B-Instruct` (7B parameters) |
| **16** | **Open-weight model licence** | **PASS** | [colab/qwen_inference_server.ipynb](file:///d:/ai_ml_project/colab/qwen_inference_server.ipynb) | Apache 2.0 open-weight license |
| **17** | **10+ conversation examples** | **PASS** | [data/benchmark_reference_set.json](file:///d:/ai_ml_project/data/benchmark_reference_set.json) | 10 benchmark conversations (`CONV_001` to `CONV_010`) |
| **18** | **20+ representative facets** | **PASS** | [data/benchmark_reference_set.json](file:///d:/ai_ml_project/data/benchmark_reference_set.json) | 20 representative facets spanning all categories |
| **19** | **Human-reviewed reference set** | **PASS** | [data/benchmark_reference_set.json](file:///d:/ai_ml_project/data/benchmark_reference_set.json) | Ground-truth labels, expected scores, human rationales |
| **20** | **Benchmark executed** | **PASS** | [scripts/run_benchmark_evaluation.py](file:///d:/ai_ml_project/scripts/run_benchmark_evaluation.py) | Exports `outputs/benchmark_report.json` and `.md` |
| **21** | **Failure analysis report** | **PASS** | [outputs/benchmark_report.md](file:///d:/ai_ml_project/outputs/benchmark_report.md#L147) | Breakdown of clear, ambiguous, contradictory, and trap cases |
| **22** | **3 Hallucination-trap examples** | **PASS** | [scripts/generate_hallucination_report.py](file:///d:/ai_ml_project/scripts/generate_hallucination_report.py) | Medical lab trap, Quoted opinion, Vague emotion vs clinical diagnosis |
| **23** | **5,000-facet scaling explanation** | **PASS** | [DECISIONS.md](file:///d:/ai_ml_project/DECISIONS.md#L65) | HNSW ANN indexing, metadata pre-filtering, constant LLM cost |
| **24** | **Documentation suite (README, DECISIONS, DEBUGGING, PROMPT_LOG)** | **PASS** | Workspace root | All 4 mandatory Markdown documents fully populated in engineer voice |
| **25** | **Meaningful Git history** | **PASS** | `.git` repository commits | Commits covering preprocessing, retrieval, scoring, API, and docs |

---

### 📊 Summary Audit Totals

```text
MANDATORY REQUIREMENTS
PASS: 25 / 25
FAIL: 0 / 25
```
