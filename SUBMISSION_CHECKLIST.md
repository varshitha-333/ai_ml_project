# Final Candidate Assignment Submission Checklist

This checklist maps every mandatory requirement from the assignment specification to its implementation files, test evidence, and documentation locations.

---

## 1. Mandatory Preprocessing & Taxonomy (Part 1)

| Requirement | Implementation File | Verification / Test Evidence | Documentation Location |
| :--- | :--- | :--- | :--- |
| **Reproducible Preprocessing** | `src/preprocessing/pipeline.py` | `python scripts/run_preprocessing.py` | [README.md](file:///d:/ai_ml_project/README.md) |
| **Raw vs Normalized Preservation** | `src/preprocessing/cleaner.py` | `test_raw_vs_normalized_preservation` | [DECISIONS.md](file:///d:/ai_ml_project/DECISIONS.md) |
| **8-Category Conservative Taxonomy** | `src/preprocessing/taxonomy.py` | `test_medical_biomarker_classification` | [README.md](file:///d:/ai_ml_project/README.md) |
| **Anomaly Detection (Headers/Prefixes)** | `src/preprocessing/cleaner.py` | `test_trailing_colon_header_detection` | [outputs/audit_report.md](file:///d:/ai_ml_project/outputs/audit_report.md) |
| **Enriched Output CSV & JSON** | `data/processed/enriched_facets.csv` | Output file existence (`399` rows) | [README.md](file:///d:/ai_ml_project/README.md) |
| **Dataset Audit Report** | `scripts/generate_audit_report.py` | `outputs/audit_report.json` & `.md` | [outputs/audit_report.md](file:///d:/ai_ml_project/outputs/audit_report.md) |

---

## 2. Mandatory Candidate Retrieval & Scoring Architecture (Part 2)

| Requirement | Implementation File | Verification / Test Evidence | Documentation Location |
| :--- | :--- | :--- | :--- |
| **No 400-Facet One-Shot LLM Call** | `src/retrieval/search.py` | Top-$K$ cutoff ($K=30$, batch_size=5) | [DECISIONS.md](file:///d:/ai_ml_project/DECISIONS.md) |
| **Precomputed Semantic Representations** | `src/retrieval/indexer.py` | Encodes `normalized_facet` + `facet_type` + `scoring_definition` | [README.md](file:///d:/ai_ml_project/README.md) |
| **Hybrid BM25 + Dense RRF Search** | `src/retrieval/search.py` | `test_hybrid_retrieval` | [outputs/retrieval_ablation_report.md](file:///d:/ai_ml_project/outputs/retrieval_ablation_report.md) |
| **Observable Pre-Filtering & Routing** | `src/evaluator_pipeline.py` | `test_unobservable_facet_prefiltering` | [DECISIONS.md](file:///d:/ai_ml_project/DECISIONS.md) |
| **Open-Weight LLM ($\le$16B compliant)** | `src/scoring/inference_backend.py` | Qwen2.5-14B-Instruct-AWQ / Qwen2.5-7B | [README.md](file:///d:/ai_ml_project/README.md) |
| **5-Level Anchored Scoring (1..5)** | `src/scoring/prompt_templates.py` | `test_schema_valid_scored_result` | [README.md](file:///d:/ai_ml_project/README.md) |
| **Strict Abstention (`score = null`)** | `src/scoring/scorer.py` | `test_schema_abstained_score_forced_null` | [outputs/hallucination_report.md](file:///d:/ai_ml_project/outputs/hallucination_report.md) |
| **Grounded Quote Evidence Extraction** | `src/scoring/parser.py` | Grounded quotes in output JSON | [outputs/evaluation_results.json](file:///d:/ai_ml_project/outputs/evaluation_results.json) |
| **Resilient Pydantic JSON Parser** | `src/scoring/parser.py` | `test_parser_malformed_json_recovery` | [DECISIONS.md](file:///d:/ai_ml_project/DECISIONS.md) |

---

## 3. Mandatory Evaluation, Benchmark & Documentation (Part 3)

| Requirement | Implementation File | Verification / Test Evidence | Documentation Location |
| :--- | :--- | :--- | :--- |
| **Human Reference Benchmark Set** | `data/benchmark_reference_set.json` | 10 conversations / 20 representative facets | [outputs/benchmark_report.md](file:///d:/ai_ml_project/outputs/benchmark_report.md) |
| **Hallucination Stress Tests (3 Cases)** | `scripts/generate_hallucination_report.py` | Medical, External Log & Clinical traps | [outputs/hallucination_report.md](file:///d:/ai_ml_project/outputs/hallucination_report.md) |
| **Benchmark Evaluation Report** | `scripts/run_benchmark_evaluation.py` | Accuracy, MAE, Abstention Precision/Recall | [outputs/benchmark_report.md](file:///d:/ai_ml_project/outputs/benchmark_report.md) |
| **Retrieval Ablation Study** | `scripts/run_retrieval_ablation.py` | BM25 vs Dense vs Hybrid Recall@K | [outputs/retrieval_ablation_report.md](file:///d:/ai_ml_project/outputs/retrieval_ablation_report.md) |
| **$\ge$5,000 Facet Scaling Discussion** | `DECISIONS.md` | Memory budget, ANN indexing, latency breakdown | [DECISIONS.md](file:///d:/ai_ml_project/DECISIONS.md) |
| **3+ Non-Trivial ADRs** | `DECISIONS.md` | Problem, Alternatives, Chosen, Trade-off | [DECISIONS.md](file:///d:/ai_ml_project/DECISIONS.md) |
| **2+ Real Bug Tracebacks** | `DEBUGGING.md` | Symptom, Diagnosis, Root cause, Fix | [DEBUGGING.md](file:///d:/ai_ml_project/DEBUGGING.md) |
| **AI Audit & "What AI Got Wrong"** | `PROMPT_LOG.md` | Material prompts & 3+ human corrections | [PROMPT_LOG.md](file:///d:/ai_ml_project/PROMPT_LOG.md) |

---

## 4. FastAPI Layer, Colab GPU Integration & Dockerization (Part 4)

| Requirement | Implementation File | Verification / Test Evidence | Documentation Location |
| :--- | :--- | :--- | :--- |
| **Colab Qwen2.5-14B-AWQ Notebook** | `colab/qwen_inference_server.ipynb` | Verified cell execution & Ngrok tunnel | [README.md](file:///d:/ai_ml_project/README.md) |
| **InferenceClient Abstraction** | `src/scoring/inference_client.py` | `test_inference_client_generate_success` | [DECISIONS.md](file:///d:/ai_ml_project/DECISIONS.md) |
| **FastAPI Backend Application** | `src/api/main.py` & `routes.py` | `test_api_evaluate_valid_request` | [README.md](file:///d:/ai_ml_project/README.md) |
| **Health Check Endpoint (`GET /health`)** | `src/api/routes.py` | `test_api_health_endpoint` | [README.md](file:///d:/ai_ml_project/README.md) |
| **Primary Evaluation (`POST /evaluate`)** | `src/api/routes.py` | `test_api_evaluate_valid_request` | [README.md](file:///d:/ai_ml_project/README.md) |
| **Error Handling & Secret Protection** | `src/api/main.py` | `test_api_evaluate_empty_conversation` | [README.md](file:///d:/ai_ml_project/README.md) |
| **Lightweight Docker Container** | `Dockerfile` & `.dockerignore` | Container builds without model weights | [README.md](file:///d:/ai_ml_project/README.md) |
| **Frontend API Contract & cURL Examples** | `src/api/schemas.py` | OpenAPI schema docs at `/docs` | [README.md](file:///d:/ai_ml_project/README.md) |

---

## 🛠️ Final Verification Commands Summary

### Local VS Code Terminal:
```bash
# Run full pytest suite (41 unit, integration, and API tests)
pytest tests/ -v
```

### Docker Commands:
```bash
docker build -t facet-evaluator-backend .
docker run -p 8000:8000 --env-file .env facet-evaluator-backend
```

### Google Colab GPU Commands:
1. Open [colab/qwen_inference_server.ipynb](file:///d:/ai_ml_project/colab/qwen_inference_server.ipynb).
2. Set runtime to **GPU (T4 or A100)**.
3. Run cells to initialize `Qwen/Qwen2.5-14B-Instruct-AWQ` on vLLM and copy `INFERENCE_URL`.
