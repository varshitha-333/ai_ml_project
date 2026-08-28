# AI Assistance & Human Supervision Log (`PROMPT_LOG.md`)

This document truthfulness records material AI assistance provided by Antigravity during the execution of Part 1, Part 2, Part 3, FastAPI Layer, Colab GPU Integration, Dockerization, and Next.js Frontend Console.

---

## 🧠 1. Human Engineering Ownership vs AI Assistance

### Human Engineering Ownership (Reviewer & Candidate Controlled):
- **Architecture Selection**: Designed Two-Stage Hybrid Retrieval (BM25 + Dense Vectors via RRF) to balance keyword precision and implicit semantics.
- **Taxonomy Pre-Routing**: Designed deterministic taxonomy pre-routing rules for biological counts vs general activity counters, eliminating 100% of LLM calls for unobservable facets.
- **Strict Status Isolation**: Re-architected exception handling to assign `status = "inference_error"` (Red Badge) on network failures, preventing conflation with conversational `insufficient_evidence` abstentions.
- **Controlled Retry Policy**: Configured `InferenceClient` to retry connect timeouts (10s) but **NEVER retry read operation timeouts** (60s).
- **Candidate Depth Ablation ($K=30$)**: Conducted retrieval candidate depth ablation study ($K=10$ vs $K=20$ vs $K=30$), selecting $K=30$ to boost candidate recall to 52.67%.
- **Human Reference Annotations**: Created ground-truth reference dataset ([data/benchmark_reference_set.json](file:///d:/ai_ml_project/data/benchmark_reference_set.json)) with independent human rationales.

### AI Assistance (Implementation Acceleration):
- Generated boilerplate FastAPI route templates, Pytest stub declarations, CSS styling for Next.js console, initial Pydantic schema declarations, and documentation scaffolding.

---

## 📑 2. Material AI Assistance Summary

| Date / Time | Phase / Tool | Prompt / Task Description | What AI Proposed / Generated | What Was Accepted / Changed | Verification Performed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2026-08-27** | Part 1 (Antigravity AI) | Preprocess raw CSV and classify facet taxonomy. | Generated regex cleaners, rule-based taxonomy classifier, and schema enrichment scripts. | Accepted taxonomy structure; modified regex to fix generic `count` keyword ambiguity. | `pytest tests/test_preprocessing.py -v` (9/9 passed). |
| **2026-08-27** | Part 2 (Antigravity AI) | Build hybrid BM25 + Dense vector retrieval and LLM scoring engine. | Generated BM25, Dense Vector indexer, RRF searcher, batched prompt templates, and Pydantic schemas. | Added `PurePythonTFVectorizer` zero-dependency fallback when ML libs are missing. | `pytest tests/test_retrieval.py test_scoring.py -v` (14/14 passed). |
| **2026-08-27** | Part 3 (Antigravity AI) | Build human reference benchmark and anti-hallucination trap tests. | Generated benchmark evaluation script, ablation runner, hallucination report, and checklist mappings. | Created human-reviewed reference labels (`benchmark_reference_set.json`) and verified ablation metrics. | `pytest tests/test_benchmark.py -v` (4/4 passed). |
| **2026-08-28** | Part 4 (Antigravity AI) | Build FastAPI application layer, Docker container, and Colab GPU notebook. | Generated FastAPI app (`main.py`, `config.py`, `schemas.py`, `routes.py`), `InferenceClient`, Colab notebook (`colab/qwen_inference_server.ipynb`), and `Dockerfile`. | Updated API response schemas to match frontend contract, added CORS origins, and verified Docker CPU wheel build context. | `pytest tests/test_api.py -v` (31/31 passed). |
| **2026-08-28** | Part 5 (Antigravity AI) | Build Next.js 16 frontend console and typed API client layer. | Generated `frontend/lib/api.ts` API client, `frontend/components/facet-console.tsx` UI console with evidence drawer, status badges, and 5 benchmark snippets. | Enforced strict backend-only API calls (frontend NEVER calls Colab directly). Configured `NEXT_PUBLIC_API_BASE_URL`. | `cd frontend; npm run build` (Build succeeded). |

---

## 🚨 3. What AI Got Wrong / What I Corrected

### Example 1: Mock Backend Rule Limits on Novel External Datasets
- **AI Suggestion**: The AI initially ran the 150-case external validation using `MockInferenceBackend` without distinguishing mock CI test execution from real Qwen model evaluation.
- **Failure Cause**: `MockInferenceBackend` contained rules tailored for the initial 20 reference facets. Running 150 novel test cases through `MockInferenceBackend` caused un-ruled observable facets to fall through to default abstentions (`insufficient_evidence`).
- **Human Correction**: Separated mock pipeline testing (`--backend mock`) from real Qwen GPU evaluation (`--backend remote`), added CLI diagnostic headers, and added `"Mock results are NOT model-quality measurements"` disclaimers.
- **Verification**: Verified via `python scripts/run_150_validation.py --backend mock` and `--backend remote`.

---

### Example 2: Generic Keyword Ambiguity in Medical Biomarker Classification
- **AI Suggestion**: The AI originally placed generic regex `r'\bcount\b'` inside `MEDICAL_KEYWORDS` in `src/preprocessing/taxonomy.py`.
- **Failure Cause**: This caused non-medical activity counts like `Passport-stamps count`, `Subscription count`, and `Skill-endorsements count` to be misclassified as `medical_biomarker`.
- **Human Correction**: Restricted `MEDICAL_KEYWORDS` strictly to biological counts (e.g. `basophil count`, `blood count`, `cell count`) and routed activity counts into `external_biographical`.
- **Verification**: Ran `test_external_biographical_classification` in `tests/test_preprocessing.py` (Passed).

---

### Example 3: Conflating Infrastructure Read Operation Timeouts with Evidence Abstention
- **AI Suggestion**: Initial exception handling inside `BatchedFacetScorer` caught all exceptions (including HTTP read operation timeouts from `InferenceClient`) and generated default fallback objects with `status = "insufficient_evidence"`.
- **Failure Cause**: When Ngrok/Colab timed out or returned HTTP 404, every facet returned `insufficient_evidence`, conflating infrastructure failures with conversational abstentions.
- **Human Correction**: Introduced explicit `status: "inference_error"` (Red Badge) for technical failures, separated connect/read timeouts in `InferenceClient`, and disabled read timeout retries.
- **Verification**: Verified via `python scripts/test_inference.py` and `pytest tests/ -v` (41/41 Passed).

---

### Example 4: Direct Colab Call Anti-Pattern in Frontend
- **AI Suggestion**: An initial UI draft considered querying the Ngrok `INFERENCE_URL` directly from React `fetch()`.
- **Failure Cause**: Violates architecture isolation guidelines and exposes Colab inference tokens/endpoints directly to client browsers.
- **Human Correction**: Enforced strict proxy pattern: Frontend talks ONLY to local FastAPI (`http://localhost:8000`), and FastAPI forwards requests to Colab via `InferenceClient`.
- **Verification**: Checked network requests in `frontend/lib/api.ts` — all calls point to `NEXT_PUBLIC_API_BASE_URL`.
