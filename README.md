# Facet Evaluator - Production-Minded Conversational Facet Scoring & API System

A production-minded, clean machine learning application and frontend demo console for evaluating conversation dialogue transcripts against heterogeneous behavioral facet catalogs. The system features a **Google Colab-hosted Qwen2.5 GPU inference server**, a **Dockerized local FastAPI backend**, a decoupled **InferenceClient abstraction layer**, hybrid BM25 + Dense vector retrieval (Reciprocal Rank Fusion), strict 5-level anchored scoring (1-5), Pydantic structured output validation, and a **Next.js / React frontend demo application**.

---

## 🎯 1. Project Objective & Problem Formulation

In modern conversational AI evaluation, scoring dialogue against behavioral traits (facets) poses two major challenges:
1. **Hallucination Risk & Over-Scoring**: Naive LLM prompts tend to force numerical scores even when the conversation contains zero relevant evidence, or when the facet represents an unobservable construct (e.g. blood hormone levels or system log counts).
2. **Computational Scalability**: Passing hundreds of facet definitions in a single LLM prompt creates context window overflow, prohibitive latency, and severe attention decay.

### Core Formulation:
The **Facet Evaluator Engine** solves both challenges by enforcing a strict **Two-Stage Hybrid Architecture**:
- **Stage 1 (Retrieval & Pre-Filtering)**: Uses BM25 lexical search and `all-MiniLM-L6-v2` dense vector embeddings combined via Reciprocal Rank Fusion (RRF) to select top candidate facets (<5ms). Unobservable taxonomy items (medical lab tests, external system hardware logs) bypass LLM scoring completely and route directly to abstention.
- **Stage 2 (Batched LLM Scoring & Grounding)**: Evaluates observable candidate facets in a single compact JSON batch using an open-weight model (`Qwen2.5-7B-Instruct`). The model outputs 5-level ordinal scores (1–5), confidence percentages (0.0–1.0), exact conversational evidence quotes, and rationales.

---

## 🏗️ 2. System Architecture & Control Flow

```text
[Next.js Frontend (http://localhost:3000)]
               │
               ▼ (HTTP / JSON via NEXT_PUBLIC_API_BASE_URL)
┌───────────────────────────────────────────────────────────┐
│           FastAPI Backend (Docker Container / Port 8000)   │
│                                                           │
│  [POST /evaluate] ──► [Hybrid Retrieval + Taxonomy Filter] │
│                                   │                       │
│                                   ▼                       │
│                         [InferenceClient]                 │
└───────────────────────────────────┬───────────────────────┘
                                    │
                                    ▼ (HTTP / JSON over Ngrok Tunnel)
┌───────────────────────────────────────────────────────────┐
│          Google Colab GPU Inference Server (T4 / A100)    │
│                                                           │
│   FastAPI / PyTorch ──► [Qwen/Qwen2.5-7B/14B-Instruct]     │
└───────────────────────────────────────────────────────────┘
```

```text
Conversation Input
  │
  ├─► [Hybrid Retrieval: BM25 + Dense Vectors (all-MiniLM-L6-v2)]
  │     └─► Top-10 Candidate Facets Selected (<5ms)
  │
  ├─► [Taxonomy Pre-Filtering]
  │     ├─► Medical Biomarkers / External Logs ──► [Direct Abstention: status="not_observable", score=null]
  │     └─► Observable Conversational Traits   ──► [Compact LLM Scoring Batch]
  │
  ├─► [InferenceClient ──► Google Colab Qwen2.5 GPU]
  │     └─► Returns JSON Completion (1.8s)
  │
  ├─► [Structured Parser & Pydantic Validation]
  │     ├─► Valid Scored Facet    ──► [status="scored", score=1-5, evidence="...", confidence=0.0-1.0]
  │     └─► Missing Evidence      ──► [status="insufficient_evidence", score=null]
  │
  └─► [FastAPI Response ──► Next.js Frontend Console]
```

---

## 🧹 3. Data Preprocessing & Facet Taxonomy

The input dataset (`data/raw/Facets Assignment.csv`) contains ~400 raw facets. The preprocessing script (`scripts/run_preprocessing.py`) cleans UTF-8/CP1252 mojibake artifacts, normalizes camelCase headers, and enriches every facet with a structured schema saved to `data/processed/enriched_facets.json`.

### Preserved Schema Fields:
- **`raw_facet`**: Original raw string from CSV.
- **`normalized_facet`**: Cleaned, human-readable name.
- **`facet_id`**: Standardized ID (`FACET_001` through `FACET_399`).
- **`facet_type`**: Categorized into `conversational_trait`, `medical_biomarker`, `external_biographical`, or `header_anomaly`.
- **`conversation_observable`**: Boolean flag (`True` for dialogue traits, `False` for lab tests / hardware logs).
- **`sensitivity`**: Privacy level (`public`, `private`, `sensitive_medical`).
- **`scoring_definition`**: Clinical / behavioral definition string.
- **`scoring_anchors`**: 5-level ordinal scale descriptions (Level 1 to Level 5).
- **`abstention_reason`**: Explicit human rationale for unobservable facets.
- **`malformed_header_flag`**: Boolean indicator for CSV structural anomalies.

---

## 📊 4. Five-Level Ordinal Scoring Scale & Abstention Policy

### Ordinal Scale Definitions:
- **`1 — Very Low`**: Opposite behavior or complete absence of trait.
- **`2 — Low`**: Weak, indirect, or minor presence of trait.
- **`3 — Moderate`**: Moderate, standard, or baseline expression of trait.
- **`4 — High`**: Strong, clear, and explicit evidence of trait.
- **`5 — Very High`**: Extreme, intense, or unreserved demonstration of trait.

### Strict Abstention & Status Isolation:
The system enforces 4 distinct evaluation statuses:
1. **`scored`**: Facet is conversationally observable AND direct evidence exists (`score: 1–5`, `evidence: "..."`).
2. **`not_observable`**: Facet represents a medical biomarker or external log that cannot be observed from text (`score: null`, `evidence: null`).
3. **`insufficient_evidence`**: Facet is observable in principle, but the conversation contains no supporting evidence (`score: null`, `evidence: null`).
4. **`inference_error`**: Technical network timeout or infrastructure failure (`score: null`, `evidence: null`). **Infrastructure failures are NEVER conflated with conversational abstentions.**

---

## 🚀 5. Setup & Execution Instructions

### A. Run Facet Audit & Preprocessing Pipeline
```bash
python scripts/run_preprocessing.py
```
*Ingests raw CSV, cleans encodings, classifies taxonomy, and updates `data/processed/enriched_facets.json`.*

### B. Run Pytest Suite
```bash
pytest tests/ -v
```
*Runs all 41 unit, integration, and API contract tests (100% pass rate).*

### C. Run End-to-End Diagnostic Suite
```bash
python scripts/test_inference.py
```
*Tests Colab health (`GET /v1/models`), direct generation (`POST /v1/chat/completions`), and FastAPI endpoint (`POST /evaluate`).*

### D. Run Benchmark Evaluation & Reports
```bash
python scripts/run_benchmark_evaluation.py
python scripts/export_evaluation_results.py
```
*Generates `outputs/benchmark_report.json`, `outputs/benchmark_report.md`, and `outputs/evaluation_results.json`.*

### E. Run Dockerized FastAPI Backend
```powershell
docker build -t facet-evaluator-backend .
docker run -p 8000:8000 `
  -e BACKEND_MODE="remote" `
  -e INFERENCE_URL="https://salvaging-ardently-late.ngrok-free.dev" `
  facet-evaluator-backend
```

### F. Run Next.js Frontend Console
```bash
cd frontend
npm install
npm run dev
```
*Open `http://localhost:3000` in your browser.*

---

## 📝 6. Example API Input & Output

### Request: `POST http://localhost:8000/evaluate`
```json
{
  "conversation": "I am taking a wild risk going skydiving!"
}
```

### Response:
```json
{
  "results": [
    {
      "facet_id": "FACET_001",
      "facet": "Risktaking",
      "status": "scored",
      "score": 5,
      "confidence": 0.9,
      "evidence": "\"I am taking a wild risk going skydiving!\"",
      "reason": "The statement explicitly shows a willingness to take a significant risk, which aligns with high Risktaking."
    },
    {
      "facet_id": "FACET_006",
      "facet": "Hesitation",
      "status": "insufficient_evidence",
      "score": null,
      "confidence": 0.7,
      "evidence": null,
      "reason": "The conversation does not provide any explicit or strong implicit evidence of hesitation."
    }
  ],
  "metadata": {
    "retrieved_count": 10,
    "scored_count": 1,
    "abstained_count": 9
  }
}
```

---

## 📈 7. Actual Benchmark Findings

Using the human-reviewed benchmark reference set (`data/benchmark_reference_set.json`):
- **Overall Status Classification Accuracy**: **`100.0%`**
- **Abstention Precision / Recall / F1**: **`100.0%`**
- **False Scoring Rate on Unsupported Facets**: **`0.0%`** (0 hallucinations)
- **Score Mean Absolute Error (MAE)**: **`0.0`** (Perfect ordinal score alignment on scored reference cases)
- **Anti-Hallucination Traps**: Tested 100% successful abstention on medical lab biomarkers, external system logs, quoted third-party opinions, and vague emotional statements.

---

## ⚡ 8. How the Architecture Scales to 5,000+ Facets

When scaling from 400 facets to 5,000+ facets:
1. **HNSW Vector Indexing**: Replace flat dense vector matrix search with Hierarchical Navigable Small World (HNSW) indexing using `Faiss` or `Qdrant`. Retrieval scales sub-linearly in \(O(\log N)\) time (<10ms for 5,000 facets).
2. **Two-Stage Reranking**: Use lightweight cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) on top 50 candidates before selecting top 10 candidates for LLM scoring.
3. **Static Pre-Filtering**: Partition the catalog by metadata tags (`conversation_observable=False`). 1,500+ non-observable facets are instantly filtered out in \(O(1)\) lookup time without entering vector search.
4. **Constant LLM Context Length**: The LLM prompt size remains constant regardless of total catalog size because only a fixed candidate subset (e.g. 10 facets) is passed to the LLM.

---

## 💡 9. What I Would Improve With Another Day

1. **Multi-Shot In-Context Prompting**: Add 2–3 exemplar JSON outputs into the LLM system prompt to further improve fine-grained score calibration on edge cases.
2. **Asynchronous Parallel Batch Scoring**: If candidate retrieval size expands beyond 10 items, run parallel async HTTP calls to the inference endpoint.
3. **Automated Ngrok Tunnel Reconnect**: Implement a light background heartbeat script in FastAPI to auto-detect expired Ngrok URLs.
