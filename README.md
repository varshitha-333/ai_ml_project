# Facet Evaluator - Production-Minded Conversational Facet Scoring & API System

A production-minded, clean machine learning application and frontend demo console for evaluating conversation dialogue transcripts against heterogeneous behavioral facet catalogs. The system features a **Google Colab-hosted Qwen2.5 GPU inference server**, a **Dockerized local FastAPI backend**, a decoupled **InferenceClient abstraction layer**, hybrid BM25 + Dense vector retrieval (Reciprocal Rank Fusion), strict 5-level anchored scoring (1-5), Pydantic structured output validation, and a **Next.js / React frontend demo application**.

---

## 🎯 1. Project Objective & Problem Framing

Evaluating dialogue transcripts against large behavioral facet catalogs poses two major challenges:
1. **Hallucination Risk & Over-Scoring**: Naive LLM prompts tend to force numerical scores even when the conversation contains zero relevant evidence, or when the facet represents an unobservable construct (e.g. blood hormone levels or system log counts).
2. **Computational Scalability**: Passing hundreds of facet definitions in a single LLM prompt creates context window overflow, prohibitive latency, and severe attention decay.

### Core Architecture Formulation:
The **Facet Evaluator Engine** solves both challenges by enforcing a strict **Two-Stage Hybrid Architecture**:
- **Stage 1 (Retrieval & Pre-Filtering)**: Uses BM25 lexical search and `all-MiniLM-L6-v2` dense vector embeddings combined via Reciprocal Rank Fusion (RRF) to select top candidate facets ($K=30$, <1ms latency). Unobservable taxonomy items (medical lab tests, external system hardware logs) bypass LLM scoring completely and route directly to `not_observable` abstention.
- **Stage 2 (Batched LLM Scoring & Grounding)**: Evaluates observable candidate facets in a single compact JSON batch using an open-weight model (`Qwen/Qwen2.5-7B-Instruct`). The model outputs 5-level ordinal scores (1–5), confidence percentages (0.0–1.0), exact conversational evidence quotes, and rationales.

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

---

## ⚙️ 3. Inference Backend Modes: Mock vs Remote

The system supports two explicit execution modes controlled via `BACKEND_MODE`:

1. **`BACKEND_MODE=mock` (Mock Mode)**:
   - *Purpose*: Fast CI unit testing, pipeline integration tests, and parser validation without requiring GPU hardware or active internet connections.
   - *Behavior*: Uses `MockInferenceBackend`. **Mock results are NOT model-quality measurements.**
2. **`BACKEND_MODE=remote` (Remote Qwen Mode)**:
   - *Purpose*: Real open-weight LLM inference evaluation.
   - *Behavior*: Connects via `InferenceClient` over HTTP to Colab vLLM / OpenAI API running `Qwen/Qwen2.5-7B-Instruct`.

---

## 🧹 4. Data Preprocessing & Facet Taxonomy

The input dataset (`data/raw/Facets Assignment.csv`) contains ~400 raw facets. The preprocessing script (`scripts/run_preprocessing.py`) cleans UTF-8/CP1252 mojibake artifacts, normalizes camelCase headers, and enriches every facet with a structured schema saved to `data/processed/enriched_facets.json`.

### Taxonomy Categories:
- **`conversational_trait`**: Observable behavioral traits (`conversation_observable = True`).
- **`medical_biomarker`**: Biological blood/hormone lab measurements (`conversation_observable = False`).
- **`external_biographical`**: External system logs, hardware counts (`conversation_observable = False`).
- **`header_anomaly`**: CSV formatting artifacts (`conversation_observable = False`).

---

## 📊 5. Five-Level Ordinal Scoring Scale & Abstention Policy

### Ordinal Scale Definitions:
- **`1 — Very Low`**: Opposite behavior or complete absence of trait.
- **`2 — Low`**: Weak, indirect, or minor presence of trait.
- **`3 — Moderate`**: Moderate, standard, or baseline expression of trait.
- **`4 — High`**: Strong, clear, and explicit evidence of trait.
- **`5 — Very High`**: Extreme, intense, or unreserved demonstration of trait.

### Strict Status Isolation:
1. **`scored`**: Facet is conversationally observable AND direct evidence exists (`score: 1–5`, `evidence: "..."`).
2. **`not_observable`**: Facet represents a medical biomarker or external log (`score: null`, `evidence: null`).
3. **`insufficient_evidence`**: Facet is observable in principle, but dialogue lacks supporting evidence (`score: null`, `evidence: null`).
4. **`inference_error`**: Technical network timeout or infrastructure failure (`score: null`, `evidence: null`). **Infrastructure failures are NEVER conflated with conversational abstentions.**

---

## 📈 6. Empirical Benchmark Results & Evaluation Methodology

The repository evaluates performance across two distinct benchmark datasets:

### A. Internal Human-Reviewed Reference Benchmark
- **Dataset**: `data/benchmark_reference_set.json` (20 reference facets across 10 hand-curated conversations).
- **Purpose**: Evaluates exact score alignment and grounded quote extraction against human reference annotations.
- **Status Classification Accuracy**: **`100.0%`**
- **Score Mean Absolute Error (MAE)**: **`0.0`**
- **Abstention Precision / Recall / F1**: **`100.0%`**

### B. External 150-Case Generalization Benchmark
- **Dataset**: `facet_evaluation_test_set_150.csv` (150 novel test cases).
- **Purpose**: Stress tests candidate retrieval coverage and novel natural language generalization.
- **Candidate Retrieval Recall@30**: **`52.67%`** (boosted from 13.33% via camelCase BM25 tokenization and $K=30$).
- **False Scoring Rate (Hallucination Rate)**: **`0.00%`** (100% direct abstention on all unobservable medical and external log traps).
- **Score MAE**: **`0.59`**

> [!IMPORTANT]
> **Methodology Distinction**: The internal reference set measures reference annotation precision, whereas the external 150-case benchmark stress tests candidate retrieval coverage and novel vocabulary generalization.

---

## 🚀 7. Setup & Execution Guide

### Run Preprocessing Pipeline
```bash
python scripts/run_preprocessing.py
```

### Run Full Pytest Suite (41 Tests)
```bash
pytest tests/ -v
```

### Run 150-Case External Validation (Mock Mode)
```bash
python scripts/run_150_validation.py --backend mock --retrieval-k 30
```

### Run 150-Case External Validation (Remote Real Qwen Mode)
```bash
python scripts/run_150_validation.py --backend remote --retrieval-k 30
```

### Run Dockerized Backend
```powershell
docker build -t facet-evaluator-backend .
docker run -p 8000:8000 -e BACKEND_MODE="remote" -e INFERENCE_URL="https://xxxx.ngrok-free.app" facet-evaluator-backend
```

### Run Next.js Frontend Console
```bash
cd frontend
npm install
npm run dev
```

---

## ⚡ 8. How the Architecture Scales to 5,000+ Facets

1. **Sub-Linear HNSW Vector Indexing**: Replace flat matrix search with Hierarchical Navigable Small World (HNSW) indexing using `Faiss` or `Qdrant` (<10ms for 5,000 vectors).
2. **Metadata Taxonomy Pre-Filtering**: 35%+ of catalog items (medical biomarkers, hardware logs) are pre-filtered out in $O(1)$ time, reducing vector search space to 3,250 facets.
3. **Constant LLM Context Cost**: Top-$K$ cutoff fixed at $K=30$, candidate batch size fixed at $10$. Prompt length remains strictly constant regardless of total catalog size.

---

## 💡 9. What I Would Improve With Another Day

1. **2-Stage Cross-Encoder Reranking**: Add a lightweight cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) on top 50 candidates to boost candidate `Recall@30` from 52.6% to >85%.
2. **Multi-Shot In-Context Exemplars**: Add 2–3 exemplar JSON outputs in LLM prompts to refine score calibration on complex sarcasm.
