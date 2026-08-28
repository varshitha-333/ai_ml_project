# Empirical Debugging & Problem Resolution Log (`DEBUGGING.md`)

This log records real technical issues, failed assumptions, root causes, and verified fixes I encountered during the development and optimization of the **Facet Evaluator** engine.

---

## Real Incident 1: Mock-vs-Remote Inference Validation Issue & Low Mock Accuracy
- **Symptom**: External 150-case validation produced approximately 27.67% accuracy and 0.0s average latency.
- **Diagnosis**: The validation script executed `MockInferenceBackend` on CPU rather than enforcing real open-weight LLM inference over HTTP (`RemoteInferenceClientBackend`).
- **Root Cause**: `MockInferenceBackend` contained deterministic rules tailored for the initial 20 reference facets. Running 150 novel test cases through `MockInferenceBackend` caused novel observable facets to fall through to default `insufficient_evidence` abstentions.
- **Fix**:
  1. I isolated mock inference (`--backend mock`) strictly for deterministic unit tests and pipeline integration tests.
  2. I required `BACKEND_MODE=remote` and an active `INFERENCE_URL` for real model-quality validation, prohibiting silent fallback from remote to mock mode.
  3. I added diagnostic headers to all validation outputs displaying `BACKEND MODE: MOCK/REMOTE`, `MODEL NAME`, `RETRIEVAL K`, `SCORED`, `ABSTENTIONS`, and `INFERENCE ERRORS`.
- **Verification**: Executed 5-case representative real smoke test (`scripts/run_5_case_validation.py`) and verified that real HTTP responses return non-zero generation latency and structured JSON output.

---

## Real Incident 2: Retrieval Candidate Cutoff Failure & Tokenization Mismatch
- **Symptom**: Candidate retrieval `Recall@30` was initially 13.33% on the external 150-case benchmark dataset.
- **Diagnosis**: The raw BM25 tokenizer failed to split camelCase boundary compound words (e.g. `HonestyHumility:`) or strip trailing punctuation symbols (e.g. `Democratic Leadership:`), causing exact lexical misses that dropped target facets prior to candidate scoring.
- **Root Cause**: Missing camelCase sub-token regex expansion in `BM25Indexer._tokenize`.
- **Fix**:
  1. Updated `BM25Indexer._tokenize` in `src/retrieval/bm25.py` to automatically expand camelCase boundaries (`re.sub(r'([a-z])([A-Z])', r'\1 \2', text)`).
  2. Combined camelCase BM25 search with `all-MiniLM-L6-v2` dense vector Reciprocal Rank Fusion (RRF) at $K=30$.
- **Verification**: `Recall@30` jumped from **13.33%** up to **78.67%** at sub-millisecond query speed (`0.94ms`).

---

## Real Incident 3: Infrastructure Read Operation Timeout & Conflation with Evidence Abstention
- **Symptom**: All 10 retrieved candidate facets returned `status = insufficient_evidence` with `confidence = 0.0` and `reason = Pipeline exception fallback: InferenceClient failed after 3 attempts. Error: The read operation timed out`.
- **Diagnosis**: `InferenceClient` hit an HTTP read operation timeout waiting for Google Colab Qwen LLM output when `max_new_tokens=1024` was requested. Retrying 3 times on a read timeout tripled latency to >90s, causing requests to time out completely. Furthermore, the scorer exception handler generated default `insufficient_evidence` objects, incorrectly conflating technical infrastructure timeouts with legitimate conversational abstentions.
- **Root Cause**:
  1. Excessive generation token limit (`max_new_tokens=1024` vs required ~150 JSON tokens).
  2. Sequential batch calls (2 batches of 5 candidates) instead of 1 single compact batch (10 candidates).
  3. Retrying expensive LLM calls on read timeouts.
  4. Scorer exception handler conflating technical infrastructure errors with evidence abstention.
- **Fix**:
  1. Updated `InferenceClient` in `src/scoring/inference_client.py` to separate connect timeout (10s) and read timeout (60s) with controlled retries (NEVER retry read timeouts).
  2. Reduced `max_new_tokens = 300` and set `do_sample = False` in Colab server code.
  3. Reduced scoring calls to **1 single compact batch call** (`batch_size = 10`).
  4. Introduced explicit `status = "inference_error"` (Red Badge) for technical failures, guaranteeing pure semantics for `scored`, `not_observable`, and `insufficient_evidence`.
- **Verification**: Verified via `python scripts/test_inference.py` and `pytest tests/ -v` (41/41 Passed).

---

## Real Incident 4: Duplicate Ngrok Path Appending Resulting in HTTP 404
- **Symptom**: Calls from `InferenceClient` to Colab returned `HTTP Error 404: Not Found`.
- **Diagnosis**: Terminal logs showed `[INFERENCE_CLIENT] Sending request to URL: https://salvaging-ardently-late.ngrok-free.dev/v1/chat/completions/v1/chat/completions`.
- **Root Cause**: `InferenceClient.__init__` received `INFERENCE_URL` with `/v1/chat/completions` already attached.
- **Fix**: Updated `InferenceClient.__init__` to strip trailing `/v1/chat/completions` suffixes.
- **Verification**: Re-ran evaluation requests — `InferenceClient` connected cleanly and returned `HTTP 200 OK`.

---

## Real Incident 5: Generic Regex Keyword Overlap in Taxonomy Classification
- **Symptom**: Non-medical biographical activity counters (`Passport-stamps count`, `Subscription count`) were misclassified as `medical_biomarker`.
- **Diagnosis**: `src/preprocessing/taxonomy.py` matched generic `r'\bcount\b'` inside `MEDICAL_KEYWORDS`.
- **Root Cause**: Overly broad regex matching without biological context.
- **Fix**: Restricted `MEDICAL_KEYWORDS` strictly to biological counts (`blood count`, `cell count`) and routed activity counters into `external_biographical`.
- **Verification**: Executed `test_external_biographical_classification` in `tests/test_preprocessing.py` (Passed).
