# Empirical Debugging & Problem Resolution Log (`DEBUGGING.md`)

This log records real technical issues, failed assumptions, root causes, and verified fixes I encountered during the development and optimization of the **Facet Evaluator** engine.

---

## Issue 1: External 150-Case Discrepancy & Mock Backend Rule Limits
- **Symptom**: Internal benchmark reported 100.0% status accuracy, but external 150-case validation reported 24.67% status accuracy when executed in default testing mode.
- **Investigation**: I traced the execution log and discovered that the script executed `MockInferenceBackend` rather than `RemoteInferenceClientBackend` (Qwen2.5-7B GPU).
- **Root Cause**: `MockInferenceBackend` contained deterministic rules tailored specifically for the initial 20 reference facets. Running 150 novel test cases through `MockInferenceBackend` caused un-ruled observable facets to fall through to default abstentions (`insufficient_evidence`).
- **Fix**:
  1. I separated deterministic pipeline testing (`--backend mock`) from real Qwen model evaluation (`--backend remote`).
  2. I added mandatory CLI diagnostic headers displaying `BACKEND MODE: MOCK/REMOTE`, `MODEL NAME`, `RETRIEVAL K`, `TOTAL CASES`, `SCORED`, `ABSTENTIONS`, and `INFERENCE ERRORS`.
  3. I explicitly documented in logs and reports: `"Mock results are NOT model-quality measurements."`
- **Additional Finding**: `Recall@30` was 52.67%, exposing a genuine candidate retrieval coverage bottleneck for complex multi-word facet queries.
- **Verification**: Verified via `python scripts/run_150_validation.py --backend mock --retrieval-k 30` and `--backend remote`.

---

## Issue 2: Infrastructure Read Operation Timeout & Conflation of Failure with Abstention
- **Symptom**: All 10 retrieved candidate facets returned `status = insufficient_evidence` with `confidence = 0.0` and `reason = Pipeline exception fallback: InferenceClient failed after 3 attempts. Error: The read operation timed out`.
- **Diagnosis**: I traced the failure to `InferenceClient`, which hit an HTTP read operation timeout waiting for Google Colab Qwen LLM output when `max_new_tokens=1024` was requested. Retrying 3 times on a read timeout tripled latency to >90s, causing requests to time out completely. Furthermore, the scorer exception handler generated default `insufficient_evidence` objects, incorrectly conflating technical infrastructure timeouts with legitimate conversational abstentions.
- **Root Cause**:
  1. Excessive generation token limit (`max_new_tokens=1024` vs required ~150 JSON tokens).
  2. Sequential batch calls (2 batches of 5 candidates) instead of 1 single compact batch (10 candidates).
  3. Retrying expensive LLM calls on read timeouts.
  4. Scorer exception handler conflating technical infrastructure errors with evidence abstention.
- **Fix**:
  1. I updated `InferenceClient` in `src/scoring/inference_client.py` to separate connect timeout (10s) and read timeout (60s) with controlled retries (NEVER retry read timeouts).
  2. I reduced `max_new_tokens = 300` and set `do_sample = False` in Colab server code.
  3. I reduced scoring calls to **1 single compact batch call** (`batch_size = 10`).
  4. I introduced explicit `status = "inference_error"` (Red Badge) for technical failures, guaranteeing pure semantics for `scored`, `not_observable`, and `insufficient_evidence`.
- **Verification**: Verified via `python scripts/test_inference.py` and `pytest tests/ -v` (41/41 Passed).

---

## Issue 3: Unobservable Facet Retrieval Cutoff Fallthrough
- **Symptom**: Unobservable medical lab facets (e.g. `Parathyroid-hormone level`) were falling through to `status = "insufficient_evidence"` instead of `status = "not_observable"`.
- **Diagnosis**: When candidate retrieval filtered out unobservable facets from `self.documents`, the pipeline returned `target_res = None` for unobservable test queries, defaulting `predicted_status` to `insufficient_evidence`.
- **Root Cause**: Retrieval candidate filtering dropped unobservable items before status evaluation could take place.
- **Fix**: I introduced deterministic taxonomy pre-routing in `FacetEvaluatorPipeline` and `scripts/run_150_validation.py`. Any facet with `conversation_observable == False` routes **directly to `status = "not_observable"`** with `score = null`, `confidence = 0.99`, and `reason = "Facet requires external/medical evidence that is not present in the conversation."`.
- **Verification**: Hallucination false scoring rate dropped to **0.0%** across all unsupported facet traps.

---

## Issue 4: Duplicate Ngrok Path Appending Resulting in HTTP 404
- **Symptom**: Calls from `InferenceClient` to Colab returned `HTTP Error 404: Not Found`.
- **Diagnosis**: Terminal logs showed `[INFERENCE_CLIENT] Sending request to URL: https://salvaging-ardently-late.ngrok-free.dev/v1/chat/completions/v1/chat/completions`.
- **Root Cause**: `InferenceClient.__init__` received `INFERENCE_URL` with `/v1/chat/completions` already attached.
- **Fix**: Updated `InferenceClient.__init__` to strip trailing `/v1/chat/completions` suffixes.
- **Verification**: Re-ran evaluation requests — `InferenceClient` connected cleanly and returned `HTTP 200 OK`.

---

## Issue 5: Generic Regex Keyword Overlap in Taxonomy Classification
- **Symptom**: Non-medical biographical activity counters (`Passport-stamps count`, `Subscription count`) were misclassified as `medical_biomarker`.
- **Diagnosis**: `src/preprocessing/taxonomy.py` matched generic `r'\bcount\b'` inside `MEDICAL_KEYWORDS`.
- **Root Cause**: Overly broad regex matching without biological context.
- **Fix**: Restricted `MEDICAL_KEYWORDS` strictly to biological counts (`blood count`, `cell count`) and routed activity counters into `external_biographical`.
- **Verification**: Executed `test_external_biographical_classification` in `tests/test_preprocessing.py` (Passed).

---

## Issue 6: Windows Console `UnicodeEncodeError` in Diagnostic Script
- **Symptom**: Running `python scripts/test_inference.py` on Windows PowerShell failed with `UnicodeEncodeError: 'charmap' codec can't encode character '\u274c'`.
- **Diagnosis**: Printing Unicode emojis (`❌`, `✅`) to standard output on CP1252 Windows consoles.
- **Root Cause**: Standard output encoding default.
- **Fix**: Replaced emoji characters with standard ASCII strings (`[PASS]`, `[FAIL]`, `[WARN]`).
- **Verification**: Diagnostic script executed cleanly on Windows PowerShell.
