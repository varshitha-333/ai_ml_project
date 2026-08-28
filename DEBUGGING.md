# Empirical Debugging & Problem Resolution Log (`DEBUGGING.md`)

This log records real technical issues, failed assumptions, root causes, and verified fixes I encountered during the development and optimization of the **Facet Evaluator** engine.

---

## Issue 1: Infrastructure Read Operation Timeout & Conflation of Failure with Abstention
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
  5. I created `scripts/test_inference.py` diagnostic suite.
- **Verification**: Verified via `python scripts/test_inference.py` and `pytest tests/ -v` (41/41 Passed).

---

## Issue 2: Duplicate Ngrok Path Appending Resulting in HTTP 404
- **Symptom**: Calls from `InferenceClient` to Colab returned `HTTP Error 404: Not Found`.
- **Diagnosis**: When I examined the terminal output, I saw `[INFERENCE_CLIENT] Sending request to URL: https://salvaging-ardently-late.ngrok-free.dev/v1/chat/completions/v1/chat/completions`.
- **Root Cause**: `InferenceClient.__init__` received `INFERENCE_URL` with `/v1/chat/completions` already attached. When `generate()` appended `/v1/chat/completions`, it created a duplicate path.
- **Fix**: I updated `InferenceClient.__init__` to automatically strip any trailing `/v1/chat/completions` or `/v1/models` or `/v1` suffixes from `INFERENCE_URL`.
- **Verification**: Re-ran evaluation requests — `InferenceClient` connected cleanly and returned `HTTP 200 OK`.

---

## Issue 3: Windows Console `UnicodeEncodeError` in Diagnostic Script
- **Symptom**: Running `python scripts/test_inference.py` on Windows PowerShell failed with `UnicodeEncodeError: 'charmap' codec can't encode character '\u274c'`.
- **Diagnosis**: The diagnostic script printed Unicode emojis (`❌`, `✅`) to `sys.stdout`. On Windows terminals using legacy CP1252 character encodings, printing non-ASCII emojis raises an encoding exception.
- **Root Cause**: Windows PowerShell console standard output encoding defaults to CP1252 rather than UTF-8.
- **Fix**: I replaced emoji characters in `scripts/test_inference.py` with standard ASCII strings (`[PASS]`, `[FAIL]`, `[WARN]`).
- **Verification**: Ran `python scripts/test_inference.py` on PowerShell — executed cleanly without encoding errors.

---

## Issue 4: Docker Container Startup Failure due to Missing Keyword Argument
- **Symptom**: `docker run` failed on container startup with `TypeError: InferenceClient.__init__() got an unexpected keyword argument 'timeout'`.
- **Diagnosis**: I inspected `src/api/services.py`, which instantiated `InferenceClient(..., timeout=...)`. When I previously refactored `InferenceClient` to use `read_timeout` and `connect_timeout`, I omitted `timeout` from `__init__`.
- **Root Cause**: Mismatch between `PipelineService` signature and `InferenceClient.__init__`.
- **Fix**: I added backward-compatible `timeout: Optional[int] = None` support to `InferenceClient.__init__` in `src/scoring/inference_client.py` and rebuilt the Docker image (`docker build -t facet-evaluator-backend .`).
- **Verification**: Re-ran `docker run` — startup succeeded cleanly and `/health` returned `200 OK`.

---

## Issue 5: Generic Regex Keyword Overlap in Taxonomy Classification
- **Symptom**: Non-medical biographical activity counters (e.g. `Passport-stamps count`, `Subscription count`) were misclassified as `medical_biomarker`.
- **Diagnosis**: I inspected `src/preprocessing/taxonomy.py` and found a generic keyword rule matching `r'\bcount\b'` inside `MEDICAL_KEYWORDS`.
- **Root Cause**: Overly broad regex matching `count` without requiring biological or physiological context.
- **Fix**: I restricted `MEDICAL_KEYWORDS` strictly to biological counts (e.g. `basophil count`, `blood count`, `cell count`) and routed general activity counts into `external_biographical`.
- **Verification**: Executed `test_external_biographical_classification` in `tests/test_preprocessing.py` (Passed).

---

## Issue 6: Encoding Mojibake and Multi-Byte UTF-8 Corruption
- **Symptom**: `UnicodeDecodeError` and corrupted text strings (e.g. `â€™`, `â€"`) when ingesting `data/raw/Facets Assignment.csv`.
- **Diagnosis**: The raw CSV contained mixed Windows-1252 / CP1252 smart quotes and UTF-8 mojibake encoding artifacts.
- **Root Cause**: Inconsistent text encoding exports from spreadsheet software without explicit UTF-8 BOM declarations.
- **Fix**: Built `repair_encoding_anomalies()` in `src/preprocessing/cleaner.py` using `ftfy` / regex replacements for Windows-1252 characters (`’` $\rightarrow$ `'`, `–` $\rightarrow$ `-`).
- **Verification**: Verified using `test_encoding_repair` in `tests/test_preprocessing.py` (Passed).
