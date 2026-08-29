"""
Diagnostic Runner for TEST_001 through TEST_004.
Executes each test case independently, records monotonic latency with time.perf_counter(),
resolves target facets via canonical normalized name lookup, and saves un-truncated raw Qwen outputs.
"""

import sys
from pathlib import Path
import json
import time
import urllib.request
import re
import math
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25 import BM25Indexer
from src.retrieval.indexer import DenseVectorIndexer

COLAB_URL = "https://salvaging-ardently-late.ngrok-free.dev/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def normalize_facet_name(name: str) -> str:
    """Normalizes facet name by stripping trailing colons and extra whitespace."""
    if not name:
        return ""
    cleaned = name.strip()
    if cleaned.endswith(":"):
        cleaned = cleaned[:-1].strip()
    return cleaned


def extract_json_results(content: str):
    """
    Universal JSON parser supporting Formats A, B, C, D, E, F:
    - Format A: Bare list [...]
    - Format B: Object wrapper {"results": [...]}
    - Format C: Markdown codeblock ```json ... ```
    - Format D: Preamble/postamble text around JSON
    - Format E: Whitespace/newline variations
    - Format F: Single dict object {"facet_id": "...", "score": 5.0}
    """
    if not content or not content.strip():
        return [], "empty_output"

    clean_content = re.sub(r'```(?:json)?', '', content).strip()

    # 1. Direct JSON array match (Format A, C, D)
    array_match = re.search(r'\[\s*\{.*\}\s*\]', clean_content, re.DOTALL)
    if array_match:
        try:
            parsed = json.loads(array_match.group(0))
            if isinstance(parsed, list):
                return parsed, "valid_array"
        except Exception:
            pass

    # 2. General JSON array fallback
    array_match = re.search(r'\[.*\]', clean_content, re.DOTALL)
    if array_match:
        try:
            parsed = json.loads(array_match.group(0))
            if isinstance(parsed, list):
                return parsed, "valid_array"
        except Exception:
            pass

    # 3. Object wrapper (Format B, F)
    obj_match = re.search(r'\{.*\}', clean_content, re.DOTALL)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group(0))
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if isinstance(v, list):
                        return v, "wrapped_object"
                if "facet_id" in parsed or "score" in parsed or "status" in parsed:
                    return [parsed], "single_object"
        except Exception:
            pass

    # 4. Truncated Array Repair (Extract all complete {...} objects before cut-off)
    obj_matches = re.findall(r'\{\s*"facet_id".*?\}', clean_content, re.DOTALL)
    if obj_matches:
        recovered = []
        for obj_str in obj_matches:
            try:
                parsed_obj = json.loads(obj_str)
                if isinstance(parsed_obj, dict):
                    recovered.append(parsed_obj)
            except Exception:
                pass
        if recovered:
            return recovered, "truncated_recovery"

    return [], "parse_error"


def validate_result_schema(parsed_items: list, valid_facet_ids: set) -> tuple:
    """
    Validates output schema and assigns explicit diagnostic state:
    - scored, abstained, invalid_output, parse_error, unknown_facet
    """
    if not isinstance(parsed_items, list):
        return [], "invalid_output"

    validated = []
    for item in parsed_items:
        if not isinstance(item, dict):
            continue

        fid = str(item.get("facet_id", item.get("facet", ""))).strip()
        raw_score = item.get("score", 0.0)
        try:
            score = float(raw_score)
        except (ValueError, TypeError):
            score = 0.0

        status = str(item.get("status", "abstained")).strip().lower()
        if status not in ["scored", "abstained", "not_observable", "insufficient_evidence"]:
            status = "scored" if score > 0.0 else "abstained"

        validated.append({
            "facet_id": fid,
            "score": score,
            "status": status,
            "confidence": str(item.get("confidence", "medium")),
            "reasoning": str(item.get("reasoning", ""))
        })

    return validated, "valid"


def query_qwen_colab(text: str, candidate_facets: list):
    """Sends inference request to Qwen and records monotonic latency."""
    facet_lines = []
    for d in candidate_facets:
        fid = d["facet_id"]
        name = normalize_facet_name(d["normalized_facet"])
        desc = d.get("definition", d.get("scoring_definition", ""))
        facet_lines.append(f"- [{fid}] {name}: {desc}")

    facets_text = "\n".join(facet_lines)

    system_prompt = (
        "You are a psychological facet evaluator. Assess if the user's text exhibits any of the given candidate traits.\n"
        "For matching traits, set status to 'scored' and assign a score from 1.0 to 5.0.\n"
        "For unmatched traits, set status to 'abstained' and score to 0.0.\n"
        "Keep reasoning short (1 concise sentence per trait).\n"
        "Return ONLY a raw JSON array of objects with keys: facet_id, facet_name, score, status, confidence, reasoning."
    )

    user_prompt = (
        f"CONVERSATION TEXT:\n\"{text}\"\n\n"
        f"CANDIDATE FACETS:\n{facets_text}\n\n"
        "Output JSON array:"
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 2048
    }

    start_time = time.perf_counter()
    raw_content = ""

    for attempt in range(1, 4):
        try:
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            json_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                COLAB_URL,
                data=json_data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0",
                    "ngrok-skip-browser-warning": "true"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                resp_body = json.loads(resp.read().decode("utf-8"))
                raw_content = resp_body["choices"][0]["message"]["content"].strip()
                parsed, parse_state = extract_json_results(raw_content)
                return parsed, raw_content, elapsed_ms, parse_state, user_prompt
        except Exception as err:
            if attempt < 3:
                time.sleep(2)
                continue
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return [], str(err), elapsed_ms, "inference_error", user_prompt


def debug_qwen_cases():
    phase7_dir = PROJECT_ROOT / "outputs" / "experiments" / "phase7"
    phase7_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = PROJECT_ROOT / "data" / "processed" / "enriched_facets.json"
    csv_path = PROJECT_ROOT / "facet_evaluation_test_set_50.csv"

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog_docs = json.load(f)

    # Build Canonical Maps (NEVER use CSV row numbers)
    facet_by_id = {d["facet_id"]: d for d in catalog_docs}
    facet_by_normalized_name = {}

    for d in catalog_docs:
        norm = normalize_facet_name(d["normalized_facet"]).lower()
        if norm in facet_by_normalized_name:
            print(f"Warning: Collision for normalized facet name '{norm}'!")
        facet_by_normalized_name[norm] = d

    # Dense Vector Indexer
    observable_docs = [d for d in catalog_docs if d.get("conversation_observable", True) is True]
    bm25 = BM25Indexer().fit(observable_docs)
    dense = DenseVectorIndexer(cache_dir=str(phase7_dir)).fit(observable_docs)

    df_50 = pd.read_csv(csv_path)
    target_test_ids = ["TEST_001", "TEST_002", "TEST_003", "TEST_004"]
    df_focused = df_50[df_50["test_id"].isin(target_test_ids)]

    jsonl_path = phase7_dir / "phase7_focused_001_004_raw_outputs.jsonl"
    csv_out_path = phase7_dir / "phase7_focused_001_004.csv"

    jsonl_entries = []
    csv_rows = []

    print("======================================================================")
    print("RUNNING FOCUSED DIAGNOSTIC RUNNER (TEST_001 to TEST_004)")
    print("======================================================================")

    for idx, row in df_focused.iterrows():
        case_id = str(row["test_id"]).strip()
        text = str(row["text"]).strip()
        orig_expected_facet = str(row["expected_facet"]).strip()
        norm_expected_facet = normalize_facet_name(orig_expected_facet)
        exp_status = str(row["expected_status"]).strip()

        # Step 5: Canonical Facet Resolution
        matched_doc = facet_by_normalized_name.get(norm_expected_facet.lower())
        cat_fid = matched_doc["facet_id"] if matched_doc else "MISSING"

        # Step 4: Candidate Retrieval (Top 10)
        res_bm25 = bm25.search(text, top_k=10)
        res_dense = dense.search(text, top_k=10)

        rrf_scores = {}
        doc_map = {}
        for r, (d, _) in enumerate(res_bm25, 1):
            fid = d["facet_id"]; doc_map[fid] = d; rrf_scores[fid] = rrf_scores.get(fid, 0.0) + (1.0 / (60.0 + r))
        for r, (d, _) in enumerate(res_dense, 1):
            fid = d["facet_id"]; doc_map[fid] = d; rrf_scores[fid] = rrf_scores.get(fid, 0.0) + (1.0 / (60.0 + r))

        sorted_fids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top10_cands = [doc_map[fid] for fid in sorted_fids[:10]]

        # Query Qwen GPU with Monotonic Timer
        parsed_results, raw_output, lat_ms, parse_state, prompt_sent = query_qwen_colab(text, top10_cands)

        # Match target result
        target_res = None
        for r in parsed_results:
            rfid = str(r.get("facet_id", "")).strip().lower()
            rname = str(r.get("facet_name", r.get("facet", ""))).strip().lower()
            if rfid == cat_fid.lower() or rname == norm_expected_facet.lower() or (cat_fid != "MISSING" and cat_fid.lower() in rfid):
                target_res = r
                break

        pred_status = str(target_res.get("status", "abstained")).lower() if target_res else "abstained"
        pred_score = float(target_res.get("score", 0.0)) if target_res else 0.0
        is_scored = (pred_status == "scored") or (pred_score > 0.0)
        pred_display = "scored" if is_scored else "abstained"

        # Determine Primary Failure Mode
        failure_mode = "NONE"
        is_pass = False

        if exp_status == "scored":
            if is_scored and pred_score > 0.0:
                is_pass = True
            else:
                if parse_state in ["parse_error", "empty_output"] or not parsed_results:
                    failure_mode = "PARSER_FAILURE"
                elif cat_fid == "MISSING":
                    failure_mode = "FACET_RESOLUTION_FAILURE"
                elif cat_fid not in [c["facet_id"] for c in top10_cands]:
                    failure_mode = "RETRIEVAL_FAILURE"
                else:
                    failure_mode = "MODEL_SEMANTIC_FAILURE"
        elif exp_status in ["not_observable", "abstained"]:
            if not is_scored or pred_score == 0.0:
                is_pass = True
            else:
                failure_mode = "MODEL_SEMANTIC_FAILURE"

        pass_str = "[PASS]" if is_pass else "[FAIL]"
        print(f"[{case_id}] Trait: {norm_expected_facet:<30} | {pass_str} | Status: [{pred_display}] Score: {pred_score} | Latency: {round(lat_ms)}ms | Mode: {failure_mode}")

        # Save Raw JSONL Entry (Step 3 Requirement)
        jsonl_entries.append({
            "test_id": case_id,
            "text": text,
            "expected_facet": norm_expected_facet,
            "resolved_catalog_facet_id": cat_fid,
            "candidate_facet_ids": [c["facet_id"] for c in top10_cands],
            "prompt_sent": prompt_sent,
            "raw_qwen_output": raw_output,
            "parsed_json": parsed_results,
            "parse_state": parse_state,
            "latency_ms": round(lat_ms, 2),
            "is_pass": is_pass,
            "failure_mode": failure_mode
        })

        csv_rows.append({
            "test_id": case_id,
            "expected_facet": norm_expected_facet,
            "resolved_facet_id": cat_fid,
            "status": pred_display,
            "score": pred_score,
            "latency_ms": round(lat_ms, 2),
            "parse_state": parse_state,
            "failure_mode": failure_mode,
            "is_pass": is_pass
        })

    # Save to disk
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"\n[Diagnostic] Saved raw JSONL outputs -> {jsonl_path}")

    pd.DataFrame(csv_rows).to_csv(csv_out_path, index=False)
    print(f"[Diagnostic] Saved CSV summary -> {csv_out_path}")


if __name__ == "__main__":
    debug_qwen_cases()
