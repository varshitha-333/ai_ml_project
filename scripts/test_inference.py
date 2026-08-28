"""
End-to-End Inference Diagnostics and Health Testing CLI Script.

Tests Colab network connectivity, model status, direct endpoint generation latency,
and FastAPI backend orchestration.
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_diagnostics():
    print("==================================================================")
    print("  FACET EVALUATOR - END-TO-END INFERENCE DIAGNOSTIC SUITE")
    print("==================================================================")

    inference_url = os.getenv("INFERENCE_URL", "http://localhost:8000").rstrip("/")
    for suffix in ["/v1/chat/completions", "/v1/models", "/v1"]:
        if inference_url.endswith(suffix):
            inference_url = inference_url[:-len(suffix)].rstrip("/")

    model_name = os.getenv("INFERENCE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    backend_mode = os.getenv("BACKEND_MODE", "mock")

    print(f"[*] Configuration:")
    print(f"    - BACKEND_MODE:  {backend_mode}")
    print(f"    - INFERENCE_URL: {inference_url}")
    print(f"    - MODEL_NAME:    {model_name}\n")

    # Step 1: Test Direct Inference Model Health Endpoint
    print("[1] Testing Inference Endpoint Health (GET /v1/models)...")
    health_url = f"{inference_url}/v1/models"
    h_start = time.time()
    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "FacetDiagnostic/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            h_ms = int((time.time() - h_start) * 1000)
            if resp.status == 200:
                print(f"    [PASS] HTTP {resp.status} (latency: {h_ms}ms)")
            else:
                print(f"    [FAIL] HTTP {resp.status} (latency: {h_ms}ms)")
    except Exception as err:
        h_ms = int((time.time() - h_start) * 1000)
        print(f"    [INFO] Endpoint status at {health_url}: {err} (latency: {h_ms}ms)")

    # Step 2: Test Minimal Direct Chat Completion Call
    print("\n[2] Testing Direct Inference Generation (POST /v1/chat/completions)...")
    comp_url = f"{inference_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a JSON evaluator."},
            {"role": "user", "content": "Evaluate 'Risktaking' for text 'skydiving'. Output JSON array."}
        ],
        "temperature": 0.1,
        "max_tokens": 150
    }
    data = json.dumps(payload).encode("utf-8")
    
    g_start = time.time()
    try:
        req = urllib.request.Request(
            comp_url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "FacetDiagnostic/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            g_ms = int((time.time() - g_start) * 1000)
            if resp.status == 200:
                res_body = json.loads(resp.read().decode("utf-8"))
                choices = res_body.get("choices", [])
                content = choices[0]["message"]["content"] if choices else ""
                print(f"    [PASS] HTTP 200 (generation latency: {g_ms}ms, response_chars: {len(content)})")
                print(f"    Sample Output: {content[:100]}...")
            else:
                print(f"    [FAIL] HTTP {resp.status} (latency: {g_ms}ms)")
    except Exception as err:
        g_ms = int((time.time() - g_start) * 1000)
        print(f"    [INFO] Direct completion request at {comp_url}: {err} (latency: {g_ms}ms)")

    # Step 3: Test FastAPI Local Backend Endpoint
    print("\n[3] Testing Local FastAPI Backend (POST http://localhost:8000/evaluate)...")
    api_url = "http://localhost:8000/evaluate"
    eval_payload = {"conversation": "I am taking a wild risk going skydiving!"}
    eval_data = json.dumps(eval_payload).encode("utf-8")

    a_start = time.time()
    try:
        req = urllib.request.Request(
            api_url,
            data=eval_data,
            headers={"Content-Type": "application/json", "User-Agent": "FacetDiagnostic/1.0"}
        )
        with urllib.request.urlopen(req, timeout=35) as resp:
            a_ms = int((time.time() - a_start) * 1000)
            if resp.status == 200:
                res_json = json.loads(resp.read().decode("utf-8"))
                results = res_json.get("results", [])
                metadata = res_json.get("metadata", {})
                scored_cnt = metadata.get("scored_count", 0)
                abstained_cnt = metadata.get("abstained_count", 0)
                print(f"    [PASS] HTTP 200 (pipeline latency: {a_ms}ms)")
                print(f"    Summary: {len(results)} candidate facets evaluated ({scored_cnt} Scored, {abstained_cnt} Abstained)")
                
                # Check for inference error fallback status
                inf_errors = [r for r in results if r.get("status") == "inference_error"]
                if inf_errors:
                    print(f"    [WARN] {len(inf_errors)} facets marked with status='inference_error' (Infrastructure failure).")
            else:
                print(f"    [FAIL] HTTP {resp.status} (latency: {a_ms}ms)")
    except Exception as err:
        a_ms = int((time.time() - a_start) * 1000)
        print(f"    [INFO] Local FastAPI request at {api_url}: {err} (latency: {a_ms}ms)")

    print("\n==================================================================")
    print("  DIAGNOSTIC TEST COMPLETE")
    print("==================================================================")


if __name__ == "__main__":
    run_diagnostics()
