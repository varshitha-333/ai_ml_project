"""
Script to test direct HTTP connectivity to Colab Qwen GPU endpoint (https://salvaging-ardently-late.ngrok-free.dev).
Bypasses local FastAPI backend and tests Google Colab GPU directly over Ngrok.
"""

import sys
import os
import time
import json
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from src.api.config import get_settings


def test_direct_colab():
    settings = get_settings()
    colab_url = os.getenv("INFERENCE_URL", settings.inference_url).rstrip("/")
    for suffix in ["/v1/chat/completions", "/v1/models", "/v1"]:
        if colab_url.endswith(suffix):
            colab_url = colab_url[:-len(suffix)].rstrip("/")

    model_name = settings.model_name

    print("==================================================================")
    print("  DIRECT GOOGLE COLAB GPU CONNECTIVITY TEST")
    print("==================================================================")
    print(f"Colab Ngrok Target: {colab_url}")
    print(f"Target Model:       {model_name}\n")

    # Step 1: Direct GET /v1/models to Colab
    print("[1] Direct Health Check (GET /v1/models)...")
    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"{colab_url}/v1/models",
            headers={"User-Agent": "DirectColabTester/1.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            elapsed_ms = int((time.time() - t0) * 1000)
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8"))
                print(f"    [PASS] Direct Colab Health: HTTP 200 OK (latency: {elapsed_ms}ms)")
                print(f"    Models Available: {body}\n")
            else:
                print(f"    [FAIL] HTTP {resp.status}\n")
    except Exception as err:
        elapsed_ms = int((time.time() - t0) * 1000)
        print(f"    [FAIL] Direct Colab Health failed ({err}) (latency: {elapsed_ms}ms)")
        print("    [NOTE] Check if your Google Colab cell is active and Ngrok tunnel is running.\n")
        return

    # Step 2: Direct POST /v1/chat/completions to Colab Qwen Model
    print("[2] Direct Chat Completion Request (POST /v1/chat/completions)...")
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a JSON evaluator."},
            {"role": "user", "content": "Evaluate 'Risktaking' for conversation: 'I am going skydiving!' Return JSON array."}
        ],
        "temperature": 0.1,
        "max_tokens": 150
    }
    data = json.dumps(payload).encode("utf-8")

    t_start = time.time()
    try:
        req = urllib.request.Request(
            f"{colab_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "DirectColabTester/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            elapsed_ms = int((time.time() - t_start) * 1000)
            elapsed_s = round(elapsed_ms / 1000, 2)
            if resp.status == 200:
                res_body = json.loads(resp.read().decode("utf-8"))
                choices = res_body.get("choices", [])
                content = choices[0]["message"]["content"] if choices else ""
                print(f"    [PASS] Direct Colab LLM Response Received!")
                print(f"    HTTP Status:      200 OK")
                print(f"    Generation Time:  {elapsed_s} seconds (REAL COLAB GPU INFERENCE)")
                print(f"    Response Length:  {len(content)} characters")
                print(f"    Raw Output:       \"{content.strip()}\"\n")
            else:
                print(f"    [FAIL] HTTP {resp.status}\n")
    except Exception as err:
        elapsed_ms = int((time.time() - t_start) * 1000)
        print(f"    [FAIL] Direct Chat Completion failed: {err} (latency: {elapsed_ms}ms)\n")

    print("==================================================================")
    print("  DIRECT COLAB TEST COMPLETE")
    print("==================================================================")


if __name__ == "__main__":
    test_direct_colab()
