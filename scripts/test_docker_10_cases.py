"""
Diagnostic Script to test Docker / FastAPI backend with 10 real conversation requests.
Sends POST http://localhost:8000/evaluate requests over HTTP to Colab Qwen GPU model.
"""

import sys
import time
import json
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TEST_10_CONVERSATIONS = [
    {
        "id": "CASE_1",
        "text": "I am taking a wild risk going skydiving this weekend without a backup parachute!",
        "expected_trait": "Risktaking"
    },
    {
        "id": "CASE_2",
        "text": "I felt dizzy after my morning walk and my blood pressure reading was 130/85.",
        "expected_trait": "Blood pressure level (Medical Lab)"
    },
    {
        "id": "CASE_3",
        "text": "I LOVE staying past midnight fixing your typos for free because I care about perfection.",
        "expected_trait": "Acidity / Civility"
    },
    {
        "id": "CASE_4",
        "text": "He told the team I was exhibiting hesitation, but I completely disagree with his statement.",
        "expected_trait": "Hesitation (Quoted statement)"
    },
    {
        "id": "CASE_5",
        "text": "I handed in my resignation letter today without having another job lined up.",
        "expected_trait": "Risktaking"
    },
    {
        "id": "CASE_6",
        "text": "My knees were knocking in sheer terror as I stepped onto the stage.",
        "expected_trait": "Fearfulness / Dauntlessness"
    },
    {
        "id": "CASE_7",
        "text": "Could you please pass me the salt?",
        "expected_trait": "Civility"
    },
    {
        "id": "CASE_8",
        "text": "I have been feeling super tired and con flojera all afternoon.",
        "expected_trait": "Slothfulness"
    },
    {
        "id": "CASE_9",
        "text": "I feel blue and down in the dumps today.",
        "expected_trait": "Moroseness"
    },
    {
        "id": "CASE_10",
        "text": "I spent six years studying advanced statistical reasoning in Python.",
        "expected_trait": "Statistical Reasoning"
    }
]


def test_docker_10_cases(api_base_url: str = "http://localhost:8000"):
    print("==================================================================")
    print("  DOCKER / COLAB REAL INFERENCE TEST (10 CONVERSATIONS)")
    print("==================================================================")
    print(f"Target API Endpoint: {api_base_url}/evaluate\n")

    # Step 1: Health Check
    health_url = f"{api_base_url}/health"
    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "DockerTester/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            h_data = json.loads(resp.read().decode("utf-8"))
            print(f"[*] /health Check:")
            print(f"    - Status:                 {h_data.get('status')}")
            print(f"    - Model Backend:          {h_data.get('model_backend')}")
            print(f"    - Colab Inference Status: {h_data.get('colab_inference_status')}")
            print(f"    - Catalog Facets Loaded:  {h_data.get('catalog_facets_loaded')}\n")
    except Exception as err:
        print(f"❌ [HEALTH FAILED] Could not connect to {health_url}: {err}")
        print("👉 Make sure your Docker container is running: docker run -p 8000:8000 -e BACKEND_MODE=remote -e INFERENCE_URL=<NGROK_URL> facet-evaluator\n")
        return

    # Step 2: Send 10 POST /evaluate requests
    print("--- Sending 10 Evaluation Requests Over HTTP ---")
    total_latency_ms = 0
    scored_results_count = 0
    abstained_results_count = 0

    for item in TEST_10_CONVERSATIONS:
        cid = item["id"]
        text = item["text"]
        trait = item["expected_trait"]

        payload = {"conversation": text, "top_k": 10}
        data = json.dumps(payload).encode("utf-8")

        t0 = time.time()
        try:
            req = urllib.request.Request(
                f"{api_base_url}/evaluate",
                data=data,
                headers={"Content-Type": "application/json", "User-Agent": "DockerTester/1.0"}
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                elapsed_ms = int((time.time() - t0) * 1000)
                total_latency_ms += elapsed_ms

                if resp.status == 200:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    results = res_data.get("results", [])
                    meta = res_data.get("metadata", {})

                    scored_cnt = meta.get("scored_count", 0)
                    abstained_cnt = meta.get("abstained_count", 0)
                    scored_results_count += scored_cnt
                    abstained_results_count += abstained_cnt

                    top_result = results[0] if results else {}
                    t_status = top_result.get("status", "unknown")
                    t_score = top_result.get("score")
                    t_evidence = top_result.get("evidence")
                    t_facet = top_result.get("facet")

                    print(f"[{cid}] Trait: {trait:30s} | HTTP 200 | Latency: {elapsed_ms}ms")
                    print(f"       Top Facet: '{t_facet}' -> Status: [{t_status.upper()}] Score: {t_score}")
                    if t_evidence:
                        print(f"       Quote Evidence: \"{t_evidence}\"")
                    print("-" * 65)

        except Exception as err:
            elapsed_ms = int((time.time() - t0) * 1000)
            print(f"[{cid}] Trait: {trait:30s} | FAILED ({err}) | Latency: {elapsed_ms}ms\n" + "-" * 65)

    avg_lat_s = round((total_latency_ms / len(TEST_10_CONVERSATIONS)) / 1000, 2)
    print("\n==================================================================")
    print("  DOCKER REAL INFERENCE TEST SUMMARY")
    print("==================================================================")
    print(f"TOTAL CASES TESTED:        {len(TEST_10_CONVERSATIONS)}")
    print(f"TOTAL SCORED FACETS:       {scored_results_count}")
    print(f"TOTAL ABSTAINED FACETS:    {abstained_results_count}")
    print(f"AVERAGE REQUEST LATENCY:   {avg_lat_s} seconds (REAL GPU INFERENCE)")
    print("==================================================================\n")


if __name__ == "__main__":
    url_arg = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_docker_10_cases(url_arg)
