"""
Diagnostic Script to test Docker / FastAPI backend with 30 curated conversation requests.
Sends POST http://localhost:8000/evaluate requests over HTTP and computes accuracy metrics.
"""

import sys
import time
import json
import urllib.request
import urllib.error
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_docker_30_cases_test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/evaluate")
    args, _ = parser.parse_known_args()
    
    csv_path = PROJECT_ROOT / "facet_evaluation_test_set_50.csv"
    api_url = args.url

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        return

    df_30 = pd.read_csv(csv_path).iloc[:30]

    print("=" * 70)
    print("--- 30-Case Curated Docker / FastAPI Diagnostic Test ---")
    print(f"Target URL: {api_url}")
    print(f"Test Cases: {len(df_30)}")
    print("=" * 70)

    results = []
    total_latency_ms = 0.0
    correct_status_count = 0
    scored_count = 0
    abstained_count = 0
    false_scoring_count = 0

    for idx, row in df_30.iterrows():
        case_id = str(row.get("test_id", f"CASE_{idx+1:02d}"))
        text = str(row["text"]).strip()
        expected_facet = str(row["expected_facet"]).strip()
        expected_status = str(row["expected_status"]).strip()
        expected_score = row["expected_score"]

        payload = {"conversation": text}
        json_data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            api_url,
            data=json_data,
            headers={"Content-Type": "application/json"}
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                status_code = resp.status
                response_body = resp.read().decode("utf-8")
                res_data = json.loads(response_body)
                latency_ms = (time.time() - t0) * 1000
                total_latency_ms += latency_ms

                # Check predictions
                evaluated_facets = res_data.get("results", res_data.get("evaluated_facets", []))
                
                # Find matching target facet evaluation
                found_target = False
                target_status = "insufficient_evidence"
                target_score = None

                for item in evaluated_facets:
                    fname = (item.get("facet") or item.get("normalized_facet") or item.get("facet_name") or "").strip().lower()
                    fid = (item.get("facet_id") or "").strip().upper()
                    
                    if fname == expected_facet.lower() or fname in expected_facet.lower() or expected_facet.lower() in fname:
                        found_target = True
                        target_status = item.get("status", "insufficient_evidence")
                        target_score = item.get("score")
                        break

                if not found_target and evaluated_facets:
                    # Check top evaluated item
                    top_item = evaluated_facets[0]
                    target_status = top_item.get("status")
                    target_score = top_item.get("score")

                # Metrics check
                status_correct = (target_status == expected_status)
                if status_correct:
                    correct_status_count += 1

                if target_status == "scored":
                    scored_count += 1
                else:
                    abstained_count += 1

                # False scoring check
                if expected_status == "not_observable" and target_status == "scored":
                    false_scoring_count += 1

                results.append({
                    "id": case_id,
                    "expected_facet": expected_facet,
                    "expected_status": expected_status,
                    "predicted_status": target_status,
                    "predicted_score": target_score,
                    "status_correct": status_correct,
                    "latency_ms": round(latency_ms, 2)
                })

                status_str = "✅ PASS" if status_correct else "❌ FAIL"
                print(f"[{case_id}] Trait: {expected_facet:<35} | {status_str} | Status: [{target_status}] | Latency: {latency_ms:.0f}ms")

        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            print(f"[{case_id}] Trait: {expected_facet:<35} | ERROR: {e} | Latency: {latency_ms:.0f}ms")

    status_accuracy_pct = round((correct_status_count / len(df_30)) * 100, 2)
    avg_latency = round(total_latency_ms / len(df_30), 2)
    false_scoring_rate = round((false_scoring_count / len(df_30)) * 100, 2)

    print("=" * 70)
    print("--- 30-Case Test Summary Results ---")
    print(f"Total Test Cases:        {len(df_30)}")
    print(f"Status Accuracy:         {status_accuracy_pct}% ({correct_status_count}/{len(df_30)})")
    print(f"Scored Facets:           {scored_count}")
    print(f"Abstained Facets:        {abstained_count}")
    print(f"False Scoring Rate:      {false_scoring_rate}% (Zero Hallucination Target)")
    print(f"Average Request Latency: {avg_latency} ms")
    print("=" * 70)


if __name__ == "__main__":
    run_docker_30_cases_test()
