"""
Dedicated 30-Case Validation Runner Script.
Evaluates the first 30 curated test cases from facet_evaluation_test_set_50.csv using K=10 over real Colab Qwen GPU.
"""

import sys
from pathlib import Path
import json
import time
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator_pipeline import FacetEvaluatorPipeline
from src.scoring.inference_backend import RemoteInferenceClientBackend
from src.scoring.inference_client import InferenceClient
from src.api.config import get_settings


def run_30_validation():
    csv_path = PROJECT_ROOT / "facet_evaluation_test_set_50.csv"
    output_md = PROJECT_ROOT / "outputs" / "external_30_test_report.md"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        return

    df_30 = pd.read_csv(csv_path).iloc[:30]

    settings = get_settings()
    client = InferenceClient(
        inference_url=settings.inference_url,
        model_name=settings.model_name,
        timeout=settings.inference_timeout
    )
    backend = RemoteInferenceClientBackend(client=client)
    pipeline = FacetEvaluatorPipeline(backend=backend, top_k=10, batch_size=10)
    pipeline.initialize()

    print("=" * 70)
    print("30-CASE REAL QWEN GPU VALIDATION RUNNER")
    print("=" * 70)
    print(f"BACKEND MODE:       REMOTE")
    print(f"MODEL:              {settings.model_name}")
    print(f"RETRIEVAL K:        10")
    print(f"TOTAL CASES:        {len(df_30)}")
    print("=" * 70)

    results = []
    correct_status = 0
    scored_count = 0
    abstained_count = 0
    false_scoring_count = 0
    latencies = []

    for idx, row in df_30.iterrows():
        case_id = str(row.get("test_id", f"TEST_{idx+1:03d}"))
        text = str(row["text"]).strip()
        expected_facet = str(row["expected_facet"]).strip()
        expected_status = str(row["expected_status"]).strip()

        t0 = time.time()
        res = pipeline.evaluate_conversation(text)
        t_ms = (time.time() - t0) * 1000
        latencies.append(t_ms)

        all_res = res.evaluated_results + res.abstained_results
        
        # Target matching
        target_status = "insufficient_evidence"
        target_score = None

        for item in all_res:
            fname = item.facet.strip().lower()
            if fname == expected_facet.lower() or fname in expected_facet.lower() or expected_facet.lower() in fname:
                target_status = item.status
                target_score = item.score
                break

        is_correct = (target_status == expected_status)
        if is_correct:
            correct_status += 1

        if target_status == "scored":
            scored_count += 1
        else:
            abstained_count += 1

        if expected_status == "not_observable" and target_status == "scored":
            false_scoring_count += 1

        status_icon = "PASS" if is_correct else "FAIL"
        print(f"[{case_id}] Trait: {expected_facet:<35} | {status_icon:^6} | Status: [{target_status}] | Latency: {t_ms:.0f}ms")

        results.append({
            "id": case_id,
            "expected_facet": expected_facet,
            "expected_status": expected_status,
            "predicted_status": target_status,
            "predicted_score": target_score,
            "is_correct": is_correct,
            "latency_ms": round(t_ms, 2)
        })

    acc_pct = round((correct_status / len(df_30)) * 100, 2)
    avg_lat = round(float(np.mean(latencies)), 2)
    false_rate = round((false_scoring_count / len(df_30)) * 100, 2)

    summary_md = f"""# REAL QWEN 30-CASE EXTERNAL VALIDATION REPORT

**BACKEND MODE**: `REMOTE`  
**MODEL**: `{settings.model_name}`  
**RETRIEVAL K**: `10`  
**TOTAL CASES**: `{len(df_30)}`  
**STATUS ACCURACY**: `{acc_pct}%` ({correct_status}/{len(df_30)})  
**SCORED FACETS**: `{scored_count}`  
**ABSTAINED FACETS**: `{abstained_count}`  
**FALSE SCORING RATE**: `{false_rate}%` (Zero Hallucination Target)  
**AVERAGE LATENCY**: `{avg_lat} ms`

---

## 📊 Per-Case Evaluation Results

| Case ID | Expected Trait | Expected Status | Predicted Status | Score | Verdict | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for r in results:
        v_str = "✅ PASS" if r["is_correct"] else "❌ FAIL"
        score_str = str(r["predicted_score"]) if r["predicted_score"] is not None else "null"
        summary_md += f"| `{r['id']}` | `{r['expected_facet']}` | `{r['expected_status']}` | `{r['predicted_status']}` | `{score_str}` | {v_str} | `{r['latency_ms']} ms` |\n"

    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print("=" * 70)
    print(f"30-CASE VALIDATION COMPLETE!")
    print(f"Status Accuracy:    {acc_pct}%")
    print(f"False Scoring Rate: {false_rate}%")
    print(f"Average Latency:    {avg_lat} ms")
    print(f"Report Saved To:    {output_md}")
    print("=" * 70)


if __name__ == "__main__":
    run_30_validation()
