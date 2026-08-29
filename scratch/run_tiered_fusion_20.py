"""
Tiered Multi-Model Candidate Fusion (K <= 20) Evaluation Suite.
Combines:
- Model 1 (Short Unigram / Lexical Engine for short traits: Naivety, Aloofness, Moroseness) -> Top 10
- Model 2 (Multi-Word Semantic Engine for long phrases: assertiveness & control in relationships) -> Top 10
Merges both candidate sets into a single deduplicated list (<= 20 candidates) for Qwen GPU scoring.
"""

import sys
from pathlib import Path
import json
import time
import re
import math
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scratch.debug_qwen_cases import (
    query_qwen_colab,
    normalize_facet_name,
    BM25Indexer,
    DenseVectorIndexer
)


def run_tiered_fusion_validation():
    exp_dir = PROJECT_ROOT / "outputs" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = PROJECT_ROOT / "data" / "processed" / "enriched_facets.json"
    csv_path = PROJECT_ROOT / "facet_evaluation_test_set_50.csv"

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog_docs = json.load(f)

    facet_by_normalized_name = {normalize_facet_name(d["normalized_facet"]).lower(): d for d in catalog_docs}
    observable_docs = [d for d in catalog_docs if d.get("conversation_observable", True) is True]

    # Model 1: Lexical / Unigram Engine
    bm25_model1 = BM25Indexer().fit(observable_docs)

    # Model 2: Multi-Word Semantic Engine (Solution 1)
    dense_model2 = DenseVectorIndexer(cache_dir=str(exp_dir)).fit(observable_docs)

    df_30 = pd.read_csv(csv_path).iloc[:30]

    case_logs = []
    retrieval_target_ranks = []
    latencies_ms = []

    parse_success_count = 0
    valid_output_count = 0

    print("======================================================================")
    print("RUNNING TIERED MULTI-MODEL CANDIDATE FUSION (K <= 20) VALIDATION SUITE")
    print("======================================================================")

    for idx, row in df_30.iterrows():
        case_id = str(row.get("test_id", f"TEST_{idx+1:03d}")).strip()
        text = str(row["text"]).strip()
        orig_facet = str(row["expected_facet"]).strip()
        norm_facet = normalize_facet_name(orig_facet)
        exp_status = str(row["expected_status"]).strip()

        matched_doc = facet_by_normalized_name.get(norm_facet.lower())
        cat_fid = matched_doc["facet_id"] if matched_doc else "MISSING"

        # Model 1: Top 10 Short/Unigram Candidates
        res_m1 = bm25_model1.search(text, top_k=10)
        cands_m1 = [d for d, _ in res_m1]

        # Model 2: Top 10 Multi-Word Semantic Candidates
        res_m2 = dense_model2.search(text, top_k=10)
        cands_m2 = [d for d, _ in res_m2]

        # Tiered Candidate Fusion (Deduplicated <= 20 Candidates)
        fused_map = {}
        for d in cands_m1 + cands_m2:
            fid = d["facet_id"]
            if fid not in fused_map:
                fused_map[fid] = d

        fused_cands = list(fused_map.values())[:20]
        fused_ids = [d["facet_id"] for d in fused_cands]

        target_rank = 999
        if cat_fid != "MISSING" and cat_fid in fused_ids:
            target_rank = fused_ids.index(cat_fid) + 1

        retrieval_target_ranks.append(target_rank)

        # 2-Pass Batched Qwen Inference (2 x 10 candidates max per prompt to eliminate competition decay)
        parsed_results = []
        lat_ms = 0.0

        batch1 = fused_cands[:10]
        batch2 = fused_cands[10:20]

        p1, _, l1, s1, _ = query_qwen_colab(text, batch1)
        lat_ms += l1
        if isinstance(p1, list):
            parsed_results.extend(p1)

        s2 = "valid_array"
        if batch2:
            p2, _, l2, s2, _ = query_qwen_colab(text, batch2)
            lat_ms += l2
            if isinstance(p2, list):
                parsed_results.extend(p2)

        parse_state = s1 if s1 in ["valid_array", "wrapped_object", "single_object", "truncated_recovery"] else s2
        latencies_ms.append(lat_ms)

        if parse_state in ["valid_array", "wrapped_object", "single_object", "truncated_recovery"]:
            parse_success_count += 1
            valid_output_count += 1

        # Match target result
        target_res = None
        if isinstance(parsed_results, list):
            for r in parsed_results:
                if isinstance(r, dict):
                    rfid = str(r.get("facet_id", "")).strip().lower()
                    rname = str(r.get("facet_name", r.get("facet", ""))).strip().lower()
                    if rfid == cat_fid.lower() or rname == norm_facet.lower() or (cat_fid != "MISSING" and cat_fid.lower() in rfid):
                        target_res = r
                        break

        pred_status = str(target_res.get("status", "abstained")).lower() if target_res else "abstained"
        pred_score = float(target_res.get("score", 0.0)) if target_res else 0.0
        is_scored = (pred_status == "scored") or (pred_score > 0.0)
        pred_display = "scored" if is_scored else "abstained"

        failure_category = "NONE"
        is_pass = False

        if exp_status == "scored":
            if is_scored and pred_score > 0.0:
                is_pass = True
            else:
                if parse_state == "inference_error":
                    failure_category = "INFERENCE_FAILURE"
                elif parse_state not in ["valid_array", "wrapped_object", "single_object", "truncated_recovery"]:
                    failure_category = "PARSER_FAILURE"
                elif cat_fid == "MISSING":
                    failure_category = "FACET_RESOLUTION_FAILURE"
                elif target_rank > len(fused_cands):
                    failure_category = "RETRIEVAL_FAILURE"
                else:
                    failure_category = "MODEL_SEMANTIC_FAILURE"
        elif exp_status in ["not_observable", "abstained"]:
            if not is_scored or pred_score == 0.0:
                is_pass = True
            else:
                failure_category = "MODEL_SEMANTIC_FAILURE"

        pass_str = "[PASS]" if is_pass else "[FAIL]"
        print(f"[{case_id}] Trait: {norm_facet:<35} | {pass_str} | Status: [{pred_display}] Score: {pred_score} | Latency: {round(lat_ms)}ms | Category: {failure_category}")

        case_logs.append({
            "test_id": case_id,
            "text": text,
            "expected_facet": norm_facet,
            "resolved_facet_id": cat_fid,
            "fused_candidate_count": len(fused_cands),
            "target_rank": target_rank,
            "predicted_status": pred_display,
            "predicted_score": pred_score,
            "latency_ms": round(lat_ms, 2),
            "parse_state": parse_state,
            "failure_category": failure_category,
            "is_pass": is_pass
        })

    # Compute Metrics
    N = len(case_logs)
    end_to_end_acc = (sum(1 for c in case_logs if c["is_pass"]) / N) * 100

    r10 = sum(1 for r in retrieval_target_ranks if 1 <= r <= 10) / N * 100
    r20 = sum(1 for r in retrieval_target_ranks if 1 <= r <= 20) / N * 100
    mrr = sum(1.0 / r for r in retrieval_target_ranks if r <= 20) / N

    p95_lat = float(np.percentile(latencies_ms, 95))
    mean_lat = float(np.mean(latencies_ms))

    # Save Report
    report_path = exp_dir / "tiered_fusion_20_results.md"
    report_md = f"""# Tiered Multi-Model Candidate Fusion (K <= 20) Evaluation Report

---

## 🏆 Executive Summary Metrics
- **End-to-End System Accuracy**: **`{round(end_to_end_acc, 2)}%`**
- **Parse Success Rate**: **`{round(parse_success_count / N * 100, 1)}%`** (`{parse_success_count}/{N}`)
- **Fused Candidate Recall@10**: **`{round(r10, 2)}%`**
- **Fused Candidate Recall@20**: **`{round(r20, 2)}%`**
- **Fused MRR**: **`{round(mrr, 4)}`**
- **Mean Latency**: **`{round(mean_lat, 2)} ms`** | P95: **`{round(p95_lat, 2)} ms`**

---

## 📌 Architecture Strategy Summary
Fused Top-10 Short/Unigram Candidates (BM25 Lexical) + Top-10 Multi-Word Semantic Candidates (Solution 1 Dense) into a deduplicated candidate pool ($\le 20$ candidates).

---

## 💻 Exact Command to Reproduce:

```powershell
.\\venv\\Scripts\\python.exe scratch/run_tiered_fusion_20.py
```
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    pd.DataFrame(case_logs).to_csv(exp_dir / "tiered_fusion_20_results.csv", index=False)

    print("\n" + "=" * 70)
    print("TIERED MULTI-MODEL CANDIDATE FUSION (K <= 20) COMPLETE!")
    print(f"End-to-End Accuracy:          {round(end_to_end_acc, 2)}%")
    print(f"Parse Success Rate:           {round(parse_success_count / N * 100, 1)}%")
    print(f"Fused Candidate Recall@20:    {round(r20, 2)}%")
    print(f"P95 Latency:                  {round(p95_lat, 2)} ms")
    print(f"Report Saved -> {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_tiered_fusion_validation()
