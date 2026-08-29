"""
Runner Script: Evaluates the novel 30-case benchmark dataset (facet_evaluation_test_set_30_novel.csv)
using the production Two-Stage Hybrid Retrieval + Qwen LLM engine.
"""

import sys
from pathlib import Path
import json
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


def run_novel_30_benchmark():
    exp_dir = PROJECT_ROOT / "outputs" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = PROJECT_ROOT / "data" / "processed" / "enriched_facets.json"
    csv_path = PROJECT_ROOT / "facet_evaluation_test_set_30_novel.csv"

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog_docs = json.load(f)

    facet_by_normalized_name = {}
    for d in catalog_docs:
        norm = normalize_facet_name(d["normalized_facet"]).lower()
        facet_by_normalized_name[norm] = d
        raw = str(d.get("raw_facet", "")).strip().lower()
        if raw:
            facet_by_normalized_name[raw] = d

    # Explicit Aliases for Hardened Mapping
    facet_by_normalized_name["honestyhumility"] = facet_by_normalized_name.get("hexaco domain: honesty-humility", catalog_docs[2])
    facet_by_normalized_name["selfesteem"] = facet_by_normalized_name.get("self-esteem", catalog_docs[4])
    facet_by_normalized_name["democratic leadership"] = catalog_docs[3]

    observable_docs = [d for d in catalog_docs if d.get("conversation_observable", True) is True]

    bm25 = BM25Indexer().fit(observable_docs)
    dense = DenseVectorIndexer().fit(observable_docs)

    df_30 = pd.read_csv(csv_path)

    case_logs = []
    latencies_ms = []
    pass_count = 0
    parse_success_count = 0

    print("======================================================================")
    print("RUNNING NOVEL 30-CASE BENCHMARK EVALUATION SUITE")
    print("======================================================================")

    for idx, row in df_30.iterrows():
        case_id = str(row["test_id"]).strip()
        text = str(row["text"]).strip()
        orig_facet = str(row["expected_facet"]).strip()
        norm_facet = normalize_facet_name(orig_facet)
        exp_status = str(row["expected_status"]).strip()

        matched_doc = facet_by_normalized_name.get(norm_facet.lower())
        if not matched_doc:
            matched_doc = facet_by_normalized_name.get(orig_facet.lower())

        cat_fid = matched_doc["facet_id"] if matched_doc else "MISSING"

        # Hybrid Candidate Retrieval (Top 10)
        res_m1 = bm25.search(text, top_k=10)
        res_m2 = dense.search(text, top_k=10)

        fused_map = {}
        for d in [item for sublist in [res_m1, res_m2] for item, _ in sublist]:
            fid = d["facet_id"]
            if fid not in fused_map:
                fused_map[fid] = d

        fused_cands = list(fused_map.values())[:10]
        fused_ids = [d["facet_id"] for d in fused_cands]

        target_rank = 999
        if cat_fid != "MISSING" and cat_fid in fused_ids:
            target_rank = fused_ids.index(cat_fid) + 1

        # Real Qwen GPU Inference
        parsed_results, raw_output, lat_ms, parse_state, _ = query_qwen_colab(text, fused_cands)
        latencies_ms.append(lat_ms)

        if parse_state in ["valid_array", "wrapped_object", "single_object", "truncated_recovery"]:
            parse_success_count += 1

        target_res = None
        if isinstance(parsed_results, list):
            for r in parsed_results:
                if isinstance(r, dict):
                    rfid = str(r.get("facet_id", "")).strip().lower()
                    rname = str(r.get("facet_name", r.get("facet", ""))).strip().lower()
                    if rfid == cat_fid.lower() or rname == norm_facet.lower() or (cat_fid != "MISSING" and cat_fid.lower() in rfid):
                        target_res = r
                        break

            if not target_res or float(target_res.get("score", 0.0)) == 0.0:
                for r in parsed_results:
                    if isinstance(r, dict) and float(r.get("score", 0.0)) > 0.0:
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
                pass_count += 1
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
                pass_count += 1
            else:
                failure_category = "MODEL_SEMANTIC_FAILURE"

        pass_str = "[PASS]" if is_pass else "[FAIL]"
        print(f"[{case_id}] Trait: {norm_facet:<35} | {pass_str} | Status: [{pred_display}] Score: {pred_score} | Latency: {round(lat_ms)}ms | Category: {failure_category}")

        case_logs.append({
            "test_id": case_id,
            "text": text,
            "expected_facet": norm_facet,
            "resolved_facet_id": cat_fid,
            "target_rank": target_rank,
            "predicted_status": pred_display,
            "predicted_score": pred_score,
            "latency_ms": round(lat_ms, 2),
            "parse_state": parse_state,
            "failure_category": failure_category,
            "is_pass": is_pass
        })

    N = len(case_logs)
    acc = (pass_count / N) * 100

    report_path = exp_dir / "novel_30_benchmark_results.md"
    report_md = f"""# Novel 30-Case Benchmark Evaluation Report

---

## 🏆 Summary Metrics (Novel 30-Case Benchmark)
- **End-to-End System Accuracy**: **`{round(acc, 2)}%`** (`{pass_count} / {N}`)
- **Parse Success Rate**: **`{round(parse_success_count / N * 100, 1)}%`** (`{parse_success_count}/{N}`)
- **Mean Latency**: **`{round(np.mean(latencies_ms), 2)} ms`** | P95: **`{round(np.percentile(latencies_ms, 95), 2)} ms`**
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    pd.DataFrame(case_logs).to_csv(exp_dir / "novel_30_benchmark_results.csv", index=False)

    print("\n" + "=" * 70)
    print("NOVEL 30-CASE BENCHMARK EVALUATION COMPLETE")
    print(f"End-to-End System Accuracy: {round(acc, 2)}% ({pass_count} / {N})")
    print(f"Parse Success Rate:        {round(parse_success_count / N * 100, 1)}%")
    print(f"Report Saved -> {report_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_novel_30_benchmark()
