"""
20-Run Stability Test Runner for TEST_001 to TEST_004 (5 runs per test case).
Verifies determinism, parse success rates, and latency stability.
Saves outputs/experiments/phase7/phase7_stability_20_runs.csv.
"""

import sys
from pathlib import Path
import json
import time
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scratch.debug_qwen_cases import (
    query_qwen_colab,
    normalize_facet_name,
    BM25Indexer,
    DenseVectorIndexer
)


def run_stability_test():
    phase7_dir = PROJECT_ROOT / "outputs" / "experiments" / "phase7"
    phase7_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = PROJECT_ROOT / "data" / "processed" / "enriched_facets.json"
    csv_path = PROJECT_ROOT / "facet_evaluation_test_set_50.csv"

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog_docs = json.load(f)

    facet_by_normalized_name = {normalize_facet_name(d["normalized_facet"]).lower(): d for d in catalog_docs}
    observable_docs = [d for d in catalog_docs if d.get("conversation_observable", True) is True]

    bm25 = BM25Indexer().fit(observable_docs)
    dense = DenseVectorIndexer(cache_dir=str(phase7_dir)).fit(observable_docs)

    df_50 = pd.read_csv(csv_path)
    target_ids = ["TEST_001", "TEST_002", "TEST_003", "TEST_004"]
    df_targets = df_50[df_50["test_id"].isin(target_ids)]

    stability_rows = []

    print("======================================================================")
    print("RUNNING 20-RUN STABILITY TEST (5 RUNS x 4 CASES)")
    print("======================================================================")

    for run_num in range(1, 6):
        print(f"\n--- Stability Iteration {run_num} / 5 ---")
        for idx, row in df_targets.iterrows():
            case_id = str(row["test_id"]).strip()
            text = str(row["text"]).strip()
            orig_facet = str(row["expected_facet"]).strip()
            norm_facet = normalize_facet_name(orig_facet)
            exp_status = str(row["expected_status"]).strip()

            matched_doc = facet_by_normalized_name.get(norm_facet.lower())
            cat_fid = matched_doc["facet_id"] if matched_doc else "MISSING"

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

            parsed_results, raw_output, lat_ms, parse_state, _ = query_qwen_colab(text, top10_cands)

            target_res = None
            for r in parsed_results:
                rfid = str(r.get("facet_id", "")).strip().lower()
                rname = str(r.get("facet_name", r.get("facet", ""))).strip().lower()
                if rfid == cat_fid.lower() or rname == norm_facet.lower() or (cat_fid != "MISSING" and cat_fid.lower() in rfid):
                    target_res = r
                    break

            pred_status = str(target_res.get("status", "abstained")).lower() if target_res else "abstained"
            pred_score = float(target_res.get("score", 0.0)) if target_res else 0.0
            is_scored = (pred_status == "scored") or (pred_score > 0.0)
            pred_display = "scored" if is_scored else "abstained"

            is_pass = False
            if exp_status == "scored":
                if is_scored and pred_score > 0.0:
                    is_pass = True
            elif exp_status in ["not_observable", "abstained"]:
                if not is_scored or pred_score == 0.0:
                    is_pass = True

            pass_str = "[PASS]" if is_pass else "[FAIL]"
            print(f"Iter {run_num} | [{case_id}] Trait: {norm_facet:<25} | {pass_str} | Status: [{pred_display}] Score: {pred_score} | Latency: {round(lat_ms)}ms")

            stability_rows.append({
                "run_number": run_num,
                "test_id": case_id,
                "expected_facet": norm_facet,
                "catalog_facet_id": cat_fid,
                "status": pred_display,
                "score": pred_score,
                "latency_ms": round(lat_ms, 2),
                "parse_state": parse_state,
                "is_pass": is_pass
            })

    out_csv = phase7_dir / "phase7_stability_20_runs.csv"
    pd.DataFrame(stability_rows).to_csv(out_csv, index=False)
    print(f"\n[Stability Test Complete] Saved 20-Run Stability CSV -> {out_csv}")


if __name__ == "__main__":
    run_stability_test()
