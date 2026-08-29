"""
Comprehensive Retrieval & Reranker Ablation Study.
Compares BM25 Only, Dense Only, Hybrid RRF Baseline, and Hybrid + Cross-Encoder Reranker.
"""

import sys
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25 import BM25Indexer
from src.retrieval.indexer import DenseVectorIndexer
from src.retrieval.search import HybridFacetRetriever
from src.retrieval.reranker import CrossEncoderReranker


def calculate_mrr(retrieved_ids: list, relevant_ids: set) -> float:
    """Calculates Mean Reciprocal Rank for a single query."""
    for rank, fid in enumerate(retrieved_ids, 1):
        if fid in relevant_ids:
            return 1.0 / rank
    return 0.0


def run_retrieval_ablation():
    data_path = PROJECT_ROOT / "data" / "processed" / "enriched_facets.json"
    ref_path = PROJECT_ROOT / "data" / "benchmark_reference_set.json"
    report_json = PROJECT_ROOT / "outputs" / "retrieval_ablation_report.json"
    report_md = PROJECT_ROOT / "outputs" / "retrieval_ablation_report.md"

    with open(data_path, "r", encoding="utf-8") as f:
        all_docs = json.load(f)

    # Filter to observable facets
    observable_docs = [d for d in all_docs if d.get("conversation_observable", True) is True]

    with open(ref_path, "r", encoding="utf-8") as f:
        benchmark_set = json.load(f)

    # Pre-fit indexers
    bm25 = BM25Indexer().fit(observable_docs)
    dense = DenseVectorIndexer().fit(observable_docs)
    hybrid_baseline = HybridFacetRetriever(default_top_k=30, reranker_enabled=False).fit(observable_docs)

    # Measure CrossEncoder Load Time
    t_load_start = time.time()
    reranker = CrossEncoderReranker()
    reranker.load_model()
    cross_encoder_load_ms = round((time.time() - t_load_start) * 1000, 2)

    strategies = [
        "bm25_only",
        "dense_only",
        "hybrid_rrf_baseline",
        "hybrid_rrf_crossencoder_n30",
        "hybrid_rrf_crossencoder_n50"
    ]

    metrics = {
        s: {
            "recall_at_5": [],
            "recall_at_10": [],
            "recall_at_30": [],
            "mrr": [],
            "latencies_ms": []
        } for s in strategies
    }

    # Micro timing metrics
    micro_timing = {
        "bm25_latency_ms": [],
        "dense_latency_ms": [],
        "rrf_fusion_latency_ms": [],
        "crossencoder_inference_ms": []
    }

    per_case_comparison = []

    for item in benchmark_set:
        conv_id = item.get("conversation_id", "")
        text = item["conversation_text"]
        relevant_fids = {ann["facet_id"] for ann in item["annotations"] if ann["expected_status"] == "scored"}

        if not relevant_fids:
            continue

        # Strategy 1: BM25 Only
        t0 = time.time()
        res_bm25 = bm25.search(text, top_k=30)
        t_bm25 = (time.time() - t0) * 1000
        ret_bm25 = [doc["facet_id"] for doc, _ in res_bm25]
        micro_timing["bm25_latency_ms"].append(t_bm25)

        # Strategy 2: Dense Only
        t0 = time.time()
        res_dense = dense.search(text, top_k=30)
        t_dense = (time.time() - t0) * 1000
        ret_dense = [doc["facet_id"] for doc, _ in res_dense]
        micro_timing["dense_latency_ms"].append(t_dense)

        # Strategy 3: Hybrid RRF Baseline
        t0 = time.time()
        res_hybrid = hybrid_baseline.retrieve_candidates(text, top_k=30, use_reranker=False)
        t_hybrid = (time.time() - t0) * 1000
        ret_hybrid = [doc["facet_id"] for doc in res_hybrid]
        micro_timing["rrf_fusion_latency_ms"].append(t_hybrid)

        # Strategy 4: Hybrid RRF + CrossEncoder (Pool N=30)
        t0 = time.time()
        cand_n30 = hybrid_baseline.retrieve_candidates(text, top_k=30, use_reranker=False)
        t_ce_start = time.time()
        res_ce30 = reranker.rerank(text, cand_n30, top_k=30)
        t_ce30_inf = (time.time() - t_ce_start) * 1000
        t_ce30_total = (time.time() - t0) * 1000
        ret_ce30 = [doc["facet_id"] for doc in res_ce30]
        micro_timing["crossencoder_inference_ms"].append(t_ce30_inf)

        # Strategy 5: Hybrid RRF + CrossEncoder (Pool N=50)
        t0 = time.time()
        cand_n50 = hybrid_baseline.retrieve_candidates(text, top_k=50, use_reranker=False)
        res_ce50 = reranker.rerank(text, cand_n50, top_k=30)
        t_ce50_total = (time.time() - t0) * 1000
        ret_ce50 = [doc["facet_id"] for doc in res_ce50]

        # Log metrics
        for s_name, ret_list, t_ms in [
            ("bm25_only", ret_bm25, t_bm25),
            ("dense_only", ret_dense, t_dense),
            ("hybrid_rrf_baseline", ret_hybrid, t_hybrid),
            ("hybrid_rrf_crossencoder_n30", ret_ce30, t_ce30_total),
            ("hybrid_rrf_crossencoder_n50", ret_ce50, t_ce50_total)
        ]:
            r5 = len(set(ret_list[:5]).intersection(relevant_fids)) / len(relevant_fids)
            r10 = len(set(ret_list[:10]).intersection(relevant_fids)) / len(relevant_fids)
            r30 = len(set(ret_list[:30]).intersection(relevant_fids)) / len(relevant_fids)
            mrr_val = calculate_mrr(ret_list, relevant_fids)

            metrics[s_name]["recall_at_5"].append(r5)
            metrics[s_name]["recall_at_10"].append(r10)
            metrics[s_name]["recall_at_30"].append(r30)
            metrics[s_name]["mrr"].append(mrr_val)
            metrics[s_name]["latencies_ms"].append(t_ms)

        # Build Per-Case Analysis Table
        target_fid = list(relevant_fids)[0]
        base_rank = ret_hybrid.index(target_fid) + 1 if target_fid in ret_hybrid else None
        ce_rank = ret_ce30.index(target_fid) + 1 if target_fid in ret_ce30 else None
        
        improved = False
        if base_rank is not None and ce_rank is not None:
            improved = ce_rank < base_rank
        elif base_rank is None and ce_rank is not None:
            improved = True

        per_case_comparison.append({
            "conversation_id": conv_id,
            "query": text[:60] + "...",
            "target_facet_id": target_fid,
            "baseline_rank": base_rank if base_rank is not None else ">30",
            "reranker_rank": ce_rank if ce_rank is not None else ">30",
            "baseline_retrieved": base_rank is not None,
            "reranker_retrieved": ce_rank is not None,
            "improved": improved,
            "notes": "Reranked higher" if improved else ("Unchanged" if base_rank == ce_rank else "Reranked lower")
        })

    summary = {
        "cross_encoder_load_ms": cross_encoder_load_ms,
        "micro_timing": {
            "avg_bm25_latency_ms": round(float(np.mean(micro_timing["bm25_latency_ms"])), 2),
            "avg_dense_latency_ms": round(float(np.mean(micro_timing["dense_latency_ms"])), 2),
            "avg_rrf_fusion_ms": round(float(np.mean(micro_timing["rrf_fusion_latency_ms"])), 2),
            "avg_crossencoder_inference_ms": round(float(np.mean(micro_timing["crossencoder_inference_ms"])), 2),
        },
        "strategies": {}
    }

    for s_name in strategies:
        summary["strategies"][s_name] = {
            "recall_at_5_pct": round(float(np.mean(metrics[s_name]["recall_at_5"])) * 100, 2),
            "recall_at_10_pct": round(float(np.mean(metrics[s_name]["recall_at_10"])) * 100, 2),
            "recall_at_30_pct": round(float(np.mean(metrics[s_name]["recall_at_30"])) * 100, 2),
            "mrr": round(float(np.mean(metrics[s_name]["mrr"])), 4),
            "avg_latency_ms": round(float(np.mean(metrics[s_name]["latencies_ms"])), 2),
            "p95_latency_ms": round(float(np.percentile(metrics[s_name]["latencies_ms"], 95)), 2)
        }

    # Save JSON Report
    report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Build Error Table Markdown manually without requiring external tabulate dependency
    headers = ["conversation_id", "query", "target_facet_id", "baseline_rank", "reranker_rank", "baseline_retrieved", "reranker_retrieved", "improved", "notes"]
    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "| " + " | ".join([":---"] * len(headers)) + " |"
    data_rows = ["| " + " | ".join([str(item.get(h, "")) for h in headers]) + " |" for item in per_case_comparison]
    case_table_md = "\n".join([header_row, sep_row] + data_rows)

    md_content = f"""# Cross-Encoder Reranker Experimental Ablation Report

This report presents an empirical evaluation of adding a **Cross-Encoder Reranker** (`cross-encoder/ms-marco-MiniLM-L6-v2`) to the **Hybrid BM25 + Dense RRF Retrieval Pipeline**.

---

## 📊 Quantitative Retrieval Performance Comparison

| Configuration | Recall@5 | Recall@10 | Recall@30 | MRR | Avg Latency | P95 Latency | Overhead vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. BM25 Only** | `{summary['strategies']['bm25_only']['recall_at_5_pct']}%` | `{summary['strategies']['bm25_only']['recall_at_10_pct']}%` | `{summary['strategies']['bm25_only']['recall_at_30_pct']}%` | `{summary['strategies']['bm25_only']['mrr']}` | `{summary['strategies']['bm25_only']['avg_latency_ms']} ms` | `{summary['strategies']['bm25_only']['p95_latency_ms']} ms` | `-` |
| **B. Dense Only (`all-MiniLM-L6-v2`)** | `{summary['strategies']['dense_only']['recall_at_5_pct']}%` | `{summary['strategies']['dense_only']['recall_at_10_pct']}%` | `{summary['strategies']['dense_only']['recall_at_30_pct']}%` | `{summary['strategies']['dense_only']['mrr']}` | `{summary['strategies']['dense_only']['avg_latency_ms']} ms` | `{summary['strategies']['dense_only']['p95_latency_ms']} ms` | `-` |
| **C. BM25 + Dense RRF Baseline** | `{summary['strategies']['hybrid_rrf_baseline']['recall_at_5_pct']}%` | `{summary['strategies']['hybrid_rrf_baseline']['recall_at_10_pct']}%` | `{summary['strategies']['hybrid_rrf_baseline']['recall_at_30_pct']}%` | `{summary['strategies']['hybrid_rrf_baseline']['mrr']}` | `{summary['strategies']['hybrid_rrf_baseline']['avg_latency_ms']} ms` | `{summary['strategies']['hybrid_rrf_baseline']['p95_latency_ms']} ms` | **Baseline (0.0 ms)** |
| **D. Hybrid RRF + CrossEncoder (Pool N=30)** | `{summary['strategies']['hybrid_rrf_crossencoder_n30']['recall_at_5_pct']}%` | `{summary['strategies']['hybrid_rrf_crossencoder_n30']['recall_at_10_pct']}%` | `{summary['strategies']['hybrid_rrf_crossencoder_n30']['recall_at_30_pct']}%` | `{summary['strategies']['hybrid_rrf_crossencoder_n30']['mrr']}` | `{summary['strategies']['hybrid_rrf_crossencoder_n30']['avg_latency_ms']} ms` | `{summary['strategies']['hybrid_rrf_crossencoder_n30']['p95_latency_ms']} ms` | `+{(summary['strategies']['hybrid_rrf_crossencoder_n30']['avg_latency_ms'] - summary['strategies']['hybrid_rrf_baseline']['avg_latency_ms']):.2f} ms` |
| **E. Hybrid RRF + CrossEncoder (Pool N=50)** | `{summary['strategies']['hybrid_rrf_crossencoder_n50']['recall_at_5_pct']}%` | `{summary['strategies']['hybrid_rrf_crossencoder_n50']['recall_at_10_pct']}%` | `{summary['strategies']['hybrid_rrf_crossencoder_n50']['recall_at_30_pct']}%` | `{summary['strategies']['hybrid_rrf_crossencoder_n50']['mrr']}` | `{summary['strategies']['hybrid_rrf_crossencoder_n50']['avg_latency_ms']} ms` | `{summary['strategies']['hybrid_rrf_crossencoder_n50']['p95_latency_ms']} ms` | `+{(summary['strategies']['hybrid_rrf_crossencoder_n50']['avg_latency_ms'] - summary['strategies']['hybrid_rrf_baseline']['avg_latency_ms']):.2f} ms` |

---

## ⏱️ Sub-Component Latency Breakdown

- **Cross-Encoder Model Load Time (One-time startup)**: `{cross_encoder_load_ms} ms`
- **BM25 Search Latency**: `{summary['micro_timing']['avg_bm25_latency_ms']} ms`
- **Dense Embedding Search Latency**: `{summary['micro_timing']['avg_dense_latency_ms']} ms`
- **RRF Rank Fusion Latency**: `{summary['micro_timing']['avg_rrf_fusion_ms']} ms`
- **Cross-Encoder Candidate Inference Latency (Batch N=30)**: `{summary['micro_timing']['avg_crossencoder_inference_ms']} ms`

---

## 📋 Per-Case Benchmark Ranking Error Table

{case_table_md}
"""

    with open(report_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Successfully saved retrieval ablation JSON to: {report_json}")
    print(f"Successfully saved retrieval ablation MD to: {report_md}")


if __name__ == "__main__":
    run_retrieval_ablation()
