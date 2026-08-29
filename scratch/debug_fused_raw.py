"""
Fetch raw Qwen output for TEST_001 under Tiered Fusion (K <= 20 candidates).
"""

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scratch.debug_qwen_cases import (
    query_qwen_colab,
    normalize_facet_name,
    BM25Indexer,
    DenseVectorIndexer,
    extract_json_results
)

def debug_fused():
    catalog_path = PROJECT_ROOT / "data" / "processed" / "enriched_facets.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog_docs = json.load(f)

    observable_docs = [d for d in catalog_docs if d.get("conversation_observable", True) is True]
    bm25 = BM25Indexer().fit(observable_docs)
    dense = DenseVectorIndexer().fit(observable_docs)

    text = "I signed up for a solo skydiving trip even though I had never tried it before."
    norm_facet = "Risktaking"

    res_m1 = bm25.search(text, top_k=10)
    cands_m1 = [d for d, _ in res_m1]

    res_m2 = dense.search(text, top_k=10)
    cands_m2 = [d for d, _ in res_m2]

    fused_map = {}
    for d in cands_m1 + cands_m2:
        fid = d["facet_id"]
        if fid not in fused_map:
            fused_map[fid] = d
    fused_cands = list(fused_map.values())[:20]

    print(f"Fused candidate count: {len(fused_cands)}")
    print("Calling Qwen GPU...")

    parsed_results, raw_output, lat_ms, parse_state, prompt_sent = query_qwen_colab(text, fused_cands)

    print(f"\n--- RAW UN-TRUNCATED QWEN OUTPUT ({round(lat_ms)}ms) ---")
    print(repr(raw_output))
    print("--- END RAW OUTPUT ---\n")

    print(f"Parse State: '{parse_state}', item_count={len(parsed_results)}")
    print(f"Parsed Items: {json.dumps(parsed_results, indent=2)}")

if __name__ == "__main__":
    debug_fused()
