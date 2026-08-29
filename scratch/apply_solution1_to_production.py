"""
Script: Applies Solution 1 Multi-Example Conversational Utterance Indexing to production data/processed/enriched_facets.json
and recomputes production dense vector embedding cache.
"""

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.indexer import DenseVectorIndexer

def apply_solution1():
    prod_facets_path = PROJECT_ROOT / "data" / "processed" / "enriched_facets.json"
    sol1_facets_path = PROJECT_ROOT / "outputs" / "experiments" / "solution1_multi_example_facets.json"

    with open(prod_facets_path, "r", encoding="utf-8") as f:
        prod_docs = json.load(f)

    if sol1_facets_path.exists():
        with open(sol1_facets_path, "r", encoding="utf-8") as f:
            sol1_docs = json.load(f)
        
        sol1_examples_map = {d["facet_id"]: d.get("conversational_examples", d.get("examples", [])) for d in sol1_docs}
        
        for d in prod_docs:
            fid = d["facet_id"]
            if fid in sol1_examples_map:
                d["conversational_examples"] = sol1_examples_map[fid]

        with open(prod_facets_path, "w", encoding="utf-8") as f:
            json.dump(prod_docs, f, indent=2)
        print(f"[Solution 1] Merged conversational examples into production {prod_facets_path}")

    # Remove stale cache file to force embedding recomputation
    cache_path = PROJECT_ROOT / "data" / "processed" / "facet_embeddings_cache.npz"
    if cache_path.exists():
        cache_path.unlink()
        print(f"[Solution 1] Unlinked stale embedding cache -> {cache_path}")

    # Re-fit production DenseVectorIndexer to regenerate cache with Solution 1 examples
    observable_docs = [d for d in prod_docs if d.get("conversation_observable", True) is True]
    indexer = DenseVectorIndexer().fit(observable_docs)
    print(f"[Solution 1] Recomputed production dense vector embedding cache ({len(indexer.embeddings)} embeddings saved).")

if __name__ == "__main__":
    apply_solution1()
