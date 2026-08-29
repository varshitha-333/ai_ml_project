"""
Strict 30-Case Retrieval A/B Experiment Runner.
Evaluates Baseline vs BGE-M3 vs Hybrid BGE-M3 vs BGE-M3 + Reranker.
Saves all experimental outputs under outputs/experiments/.
"""

import sys
from pathlib import Path
import json
import time
import pickle
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Production Baseline Imports (Used READ-ONLY, NOT MUTATED)
from src.retrieval.bm25 import BM25Indexer
from src.retrieval.indexer import DenseVectorIndexer
from src.retrieval.search import HybridFacetRetriever
from src.retrieval.reranker import CrossEncoderReranker

# Try loading BGE-M3 via sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def normalize_facet_name(name: str) -> str:
    """Normalizes facet name by stripping trailing colons and extra whitespace."""
    if not name:
        return ""
    cleaned = name.strip()
    if cleaned.endswith(":"):
        cleaned = cleaned[:-1].strip()
    return cleaned


def build_rich_facet_text(doc: dict) -> str:
    """Builds identical rich text representation for all semantic models."""
    name = doc.get("normalized_facet", doc.get("raw_facet", ""))
    raw = doc.get("raw_facet", "")
    ftype = doc.get("facet_type", "conversational_trait")
    definition = doc.get("definition", "")
    scoring_rules = doc.get("scoring_definition", "")
    keywords = ", ".join(doc.get("keywords", []))
    abstain_reason = doc.get("abstention_reason", "")

    return (
        f"Facet Name: {name} | Raw Facet: {raw} | Type: {ftype} | "
        f"Definition: {definition} | Scoring Rules: {scoring_rules} | "
        f"Keywords: {keywords} | Abstention Reason: {abstain_reason}"
    )


def calculate_metrics(target_ranks: list):
    """Calculates Recall@K, MRR, and Hit Rates."""
    N = len(target_ranks)
    if N == 0:
        return {}

    r1 = sum(1 for r in target_ranks if r == 1) / N
    r3 = sum(1 for r in target_ranks if 1 <= r <= 3) / N
    r5 = sum(1 for r in target_ranks if 1 <= r <= 5) / N
    r10 = sum(1 for r in target_ranks if 1 <= r <= 10) / N
    r20 = sum(1 for r in target_ranks if 1 <= r <= 20) / N
    r30 = sum(1 for r in target_ranks if 1 <= r <= 30) / N

    mrr = sum(1.0 / r for r in target_ranks if r <= 30) / N
    hit10 = sum(1 for r in target_ranks if r <= 10) / N
    hit30 = sum(1 for r in target_ranks if r <= 30) / N

    return {
        "recall_at_1": round(r1 * 100, 2),
        "recall_at_3": round(r3 * 100, 2),
        "recall_at_5": round(r5 * 100, 2),
        "recall_at_10": round(r10 * 100, 2),
        "recall_at_20": round(r20 * 100, 2),
        "recall_at_30": round(r30 * 100, 2),
        "mrr": round(mrr, 4),
        "hit_rate_10": round(hit10 * 100, 2),
        "hit_rate_30": round(hit30 * 100, 2)
    }


class BGEM3VectorIndexer:
    """Experimental BGE-M3 Indexer (Cached separately under outputs/experiments/bge_m3/)."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self.model = None
        self.documents = []
        self.embeddings = None
        self.cache_dir = PROJECT_ROOT / "outputs" / "experiments" / "bge_m3"
        self.cache_path = self.cache_dir / "bge_m3_embeddings.pkl"

    def fit(self, documents: list):
        self.documents = documents
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.cache_path.exists():
            print(f"[BGE-M3] Loading precomputed embeddings from experiment cache: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                cached_data = pickle.load(f)
                self.embeddings = cached_data["embeddings"]
                return self

        print(f"[BGE-M3] Initializing model '{self.model_name}'...")
        self.model = SentenceTransformer(self.model_name)
        texts = [build_rich_facet_text(doc) for doc in documents]
        print(f"[BGE-M3] Generating embeddings for {len(texts)} facets...")
        self.embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        with open(self.cache_path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "model_name": self.model_name}, f)
        print(f"[BGE-M3] Saved experiment cache to {self.cache_path}")
        return self

    def search(self, query: str, top_k: int = 30):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        q_emb = self.model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]
        sims = np.dot(self.embeddings, q_emb)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.documents[idx], float(sims[idx])) for idx in top_indices]


def run_experiment():
    exp_dir = PROJECT_ROOT / "outputs" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = PROJECT_ROOT / "data" / "processed" / "enriched_facets.json"
    csv_path = PROJECT_ROOT / "facet_evaluation_test_set_50.csv"

    with open(catalog_path, "r", encoding="utf-8") as f:
        all_docs = json.load(f)

    # Observable facets filter
    observable_docs = [d for d in all_docs if d.get("conversation_observable", True) is True]

    doc_by_id = {d["facet_id"]: d for d in all_docs}
    doc_by_norm = {normalize_facet_name(d["normalized_facet"]).lower(): d for d in all_docs}
    doc_by_raw = {normalize_facet_name(d["raw_facet"]).lower(): d for d in all_docs}

    # Select 30 representative test cases
    df_50 = pd.read_csv(csv_path).iloc[:30]
    cases = []

    for idx, row in df_50.iterrows():
        case_id = str(row.get("test_id", f"CASE_{idx+1:03d}"))
        text = str(row["text"]).strip()
        orig_facet = str(row["expected_facet"]).strip()
        norm_facet = normalize_facet_name(orig_facet)
        status = str(row["expected_status"]).strip()

        matched = doc_by_norm.get(norm_facet.lower()) or doc_by_raw.get(norm_facet.lower())
        cat_fid = matched["facet_id"] if matched else "MISSING"

        cases.append({
            "case_id": case_id,
            "input_text": text,
            "original_target_facet": orig_facet,
            "normalized_target_facet": norm_facet,
            "expected_status": status,
            "target_facet_exists_in_catalog": matched is not None,
            "catalog_facet_id": cat_fid
        })

    # Save Phase 1 Dataset JSON
    dataset_json_path = exp_dir / "retrieval_30_cases.json"
    with open(dataset_json_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    # Generate Phase 1 Audit Report
    audit_md_path = exp_dir / "retrieval_30_case_audit.md"
    audit_md = "# PHASE 1 — 30-Case Retrieval Audit Report\n\n"
    audit_md += "| Case ID | Input Text Snippet | Original Target | Normalized Target | Exists in Catalog | Catalog Facet ID |\n"
    audit_md += "| :--- | :--- | :--- | :--- | :---: | :--- |\n"
    for c in cases:
        exists_str = "YES" if c["target_facet_exists_in_catalog"] else "NO"
        snip = c["input_text"][:50] + "..." if len(c["input_text"]) > 50 else c["input_text"]
        audit_md += f"| `{c['case_id']}` | `{snip}` | `{c['original_target_facet']}` | `{c['normalized_target_facet']}` | `{exists_str}` | `{c['catalog_facet_id']}` |\n"

    with open(audit_md_path, "w", encoding="utf-8") as f:
        f.write(audit_md)

    print(f"[Phase 1] Dataset saved to {dataset_json_path}")
    print(f"[Phase 1] Audit report saved to {audit_md_path}")

    # ==========================================
    # PHASE 2 — CURRENT BASELINE RETRIEVER
    # ==========================================
    print("\n--- Running Phase 2: Current Production Baseline ---")
    baseline_retriever = HybridFacetRetriever(default_top_k=30, reranker_enabled=False).fit(all_docs)

    baseline_results = []
    baseline_ranks = []
    baseline_latencies = []

    for c in cases:
        t0 = time.time()
        res_docs = baseline_retriever.retrieve_candidates(c["input_text"], top_k=30)
        t_ms = (time.time() - t0) * 1000
        baseline_latencies.append(t_ms)

        retrieved_ids = [d["facet_id"] for d in res_docs]
        retrieved_names = [d["normalized_facet"] for d in res_docs]
        target_fid = c["catalog_facet_id"]

        rank = 999
        if target_fid != "MISSING" and target_fid in retrieved_ids:
            rank = retrieved_ids.index(target_fid) + 1

        baseline_ranks.append(rank)
        baseline_results.append({
            "case_id": c["case_id"],
            "target_facet": c["normalized_target_facet"],
            "catalog_facet_id": target_fid,
            "rank": rank,
            "hit_at_10": rank <= 10,
            "hit_at_30": rank <= 30,
            "top_10_candidates": retrieved_names[:10],
            "top_30_candidates": retrieved_names[:30],
            "latency_ms": round(t_ms, 2)
        })

    baseline_metrics = calculate_metrics(baseline_ranks)
    baseline_metrics["avg_latency_ms"] = round(float(np.mean(baseline_latencies)), 2)
    baseline_metrics["p95_latency_ms"] = round(float(np.percentile(baseline_latencies, 95)), 2)

    with open(exp_dir / "current_30_results.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": baseline_metrics, "cases": baseline_results}, f, indent=2)

    # ==========================================
    # PHASE 3 — BGE-M3 DENSE ONLY
    # ==========================================
    print("\n--- Running Phase 3: BGE-M3 Dense Only ---")
    bge_indexer = BGEM3VectorIndexer().fit(observable_docs)

    bge_results = []
    bge_ranks = []
    bge_latencies = []

    for c in cases:
        t0 = time.time()
        res = bge_indexer.search(c["input_text"], top_k=30)
        t_ms = (time.time() - t0) * 1000
        bge_latencies.append(t_ms)

        res_docs = [doc for doc, _ in res]
        retrieved_ids = [d["facet_id"] for d in res_docs]
        retrieved_names = [d["normalized_facet"] for d in res_docs]
        target_fid = c["catalog_facet_id"]

        rank = 999
        if target_fid != "MISSING" and target_fid in retrieved_ids:
            rank = retrieved_ids.index(target_fid) + 1

        bge_ranks.append(rank)
        bge_results.append({
            "case_id": c["case_id"],
            "target_facet": c["normalized_target_facet"],
            "catalog_facet_id": target_fid,
            "rank": rank,
            "hit_at_10": rank <= 10,
            "hit_at_30": rank <= 30,
            "top_10_candidates": retrieved_names[:10],
            "top_30_candidates": retrieved_names[:30],
            "latency_ms": round(t_ms, 2)
        })

    bge_metrics = calculate_metrics(bge_ranks)
    bge_metrics["avg_latency_ms"] = round(float(np.mean(bge_latencies)), 2)
    bge_metrics["p95_latency_ms"] = round(float(np.percentile(bge_latencies, 95)), 2)

    with open(exp_dir / "bge_m3_30_results.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": bge_metrics, "cases": bge_results}, f, indent=2)

    # ==========================================
    # PHASE 4 — HYBRID BM25 + BGE-M3 RRF
    # ==========================================
    print("\n--- Running Phase 4: Hybrid BM25 + BGE-M3 RRF ---")
    bm25_indexer = BM25Indexer().fit(observable_docs)

    hybrid_results = []
    hybrid_ranks = []
    hybrid_latencies = []

    for c in cases:
        t0 = time.time()
        res_bm25 = bm25_indexer.search(c["input_text"], top_k=50)
        res_bge = bge_indexer.search(c["input_text"], top_k=50)
        
        rrf_scores = {}
        doc_map = {}

        for r, (d, _) in enumerate(res_bm25, 1):
            fid = d["facet_id"]
            doc_map[fid] = d
            rrf_scores[fid] = rrf_scores.get(fid, 0.0) + (1.0 / (60.0 + r))

        for r, (d, _) in enumerate(res_bge, 1):
            fid = d["facet_id"]
            doc_map[fid] = d
            rrf_scores[fid] = rrf_scores.get(fid, 0.0) + (1.0 / (60.0 + r))

        sorted_fids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        t_ms = (time.time() - t0) * 1000
        hybrid_latencies.append(t_ms)

        retrieved_docs = [doc_map[fid] for fid in sorted_fids[:30]]
        retrieved_ids = [d["facet_id"] for d in retrieved_docs]
        retrieved_names = [d["normalized_facet"] for d in retrieved_docs]
        target_fid = c["catalog_facet_id"]

        rank = 999
        if target_fid != "MISSING" and target_fid in retrieved_ids:
            rank = retrieved_ids.index(target_fid) + 1

        hybrid_ranks.append(rank)
        hybrid_results.append({
            "case_id": c["case_id"],
            "target_facet": c["normalized_target_facet"],
            "catalog_facet_id": target_fid,
            "rank": rank,
            "hit_at_10": rank <= 10,
            "hit_at_30": rank <= 30,
            "top_10_candidates": retrieved_names[:10],
            "top_30_candidates": retrieved_names[:30],
            "latency_ms": round(t_ms, 2)
        })

    hybrid_metrics = calculate_metrics(hybrid_ranks)
    hybrid_metrics["avg_latency_ms"] = round(float(np.mean(hybrid_latencies)), 2)
    hybrid_metrics["p95_latency_ms"] = round(float(np.percentile(hybrid_latencies, 95)), 2)

    with open(exp_dir / "hybrid_bge_m3_30_results.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": hybrid_metrics, "cases": hybrid_results}, f, indent=2)

    # ==========================================
    # PHASE 5 — OPTIONAL BGE-M3 + RERANKER
    # ==========================================
    print("\n--- Running Phase 5: Hybrid BGE-M3 + Cross-Encoder Reranker ---")
    reranker = CrossEncoderReranker()

    rerank_results = []
    rerank_ranks = []
    rerank_latencies = []

    for idx, c in enumerate(cases):
        t0 = time.time()
        res_bm25 = bm25_indexer.search(c["input_text"], top_k=50)
        res_bge = bge_indexer.search(c["input_text"], top_k=50)
        
        rrf_scores = {}
        doc_map = {}

        for r, (d, _) in enumerate(res_bm25, 1):
            fid = d["facet_id"]
            doc_map[fid] = d
            rrf_scores[fid] = rrf_scores.get(fid, 0.0) + (1.0 / (60.0 + r))

        for r, (d, _) in enumerate(res_bge, 1):
            fid = d["facet_id"]
            doc_map[fid] = d
            rrf_scores[fid] = rrf_scores.get(fid, 0.0) + (1.0 / (60.0 + r))

        sorted_fids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        hybrid_cand_docs = [doc_map[fid] for fid in sorted_fids[:30]]
        
        res_ce = reranker.rerank(c["input_text"], hybrid_cand_docs, top_k=30)
        t_ms = (time.time() - t0) * 1000
        rerank_latencies.append(t_ms)

        retrieved_ids = [d["facet_id"] for d in res_ce]
        retrieved_names = [d["normalized_facet"] for d in res_ce]
        target_fid = c["catalog_facet_id"]

        rank = 999
        if target_fid != "MISSING" and target_fid in retrieved_ids:
            rank = retrieved_ids.index(target_fid) + 1

        rerank_ranks.append(rank)
        rerank_results.append({
            "case_id": c["case_id"],
            "target_facet": c["normalized_target_facet"],
            "catalog_facet_id": target_fid,
            "rank": rank,
            "hit_at_10": rank <= 10,
            "hit_at_30": rank <= 30,
            "top_10_candidates": retrieved_names[:10],
            "top_30_candidates": retrieved_names[:30],
            "latency_ms": round(t_ms, 2)
        })

    rerank_metrics = calculate_metrics(rerank_ranks)
    rerank_metrics["avg_latency_ms"] = round(float(np.mean(rerank_latencies)), 2)
    rerank_metrics["p95_latency_ms"] = round(float(np.percentile(rerank_latencies, 95)), 2)

    with open(exp_dir / "bge_m3_reranker_30_results.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": rerank_metrics, "cases": rerank_results}, f, indent=2)

    # ==========================================
    # PHASE 6 — DIRECT CASE-BY-CASE COMPARISON
    # ==========================================
    comp_md_path = exp_dir / "retrieval_30_comparison.md"
    comp_md = f"""# PHASE 6 — Direct Case-by-Case Retrieval Comparison (30 Cases)

---

## 📊 Detailed Case-by-Case Rank Matrix

| Case ID | Target Facet | Current Baseline Rank | BGE-M3 Dense Rank | Hybrid BGE-M3 Rank | Reranked Rank |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for i in range(len(cases)):
        c = cases[i]
        r_base = ">30" if baseline_ranks[i] > 30 else f"#{baseline_ranks[i]}"
        r_bge = ">30" if bge_ranks[i] > 30 else f"#{bge_ranks[i]}"
        r_hyb = ">30" if hybrid_ranks[i] > 30 else f"#{hybrid_ranks[i]}"
        r_rerank = ">30" if rerank_ranks[i] > 30 else f"#{rerank_ranks[i]}"

        comp_md += f"| `{c['case_id']}` | `{c['normalized_target_facet']}` | `{r_base}` | `{r_bge}` | `{r_hyb}` | `{r_rerank}` |\n"

    comp_md += f"""
---

## 🔍 Specific Evaluation of Known Baseline Failures

1. **Hesitation** (*"I tend to pause for a long time before making an important decision."*):
   - Baseline Rank: `#{baseline_ranks[3]}`
   - BGE-M3 Rank: `#{bge_ranks[3]}`
   - Hybrid BGE-M3 Rank: `#{hybrid_ranks[3]}`
   - Verdict: {"FIXED into Top 10!" if hybrid_ranks[3] <= 10 else "STILL OUTSIDE Top 10"}

2. **Merriness** (*"That was hilarious—I couldn't stop laughing for ten minutes."*):
   - Baseline Rank: `#{baseline_ranks[6]}`
   - BGE-M3 Rank: `#{bge_ranks[6]}`
   - Hybrid BGE-M3 Rank: `#{hybrid_ranks[6]}`
   - Verdict: {"FIXED into Top 10!" if hybrid_ranks[6] <= 10 else "STILL OUTSIDE Top 10"}

3. **Emotionalism** (*"I got so emotional during the movie that I had to take a break."*):
   - Baseline Rank: `#{baseline_ranks[7]}`
   - BGE-M3 Rank: `#{bge_ranks[7]}`
   - Hybrid BGE-M3 Rank: `#{hybrid_ranks[7]}`
   - Verdict: {"FIXED into Top 10!" if hybrid_ranks[7] <= 10 else "STILL OUTSIDE Top 10"}

4. **Self-improvement** (*"I started a course because I want to improve my Python skills."*):
   - Baseline Rank: `#{baseline_ranks[8]}`
   - BGE-M3 Rank: `#{bge_ranks[8]}`
   - Hybrid BGE-M3 Rank: `#{hybrid_ranks[8]}`
   - Verdict: {"FIXED into Top 10!" if hybrid_ranks[8] <= 10 else "STILL OUTSIDE Top 10"}

5. **Democratic Leadership** (*"We voted together and chose the option supported by most of the team."*):
   - Baseline Rank: `#{baseline_ranks[2]}`
   - BGE-M3 Rank: `#{bge_ranks[2]}`
   - Hybrid BGE-M3 Rank: `#{hybrid_ranks[2]}`
   - Verdict: {"FIXED into Top 10!" if hybrid_ranks[2] <= 10 else "STILL OUTSIDE Top 10"}

---

## 🏆 Final Summary Decision Table

| System | Recall@10 | Recall@30 | MRR | Avg Latency | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Current Baseline** | `{baseline_metrics['recall_at_10']}%` | `{baseline_metrics['recall_at_30']}%` | `{baseline_metrics['mrr']}` | `{baseline_metrics['avg_latency_ms']} ms` | `{baseline_metrics['p95_latency_ms']} ms` |
| **BGE-M3 Dense Only** | `{bge_metrics['recall_at_10']}%` | `{bge_metrics['recall_at_30']}%` | `{bge_metrics['mrr']}` | `{bge_metrics['avg_latency_ms']} ms` | `{bge_metrics['p95_latency_ms']} ms` |
| **Hybrid BM25 + BGE-M3 RRF** | `{hybrid_metrics['recall_at_10']}%` | `{hybrid_metrics['recall_at_30']}%` | `{hybrid_metrics['mrr']}` | `{hybrid_metrics['avg_latency_ms']} ms` | `{hybrid_metrics['p95_latency_ms']} ms` |
| **BGE-M3 + Reranker** | `{rerank_metrics['recall_at_10']}%` | `{rerank_metrics['recall_at_30']}%` | `{rerank_metrics['mrr']}` | `{rerank_metrics['avg_latency_ms']} ms` | `{rerank_metrics['p95_latency_ms']} ms` |
"""

    with open(comp_md_path, "w", encoding="utf-8") as f:
        f.write(comp_md)

    print(f"\nSuccessfully generated Phase 6 Comparison Report: {comp_md_path}")

    # Print Blunt Conclusion to Console
    print("\n" + "=" * 70)
    print("FINAL BLUNT CONCLUSION SUMMARY")
    print("=" * 70)
    print(f"Current Baseline Recall@10: {baseline_metrics['recall_at_10']}% | MRR: {baseline_metrics['mrr']}")
    print(f"Hybrid BGE-M3 Recall@10:    {hybrid_metrics['recall_at_10']}% | MRR: {hybrid_metrics['mrr']}")
    print(f"BGE-M3 + Reranker R@10:     {rerank_metrics['recall_at_10']}% | MRR: {rerank_metrics['mrr']}")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
