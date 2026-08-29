# Phase 7 Architecture Specification: Enriched Facet + Semantic Query Expansion Retrieval

---

## 🏛️ System Pipeline Architecture

1. **Input Stage**: User Dialogue Transcript.
2. **Semantic Query Expansion Stage**: Deterministic keyword/synonym expansion extracting intent.
3. **BM25 Lexical Retrieval Stage**: Okapi BM25 over Phase 7 Enriched Facet Schema (Top 100 Pool).
4. **Dense Vector Retrieval Stage**: MiniLM embeddings over (A) Original Query and (B) Expanded Query (Top 100 Pool each).
5. **Hybrid Tri-Fusion Stage**: Reciprocal Rank Fusion (RRF, $k=60.0$) combining BM25 + Dense Original + Dense Expanded into a Top-100 Candidate Pool.
6. **Semantic Reranking Stage**: Cross-Encoder (`ms-marco-MiniLM-L6-v2`) reranking Top-100 pool into final Top-30 output candidates.

---

## 🛡️ Production Preservation Rules
- Production code (`src/retrieval/search.py`, `src/retrieval/bm25.py`, `src/retrieval/indexer.py`) remains 100% untouched.
- Production disk cache (`data/processed/facet_embeddings_cache.npz`) remains 100% untouched.
- All Phase 7 experimental artifacts are saved under `outputs/experiments/phase7/`.
