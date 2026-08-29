"""
Hybrid Retrieval Engine with Reciprocal Rank Fusion (RRF) and Observable Pre-Filtering.

Combines BM25 lexical ranking and Dense Vector semantic ranking, filters unobservable
facets, applies similarity thresholding, and returns structured candidates.
"""

from typing import List, Dict, Any, Tuple, Optional
from src.retrieval.bm25 import BM25Indexer
from src.retrieval.indexer import DenseVectorIndexer
from src.retrieval.reranker import CrossEncoderReranker


class HybridFacetRetriever:
    """
    Hybrid Lexical + Dense Vector Search with Reciprocal Rank Fusion (RRF) and optional Cross-Encoder Reranking.
    """

    def __init__(
        self,
        rrf_k: float = 60.0,
        similarity_threshold: float = 0.15,
        default_top_k: int = 10,
        reranker_enabled: bool = False,
        rerank_initial_pool_size: int = 30
    ):
        self.rrf_k = rrf_k
        self.similarity_threshold = similarity_threshold
        self.default_top_k = default_top_k
        self.reranker_enabled = reranker_enabled
        self.rerank_initial_pool_size = rerank_initial_pool_size
        self.bm25_indexer = BM25Indexer()
        self.dense_indexer = DenseVectorIndexer()
        self.reranker = CrossEncoderReranker() if reranker_enabled else None
        self.documents: List[Dict[str, Any]] = []

    def fit(self, documents: List[Dict[str, Any]]) -> "HybridFacetRetriever":
        """
        Pre-filters observable facets and builds both BM25 and Dense Vector indices.
        """
        # Strictly filter to conversation-observable facets
        observable_docs = [
            doc for doc in documents
            if doc.get("conversation_observable", True) is True
        ]
        self.documents = observable_docs

        if observable_docs:
            self.bm25_indexer.fit(observable_docs)
            self.dense_indexer.fit(observable_docs)

        return self

    def retrieve_candidates(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        min_score_threshold: Optional[float] = None,
        use_reranker: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid BM25 + Dense retrieval using Reciprocal Rank Fusion,
        optionally reranking candidates with a Cross-Encoder.
        """
        k_val = top_k if top_k is not None else self.default_top_k
        enable_rerank = use_reranker if use_reranker is not None else self.reranker_enabled

        if not self.documents or not query_text.strip():
            return []

        fetch_k = max(k_val, self.rerank_initial_pool_size) if enable_rerank else k_val

        # 1. Fetch BM25 and Dense vector search results
        bm25_results = self.bm25_indexer.search(query_text, top_k=len(self.documents))
        dense_results = self.dense_indexer.search(query_text, top_k=len(self.documents))

        bm25_score_map: Dict[str, float] = {}
        dense_score_map: Dict[str, float] = {}
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        for rank, (doc, bm25_score) in enumerate(bm25_results):
            facet_id = doc["facet_id"]
            doc_map[facet_id] = doc
            bm25_score_map[facet_id] = bm25_score
            rrf_scores[facet_id] = rrf_scores.get(facet_id, 0.0) + (1.0 / (self.rrf_k + (rank + 1)))

        for rank, (doc, dense_score) in enumerate(dense_results):
            facet_id = doc["facet_id"]
            doc_map[facet_id] = doc
            dense_score_map[facet_id] = dense_score
            rrf_scores[facet_id] = rrf_scores.get(facet_id, 0.0) + (1.0 / (self.rrf_k + (rank + 1)))

        # 2. Rank facets by RRF score descending
        sorted_facet_ids = sorted(rrf_scores.keys(), key=lambda fid: rrf_scores[fid], reverse=True)

        # 3. Retrieve top candidates pool
        candidates = []
        for fid in sorted_facet_ids:
            doc = doc_map[fid]
            d_score = dense_score_map.get(fid, 0.0)
            b_score = bm25_score_map.get(fid, 0.0)

            if min_score_threshold is not None and d_score < min_score_threshold and b_score <= 0.0:
                continue

            doc_copy = dict(doc)
            doc_copy["rrf_score"] = rrf_scores[fid]
            doc_copy["retrieval_rrf_score"] = rrf_scores[fid]
            doc_copy["dense_score"] = d_score
            doc_copy["bm25_score"] = b_score
            candidates.append(doc_copy)

            if len(candidates) >= fetch_k:
                break

        # 4. Optional CrossEncoder Reranking
        if enable_rerank:
            if self.reranker is None:
                self.reranker = CrossEncoderReranker()
            candidates = self.reranker.rerank(query_text, candidates, top_k=k_val)
        else:
            candidates = candidates[:k_val]

        return candidates
