"""
Experimental Facet Retriever Architecture supporting BM25, Dense, RRF, and optional Cross-Encoder Reranking.
Isolated implementation to evaluate retrieval strategies without mutating baseline retriever.
"""

from typing import List, Dict, Any, Optional
import time
import numpy as np

from src.retrieval.bm25 import BM25Indexer
from src.retrieval.indexer import DenseVectorIndexer
from src.retrieval.reranker import CrossEncoderReranker


class ExperimentalFacetRetriever:
    """
    Experimental multi-strategy retriever supporting BM25, Dense, RRF, and Cross-Encoder reranking.
    """

    def __init__(
        self,
        default_top_k: int = 10,
        reranker_enabled: bool = False,
        rrf_k: int = 60
    ):
        self.default_top_k = default_top_k
        self.reranker_enabled = reranker_enabled
        self.rrf_k = rrf_k
        
        self.documents: List[Dict[str, Any]] = []
        self.observable_documents: List[Dict[str, Any]] = []
        
        self.bm25_indexer = BM25Indexer()
        self.dense_indexer = DenseVectorIndexer()
        self.reranker: Optional[CrossEncoderReranker] = None
        
        if self.reranker_enabled:
            self.reranker = CrossEncoderReranker()

    def fit(self, documents: List[Dict[str, Any]]) -> "ExperimentalFacetRetriever":
        """
        Fits BM25 and Dense indexers strictly over observable facets.
        """
        self.documents = documents
        # Deterministic pre-filtering of unobservable facets (medical/hardware)
        self.observable_documents = [
            doc for doc in documents if doc.get("conversation_observable", True) is True
        ]
        
        self.bm25_indexer.fit(self.observable_documents)
        self.dense_indexer.fit(self.observable_documents)
        return self

    def retrieve_bm25_only(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        results = self.bm25_indexer.search(query, top_k=top_k)
        return [doc for doc, _ in results]

    def retrieve_dense_only(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        results = self.dense_indexer.search(query, top_k=top_k)
        return [doc for doc, _ in results]

    def retrieve_rrf(self, query: str, top_k: int = 10, candidate_pool_size: int = 50) -> List[Dict[str, Any]]:
        """
        Hybrid BM25 + Dense Reciprocal Rank Fusion (RRF).
        """
        bm25_results = self.bm25_indexer.search(query, top_k=candidate_pool_size)
        dense_results = self.dense_indexer.search(query, top_k=candidate_pool_size)
        
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        
        # Rank fusion BM25
        for rank, (doc, _) in enumerate(bm25_results, 1):
            fid = doc["facet_id"]
            doc_map[fid] = doc
            rrf_scores[fid] = rrf_scores.get(fid, 0.0) + (1.0 / (self.rrf_k + rank))
            
        # Rank fusion Dense
        for rank, (doc, _) in enumerate(dense_results, 1):
            fid = doc["facet_id"]
            doc_map[fid] = doc
            rrf_scores[fid] = rrf_scores.get(fid, 0.0) + (1.0 / (self.rrf_k + rank))
            
        # Sort candidates by RRF score descending
        sorted_fids = sorted(rrf_scores.keys(), key=lambda fid: rrf_scores[fid], reverse=True)
        return [doc_map[fid] for fid in sorted_fids[:top_k]]

    def retrieve_rrf_with_reranker(
        self,
        query: str,
        top_k: int = 10,
        candidate_pool_size: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Hybrid BM25 + Dense RRF followed by Cross-Encoder reranking.
        """
        candidates = self.retrieve_rrf(query, top_k=candidate_pool_size, candidate_pool_size=candidate_pool_size)
        if not self.reranker:
            self.reranker = CrossEncoderReranker()
            
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        return reranked
