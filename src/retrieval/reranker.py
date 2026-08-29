"""
Cross-Encoder Reranker using Sentence-Transformers (cross-encoder/ms-marco-MiniLM-L6-v2).
"""

from typing import List, Dict, Any, Optional
import os
import time


class CrossEncoderReranker:
    """
    Lightweight Cross-Encoder Reranker for Candidate Facets.
    Reranks candidate facet pool output by BM25/Dense/RRF.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.is_loaded = False

    def load_model(self) -> bool:
        if self.is_loaded and self.model is not None:
            return True
        try:
            from sentence_transformers import CrossEncoder

            print(f"[RERANKER] Loading CrossEncoder model: '{self.model_name}'...")
            self.model = CrossEncoder(self.model_name)
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"[RERANKER WARN] Failed to load CrossEncoder model '{self.model_name}': {e}")
            self.is_loaded = False
            return False

    def _prepare_document_text(self, facet: Dict[str, Any]) -> str:
        """
        Builds enriched text representation for Cross-Encoder scoring.
        Uses normalized_facet, raw_facet, facet_type, scoring_definition, keywords, abstention_reason.
        """
        name = facet.get("normalized_facet", "")
        raw_name = facet.get("raw_facet", "")
        ftype = facet.get("facet_type", "conversational_trait")
        desc = facet.get("scoring_definition", "")
        reason = facet.get("abstention_reason", "")
        keywords_raw = facet.get("keywords", [])
        keywords_str = " ".join(keywords_raw) if isinstance(keywords_raw, list) else str(keywords_raw)
        return f"Facet Name: {name} | Raw Facet: {raw_name} | Category: {ftype} | Definition: {desc} | Keywords: {keywords_str} | Note: {reason}".strip()

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate facet dictionaries using the CrossEncoder.

        Returns:
            List[Dict[str, Any]]: Sorted top_k candidate facets with reranker_score attached.
        """
        if not candidates or not query.strip():
            return candidates[:top_k]

        if not self.is_loaded:
            success = self.load_model()
            if not success or self.model is None:
                # Controlled safe fallback to initial candidate ordering
                return candidates[:top_k]

        # Build query-document pairs
        pairs = []
        for doc in candidates:
            doc_text = self._prepare_document_text(doc)
            pairs.append([query, doc_text])

        # Batch inference over candidates
        t0 = time.time()
        scores = self.model.predict(pairs)
        inference_time_ms = (time.time() - t0) * 1000

        # Attach reranker_score and sort
        reranked_candidates = []
        for i, doc in enumerate(candidates):
            doc_copy = dict(doc)
            score_val = float(scores[i])
            doc_copy["reranker_score"] = score_val
            doc_copy["reranker_latency_ms"] = inference_time_ms
            reranked_candidates.append(doc_copy)

        # Sort by reranker_score descending
        reranked_candidates.sort(key=lambda d: d["reranker_score"], reverse=True)

        return reranked_candidates[:top_k]
