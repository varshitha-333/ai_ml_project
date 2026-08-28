"""
Top-Level End-to-End Facet Evaluator Pipeline.

Connects Conversation Input -> Hybrid Candidate Retrieval -> Observable Pre-Filtering
-> Batched LLM Scoring -> Pydantic Output Validation -> Structured Results.
"""

import time
import uuid
from typing import List, Dict, Any, Optional
from src.retrieval.pipeline import RetrievalPipeline
from src.scoring.scorer import BatchedFacetScorer
from src.scoring.schemas import FacetScoreResult, EvaluationPipelineResponse
from src.scoring.inference_backend import BaseInferenceBackend


class FacetEvaluatorPipeline:
    """
    End-to-end Evaluation Pipeline manager with structured timing diagnostics.
    """

    def __init__(
        self,
        backend: Optional[BaseInferenceBackend] = None,
        top_k: int = 10,
        batch_size: int = 10
    ):
        self.top_k = top_k
        self.retrieval_pipeline = RetrievalPipeline(top_k=top_k)
        self.scorer = BatchedFacetScorer(backend=backend, batch_size=batch_size)

    def initialize(self) -> "FacetEvaluatorPipeline":
        """
        Initializes vector indices and retrieval pipeline.
        """
        self.retrieval_pipeline.initialize()
        return self

    @property
    def all_facets(self) -> List[Dict[str, Any]]:
        if self.retrieval_pipeline and hasattr(self.retrieval_pipeline, "documents"):
            return self.retrieval_pipeline.documents
        return []

    @property
    def retriever(self):
        return self.retrieval_pipeline.retriever

    def evaluate_conversation(
        self,
        conversation_text: str,
        top_k: Optional[int] = None,
        specific_facets: Optional[List[Dict[str, Any]]] = None
    ) -> EvaluationPipelineResponse:
        """
        Runs complete evaluation pipeline for a conversation transcript.

        Returns:
            EvaluationPipelineResponse: Structured output containing scored and abstained results.
        """
        pipe_start = time.time()
        request_id = f"req_{uuid.uuid4().hex[:8]}"

        print(f"[PIPELINE] request_id={request_id} conversation_len={len(conversation_text)} start")

        if not self.retrieval_pipeline.is_initialized:
            self.initialize()

        k_val = top_k if top_k is not None else self.top_k

        # Step 1: Candidate Retrieval or direct facet input
        ret_start = time.time()
        if specific_facets:
            candidates = specific_facets
        else:
            candidates = self.retrieval_pipeline.retrieve(conversation_text, top_k=k_val)

        ret_ms = int((time.time() - ret_start) * 1000)

        # Step 2: Route unobservable facets directly to abstention without LLM calls
        observable_candidates = []
        unobservable_results = []

        for candidate in candidates:
            if candidate.get("conversation_observable", True) is False:
                unobservable_results.append(
                    FacetScoreResult(
                        facet_id=candidate.get("facet_id", "UNKNOWN"),
                        facet=candidate.get("normalized_facet", candidate.get("raw_facet", "Unknown")),
                        status="not_observable",
                        score=None,
                        confidence=0.99,
                        evidence=None,
                        reason=candidate.get("abstention_reason", "Facet is unobservable from text.")
                    )
                )
            else:
                observable_candidates.append(candidate)

        print(f"[RETRIEVAL] request_id={request_id} total_candidates={len(candidates)} observable={len(observable_candidates)} unobservable={len(unobservable_results)} latency_ms={ret_ms}")

        # Step 3: Batched LLM Scoring for observable candidates in 1 single compact batch
        score_start = time.time()
        scored_results, pipeline_logs = self.scorer.score_candidates(
            conversation_text, observable_candidates, request_id=request_id
        )
        score_ms = int((time.time() - score_start) * 1000)

        total_ms = int((time.time() - pipe_start) * 1000)
        print(f"[PIPELINE] request_id={request_id} evaluated={len(scored_results)} abstained={len(unobservable_results)} total_latency_ms={total_ms}")

        return EvaluationPipelineResponse(
            conversation=conversation_text,
            total_candidates_retrieved=len(candidates),
            evaluated_results=scored_results,
            abstained_results=unobservable_results
        )
