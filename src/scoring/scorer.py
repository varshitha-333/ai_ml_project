"""
Batched Facet Scorer Engine.

Splits candidate facets into compact batches, invokes the inference backend, passes completion
to the robust parser, and aggregates structured results.
"""

import time
from typing import List, Dict, Any, Tuple, Optional
from src.scoring.schemas import FacetScoreResult
from src.scoring.prompt_templates import SCORING_SYSTEM_PROMPT, build_batched_scoring_prompt
from src.scoring.inference_backend import BaseInferenceBackend, MockInferenceBackend
from src.scoring.parser import parse_and_validate_scores


class BatchedFacetScorer:
    """
    Orchestrates compact batched LLM evaluation over candidate facets.
    """

    def __init__(
        self,
        backend: Optional[BaseInferenceBackend] = None,
        batch_size: int = 10
    ):
        self.backend = backend if backend is not None else MockInferenceBackend()
        self.batch_size = max(1, batch_size)

    def score_candidates(
        self,
        conversation_text: str,
        candidate_facets: List[Dict[str, Any]],
        request_id: str = "req_1"
    ) -> Tuple[List[FacetScoreResult], List[str]]:
        """
        Evaluates candidate facets in compact batches against conversation text.

        Returns:
            Tuple[List[FacetScoreResult], List[str]]: (scored_results, all_pipeline_logs)
        """
        if not candidate_facets or not conversation_text.strip():
            return [], []

        all_results: List[FacetScoreResult] = []
        all_logs: List[str] = []

        total_batches = (len(candidate_facets) + self.batch_size - 1) // self.batch_size

        # Evaluate candidate facets in compact batches
        for batch_idx, i in enumerate(range(0, len(candidate_facets), self.batch_size), start=1):
            batch = candidate_facets[i : i + self.batch_size]

            system_prompt = SCORING_SYSTEM_PROMPT
            user_prompt = build_batched_scoring_prompt(conversation_text, batch)

            b_start = time.time()
            print(f"[INFERENCE] request_id={request_id} batch={batch_idx}/{total_batches} count={len(batch)} start")

            try:
                raw_completion = self.backend.generate(system_prompt, user_prompt)
                parsed_results, batch_logs = parse_and_validate_scores(raw_completion, batch)
                all_results.extend(parsed_results)
                all_logs.extend(batch_logs)
                b_ms = int((time.time() - b_start) * 1000)
                print(f"[INFERENCE] request_id={request_id} batch={batch_idx}/{total_batches} status=success latency_ms={b_ms}")

            except Exception as e:
                b_ms = int((time.time() - b_start) * 1000)
                err_msg = f"Inference backend failure on batch {batch_idx}/{total_batches}: {e}"
                print(f"[INFERENCE] request_id={request_id} batch={batch_idx}/{total_batches} status=inference_error latency_ms={b_ms} error='{e}'")
                all_logs.append(err_msg)

                # STRICT ABSTENTION INVARIANT: Infrastructure errors route to 'inference_error'
                # (NEVER conflated with conversational 'insufficient_evidence')
                for f in batch:
                    all_results.append(
                        FacetScoreResult(
                            facet_id=f.get("facet_id", "UNKNOWN"),
                            facet=f.get("normalized_facet", f.get("raw_facet", "Unknown")),
                            status="inference_error",
                            score=None,
                            confidence=0.0,
                            evidence=None,
                            reason=f"Model inference failed: {e}"
                        )
                    )

        return all_results, all_logs
