"""
Pydantic Schema Definitions for Facet Evaluation Results and Pipeline Responses.
"""

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal


class FacetScoreResult(BaseModel):
    """
    Schema for a single evaluated facet result.
    Enforces strict score bounds (1-5) for 'scored' status and null scores/evidence for abstentions.
    """
    facet_id: str = Field(..., description="Unique facet identifier (e.g. FACET_001)")
    facet: str = Field(..., description="Normalized facet name")
    status: Literal["scored", "insufficient_evidence", "not_observable", "invalid_facet", "inference_error"] = Field(
        ..., description="Evaluation status classification"
    )
    score: Optional[int] = Field(
        None, description="Integer score between 1 and 5; null if abstained"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0"
    )
    evidence: Optional[str] = Field(
        None, description="Direct quote/grounded snippet from conversation text; null if abstained"
    )
    reason: str = Field(
        ..., description="Detailed rationale explaining score or abstention reason"
    )

    @model_validator(mode="after")
    def validate_score_and_evidence(self):
        if self.status == "scored":
            if self.score is None:
                raise ValueError("Score cannot be null when status is 'scored'.")
            if self.score < 1 or self.score > 5:
                raise ValueError(f"Score must be an integer between 1 and 5, got {self.score}.")
        else:
            self.score = None
            self.evidence = None
        return self


class BatchScoreResponse(BaseModel):
    """
    Batch evaluation response containing a list of score results.
    """
    results: List[FacetScoreResult] = Field(default_factory=list)


class EvaluationPipelineResponse(BaseModel):
    """
    Complete pipeline output response for a conversation text.
    """
    conversation: str = Field(..., description="Original input conversation text")
    total_candidates_retrieved: int = Field(..., description="Total observable candidates retrieved")
    evaluated_results: List[FacetScoreResult] = Field(default_factory=list)
    abstained_results: List[FacetScoreResult] = Field(default_factory=list)
