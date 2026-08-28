"""
API Request and Response Pydantic Schemas for Frontend Contract Alignment.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from src.scoring.schemas import FacetScoreResult


class EvaluateRequest(BaseModel):
    """
    Request payload for POST /evaluate endpoint.
    """
    conversation: str = Field(..., description="Conversational dialogue text to evaluate", example="I am taking a wild risk going skydiving!")

    @field_validator("conversation")
    def validate_conversation_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Conversation text cannot be empty or whitespace-only.")
        if len(v) > 10000:
            raise ValueError("Conversation text exceeds maximum allowed length of 10,000 characters.")
        return v.strip()


class EvaluationMetadata(BaseModel):
    """
    Metadata summary for evaluation execution.
    """
    retrieved_count: int = Field(..., description="Total candidate facets retrieved")
    scored_count: int = Field(..., description="Total facets assigned 'scored' status")
    abstained_count: int = Field(..., description="Total facets assigned abstention status")


class EvaluateResponse(BaseModel):
    """
    Response payload for POST /evaluate endpoint.
    Matches exact required assignment format.
    """
    results: List[FacetScoreResult] = Field(..., description="List of evaluated facet results")
    metadata: EvaluationMetadata = Field(..., description="Execution metadata summary")


class RetrieveRequest(BaseModel):
    """
    Request payload for POST /retrieve endpoint.
    """
    conversation: str = Field(..., description="Input dialogue text to retrieve candidates for")
    top_k: Optional[int] = Field(10, ge=1, le=50, description="Top-K candidates to retrieve")

    @field_validator("conversation")
    def validate_conversation_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Conversation text cannot be empty or whitespace-only.")
        return v.strip()


class RetrieveResponse(BaseModel):
    """
    Response payload for POST /retrieve endpoint.
    """
    retrieved_candidates: List[Dict[str, Any]] = Field(..., description="List of retrieved candidate facet dictionaries")
    total_retrieved: int = Field(..., description="Total candidates retrieved")


class HealthResponse(BaseModel):
    """
    Response payload for GET /health endpoint.
    """
    status: str = Field("healthy", description="API health status")
    model_backend: str = Field(..., description="Active inference backend mode")
    colab_inference_status: str = Field("offline", description="Status of remote/Colab GPU inference endpoint")
    catalog_facets_loaded: int = Field(..., description="Total facets loaded in memory")
    version: str = Field("1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """
    Standard clean error response schema (hides stack traces and credentials).
    """
    error: str = Field(..., description="Error category title")
    message: str = Field(..., description="User-friendly error message")
