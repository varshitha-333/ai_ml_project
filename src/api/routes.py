"""
FastAPI Route Definitions for Health, Evaluation, Retrieval, Facet Lookup, and Benchmark Execution.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any
from src.api.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    EvaluationMetadata,
    RetrieveRequest,
    RetrieveResponse,
    HealthResponse,
    ErrorResponse
)
from src.api.services import get_pipeline_service, PipelineService
from scripts.run_benchmark_evaluation import run_benchmark_evaluation
import json
from pathlib import Path

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check and pipeline status",
    tags=["System"]
)
def health_check(service: PipelineService = Depends(get_pipeline_service)):
    total_loaded = len(service.pipeline.all_facets) if service.pipeline.all_facets else 0
    is_client_ok, msg = service.client.health_check()
    colab_status = "online" if is_client_ok else f"offline ({msg})"

    return HealthResponse(
        status="healthy",
        model_backend=service.settings.backend_mode,
        colab_inference_status=colab_status,
        catalog_facets_loaded=total_loaded,
        version="1.0.0"
    )


@router.get(
    "/v1/models",
    summary="OpenAI-compatible models list endpoint",
    tags=["System"]
)
def get_v1_models(service: PipelineService = Depends(get_pipeline_service)):
    return {
        "object": "list",
        "data": [
            {
                "id": service.settings.model_name,
                "object": "model",
                "created": 1677610602,
                "owned_by": "qwen"
            }
        ]
    }


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="Evaluate conversation dialogue transcript against facet catalog",
    tags=["Evaluation"],
    responses={
        400: {"model": ErrorResponse, "description": "Empty or invalid conversation input"},
        500: {"model": ErrorResponse, "description": "Internal processing error"}
    }
)
def evaluate_conversation(
    req: EvaluateRequest,
    service: PipelineService = Depends(get_pipeline_service)
):
    try:
        response = service.pipeline.evaluate_conversation(req.conversation)
        
        all_results = response.evaluated_results + response.abstained_results
        scored_count = sum(1 for r in response.evaluated_results if r.status == "scored")
        abstained_count = len(all_results) - scored_count

        return EvaluateResponse(
            results=all_results,
            metadata=EvaluationMetadata(
                retrieved_count=response.total_candidates_retrieved,
                scored_count=scored_count,
                abstained_count=abstained_count
            )
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing error: {str(err)}"
        )


@router.get(
    "/facets/{facet_id}",
    summary="Lookup specific facet details by ID",
    tags=["Facet Catalog"],
    responses={404: {"model": ErrorResponse, "description": "Facet ID not found"}}
)
def get_facet_by_id(
    facet_id: str,
    service: PipelineService = Depends(get_pipeline_service)
):
    fid_upper = facet_id.upper()
    found_doc = next((f for f in service.pipeline.all_facets if f.get("facet_id") == fid_upper), None)
    
    if not found_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Facet with ID '{facet_id}' was not found in catalog."
        )

    # Sanitize pandas NaN values to JSON compliant null
    import math
    cleaned_doc = {}
    for k, v in found_doc.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            cleaned_doc[k] = None
        else:
            cleaned_doc[k] = v

    return cleaned_doc


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Retrieve candidate observable facets for a conversation snippet",
    tags=["Retrieval"]
)
def retrieve_candidates(
    req: RetrieveRequest,
    service: PipelineService = Depends(get_pipeline_service)
):
    try:
        k = req.top_k or service.settings.top_k
        candidates = service.pipeline.retriever.retrieve_candidates(req.conversation, top_k=k)
        return RetrieveResponse(
            retrieved_candidates=candidates,
            total_retrieved=len(candidates)
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Candidate retrieval failed: {str(err)}"
        )


@router.post(
    "/benchmark/run",
    summary="Trigger human-reference benchmark evaluation",
    tags=["Benchmark"]
)
def run_benchmark_trigger():
    try:
        run_benchmark_evaluation()
        report_path = Path(__file__).resolve().parent.parent.parent / "outputs" / "benchmark_report.json"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "success", "message": "Benchmark executed successfully."}
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark evaluation failed: {str(err)}"
        )
