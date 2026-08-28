"""
Unit tests for Pydantic Result Schemas and Robust JSON Parser.
"""

import sys
from pathlib import Path
import json
import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring.schemas import FacetScoreResult, EvaluationPipelineResponse
from src.scoring.parser import RobustJSONParser
from src.scoring.inference_backend import MockInferenceBackend
from src.scoring.scorer import BatchedFacetScorer


def test_schema_valid_scored_result():
    res = FacetScoreResult(
        facet_id="FACET_001",
        facet="Risktaking",
        status="scored",
        score=4,
        confidence=0.9,
        evidence="going skydiving without a backup parachute",
        reason="Demonstrates willingness to engage in extreme physical risk."
    )
    assert res.score == 4
    assert res.status == "scored"
    assert res.evidence is not None


def test_schema_invalid_score_bounds():
    with pytest.raises(ValidationError):
        FacetScoreResult(
            facet_id="FACET_001",
            facet="Risktaking",
            status="scored",
            score=10,  # Invalid: score > 5
            confidence=0.9,
            evidence="skydiving",
            reason="High risk"
        )


def test_schema_abstained_score_forced_null():
    res = FacetScoreResult(
        facet_id="FACET_002",
        facet="Blood Glucose Level",
        status="not_observable",
        score=4,  # Should be forced to None by validator
        confidence=0.95,
        evidence="feeling thirsty",
        reason="Medical lab biomarker is unobservable."
    )
    assert res.score is None
    assert res.evidence is None


def test_extract_json_payload():
    parser = RobustJSONParser()
    markdown_payload = "Here is the response:\n```json\n[{\"facet_id\": \"FACET_001\", \"status\": \"scored\"}]\n```"
    extracted = parser.extract_json_text(markdown_payload)
    assert extracted == '[{"facet_id": "FACET_001", "status": "scored"}]'


def test_parser_malformed_json_recovery():
    parser = RobustJSONParser()
    malformed = '[{"facet_id": "FACET_001", "facet": "Risktaking", "status": "scored", "score": 4, "confidence": 0.9, "evidence": "wild risk", "reason": "risk"},]'
    candidates = [{"facet_id": "FACET_001", "normalized_facet": "Risktaking"}]
    
    results, logs = parser.parse_and_validate_batch(malformed, candidates)
    assert len(results) == 1
    assert results[0].facet_id == "FACET_001"
    assert results[0].score == 4


def test_parser_refusal_recovery():
    parser = RobustJSONParser()
    refusal_str = "As an AI model, I am unable to evaluate personal traits from this text."
    candidates = [{"facet_id": "FACET_001", "normalized_facet": "Risktaking"}]

    results, logs = parser.parse_and_validate_batch(refusal_str, candidates)
    assert len(results) == 1
    assert results[0].status == "insufficient_evidence"
    assert results[0].score is None
    assert "Model refused" in results[0].reason


def test_parser_float_score_rounding():
    parser = RobustJSONParser()
    float_payload = '[{"facet_id": "FACET_001", "facet": "Risktaking", "status": "scored", "score": 4.7, "confidence": 0.9, "evidence": "wild risk", "reason": "risk"}]'
    candidates = [{"facet_id": "FACET_001", "normalized_facet": "Risktaking"}]

    results, logs = parser.parse_and_validate_batch(float_payload, candidates)
    assert len(results) == 1
    assert results[0].score == 5


def test_parser_duplicate_facet_ids():
    parser = RobustJSONParser()
    dup_payload = '[{"facet_id": "FACET_001", "facet": "Risktaking", "status": "scored", "score": 4, "confidence": 0.9, "evidence": "wild risk", "reason": "risk"}, {"facet_id": "FACET_001", "facet": "Risktaking", "status": "scored", "score": 1, "confidence": 0.1, "evidence": "none", "reason": "dup"}]'
    candidates = [{"facet_id": "FACET_001", "normalized_facet": "Risktaking"}]

    results, logs = parser.parse_and_validate_batch(dup_payload, candidates)
    assert len(results) == 1
    assert results[0].score == 4  # Kept first valid item


def test_mock_backend_scoring():
    scorer = BatchedFacetScorer(backend=MockInferenceBackend())
    candidates = [
        {"facet_id": "FACET_001", "normalized_facet": "Risktaking", "facet_type": "conversational_trait"},
        {"facet_id": "FACET_002", "normalized_facet": "Parathyroid-hormone level", "facet_type": "medical_biomarker"}
    ]

    results, logs = scorer.score_candidates(
        "I am taking a wild risk going skydiving!", candidates
    )
    assert len(results) == 2
    assert results[0].status == "scored"
    assert results[0].score in [4, 5]
