"""
End-to-End Integration Tests for Facet Evaluator Pipeline.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator_pipeline import FacetEvaluatorPipeline
from src.scoring.inference_backend import MockInferenceBackend


@pytest.fixture
def pipeline():
    pipe = FacetEvaluatorPipeline(backend=MockInferenceBackend(), top_k=5)
    pipe.initialize()
    return pipe


def test_scenario_dizzy_medical_abstention(pipeline):
    res = pipeline.evaluate_conversation("I have been feeling dizzy.")
    assert res.total_candidates_retrieved > 0
    
    # Check that any medical items abstain
    for r in res.evaluated_results + res.abstained_results:
        if "blood" in r.facet.lower() or "pressure" in r.facet.lower() or "hormone" in r.facet.lower():
            assert r.score is None
            assert r.status in ["insufficient_evidence", "not_observable"]


def test_scenario_travel_unsupported_fact_abstention(pipeline):
    facet = {"facet_id": "FACET_CAR", "normalized_facet": "User owns a car", "conversation_observable": True}
    res = pipeline.evaluate_conversation("I need to travel tomorrow.", specific_facets=[facet])
    
    assert len(res.evaluated_results) == 1
    scored_item = res.evaluated_results[0]
    assert scored_item.status == "insufficient_evidence"
    assert scored_item.score is None


def test_scenario_valid_risktaking_scoring(pipeline):
    res = pipeline.evaluate_conversation("I am taking a wild risk going skydiving!")
    
    assert len(res.evaluated_results) > 0
    risk_item = next((r for r in res.evaluated_results if "risk" in r.facet.lower()), None)
    if risk_item:
        assert risk_item.status == "scored"
        assert risk_item.score in [4, 5]
        assert risk_item.evidence is not None


def test_unobservable_direct_routing(pipeline):
    unobservable_facet = {
        "facet_id": "FACET_RAW_UNOBS",
        "normalized_facet": "Parathyroid-hormone level",
        "conversation_observable": False,
        "abstention_reason": "Medical lab marker"
    }
    res = pipeline.evaluate_conversation("Hello world", specific_facets=[unobservable_facet])
    
    assert len(res.abstained_results) == 1
    assert res.abstained_results[0].status == "not_observable"
    assert res.abstained_results[0].score is None
