"""
Unit & Integration Tests for Cross-Encoder Reranker & Fallback Safety.
"""

import pytest
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.search import HybridFacetRetriever
from src.retrieval.pipeline import RetrievalPipeline


def test_reranker_initialization():
    reranker = CrossEncoderReranker()
    assert reranker.is_loaded is False
    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L6-v2"


def test_reranker_empty_candidates():
    reranker = CrossEncoderReranker()
    res = reranker.rerank("skydiving without parachute", [], top_k=10)
    assert res == []


def test_reranker_fewer_candidates_than_k():
    candidates = [
        {"facet_id": "FACET_001", "normalized_facet": "Risktaking", "scoring_definition": "Willingness to take risks"},
        {"facet_id": "FACET_002", "normalized_facet": "Naivety", "scoring_definition": "Lack of experience"}
    ]
    reranker = CrossEncoderReranker()
    res = reranker.rerank("skydiving without parachute", candidates, top_k=10)
    assert len(res) == 2
    assert "reranker_score" in res[0]


def test_retriever_feature_flag_disabled():
    retriever = HybridFacetRetriever(reranker_enabled=False)
    docs = [
        {"facet_id": "FACET_001", "normalized_facet": "Risktaking", "conversation_observable": True, "scoring_definition": "Taking risks"},
        {"facet_id": "FACET_002", "normalized_facet": "Naivety", "conversation_observable": True, "scoring_definition": "Inexperience"}
    ]
    retriever.fit(docs)
    results = retriever.retrieve_candidates("skydiving", top_k=2, use_reranker=False)
    assert len(results) == 2
    assert "reranker_score" not in results[0]


test_retriever_feature_flag_disabled()


def test_retriever_feature_flag_enabled():
    retriever = HybridFacetRetriever(reranker_enabled=True)
    docs = [
        {"facet_id": "FACET_001", "normalized_facet": "Risktaking", "conversation_observable": True, "scoring_definition": "Taking extreme risks"},
        {"facet_id": "FACET_002", "normalized_facet": "Naivety", "conversation_observable": True, "scoring_definition": "Lack of wisdom"}
    ]
    retriever.fit(docs)
    results = retriever.retrieve_candidates("skydiving without backup parachute", top_k=2, use_reranker=True)
    assert len(results) == 2
    assert "reranker_score" in results[0]


def test_unobservable_facets_never_reranked():
    retriever = HybridFacetRetriever(reranker_enabled=True)
    docs = [
        {"facet_id": "FACET_001", "normalized_facet": "Risktaking", "conversation_observable": True, "scoring_definition": "Taking risks"},
        {"facet_id": "MED_001", "normalized_facet": "Blood pressure level", "conversation_observable": False, "scoring_definition": "Blood reading"}
    ]
    retriever.fit(docs)
    results = retriever.retrieve_candidates("dizzy blood pressure 130/85", top_k=10, use_reranker=True)
    retrieved_fids = {d["facet_id"] for d in results}
    assert "MED_001" not in retrieved_fids


def test_reranker_fallback_on_model_failure(monkeypatch):
    reranker = CrossEncoderReranker()

    def mock_load_failure():
        return False

    monkeypatch.setattr(reranker, "load_model", mock_load_failure)
    candidates = [{"facet_id": "FACET_001", "normalized_facet": "Risktaking"}]
    res = reranker.rerank("skydiving", candidates, top_k=10)
    assert res == candidates
