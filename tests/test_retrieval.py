"""
Unit Tests for Candidate Retrieval & Hybrid Indexer.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25 import BM25Indexer
from src.retrieval.search import HybridFacetRetriever
from src.retrieval.pipeline import RetrievalPipeline


@pytest.fixture
def sample_facets():
    return [
        {
            "facet_id": "FACET_001",
            "raw_facet": "Risktaking",
            "normalized_facet": "Risktaking",
            "facet_type": "conversational_trait",
            "conversation_observable": True,
            "scoring_definition": "Evaluates risk-taking propensity and willingness to encounter danger."
        },
        {
            "facet_id": "FACET_002",
            "raw_facet": "Hesitation",
            "normalized_facet": "Hesitation",
            "facet_type": "conversational_trait",
            "conversation_observable": True,
            "scoring_definition": "Evaluates reluctance or delays in decision making."
        },
        {
            "facet_id": "FACET_003",
            "raw_facet": "Parathyroid-hormone level",
            "normalized_facet": "Parathyroid-hormone level",
            "facet_type": "medical_biomarker",
            "conversation_observable": False,
            "scoring_definition": "Requires biological lab testing."
        }
    ]


def test_bm25_search(sample_facets):
    indexer = BM25Indexer()
    indexer.fit(sample_facets)
    results = indexer.search("risk taking danger", top_k=2)

    assert len(results) > 0
    top_doc, score = results[0]
    assert top_doc["facet_id"] == "FACET_001"
    assert score > 0.0


def test_unobservable_facet_prefiltering(sample_facets):
    retriever = HybridFacetRetriever()
    retriever.fit(sample_facets)

    # Confirm only observable facets were indexed
    assert len(retriever.documents) == 2
    for doc in retriever.documents:
        assert doc["conversation_observable"] is True
        assert doc["facet_id"] != "FACET_003"


def test_hybrid_retrieval(sample_facets):
    retriever = HybridFacetRetriever()
    retriever.fit(sample_facets)
    candidates = retriever.retrieve_candidates("unsure pause hesitation", top_k=2)

    assert len(candidates) > 0
    assert candidates[0]["facet_id"] == "FACET_002"
    assert "retrieval_rrf_score" in candidates[0]


def test_retrieval_pipeline_initialization():
    pipeline = RetrievalPipeline(top_k=5)
    pipeline.initialize()
    candidates = pipeline.retrieve("I am feeling adventurous and willing to take risks.")

    assert len(candidates) > 0
    assert len(candidates) <= 5
    for c in candidates:
        assert c["conversation_observable"] is True


def test_pure_python_vectorizer_fallback(sample_facets):
    from src.retrieval.indexer import DenseVectorIndexer
    indexer = DenseVectorIndexer()
    indexer.mode = "python"
    indexer.fit(sample_facets)
    results = indexer.search("risk taking danger", top_k=2)

    assert len(results) > 0
    assert results[0][0]["facet_id"] == "FACET_001"

