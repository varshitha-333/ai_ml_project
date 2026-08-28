"""
Unit Tests for Benchmark Reference Set & Evaluation Metrics.
"""

import sys
from pathlib import Path
import json
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_benchmark_evaluation import run_benchmark_evaluation
from scripts.generate_hallucination_report import generate_hallucination_report
from scripts.run_retrieval_ablation import run_retrieval_ablation


def test_reference_set_structure():
    ref_path = PROJECT_ROOT / "data" / "benchmark_reference_set.json"
    assert ref_path.exists()

    with open(ref_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) >= 10
    total_facets = set()

    for item in data:
        assert "conversation_id" in item
        assert "conversation_text" in item
        assert "annotations" in item
        for ann in item["annotations"]:
            total_facets.add(ann["facet_id"])
            assert ann["expected_status"] in ["scored", "insufficient_evidence", "not_observable"]
            if ann["expected_status"] == "scored":
                assert ann["expected_score"] is not None
                assert 1 <= ann["expected_score"] <= 5
            else:
                assert ann["expected_score"] is None

    assert len(total_facets) >= 15


def test_run_benchmark_evaluation():
    run_benchmark_evaluation()
    report_json = PROJECT_ROOT / "outputs" / "benchmark_report.json"
    assert report_json.exists()

    with open(report_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "summary_metrics" in metrics
    summary = metrics["summary_metrics"]
    assert summary["status_accuracy_pct"] >= 50.0
    assert summary["abstention_f1_pct"] >= 50.0


def test_generate_hallucination_report():
    generate_hallucination_report()
    report_md = PROJECT_ROOT / "outputs" / "hallucination_report.md"
    assert report_md.exists()

    with open(report_md, "r", encoding="utf-8") as f:
        text = f.read()

    assert "Hallucination & Abstention Stress Test Report" in text
    assert "YES ✅" in text


def test_run_retrieval_ablation():
    run_retrieval_ablation()
    ablation_json = PROJECT_ROOT / "outputs" / "retrieval_ablation_report.json"
    assert ablation_json.exists()

    with open(ablation_json, "r", encoding="utf-8") as f:
        ab_data = json.load(f)

    assert "bm25_only" in ab_data
    assert "dense_only" in ab_data
    assert "hybrid_rrf" in ab_data
    assert ab_data["hybrid_rrf"]["recall_at_30_pct"] >= 25.0
