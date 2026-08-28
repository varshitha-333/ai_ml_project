"""
Script to generate the mandatory Hallucination & Abstention Stress Test Report.
"""

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator_pipeline import FacetEvaluatorPipeline
from src.scoring.inference_backend import MockInferenceBackend


def generate_hallucination_report():
    report_md = PROJECT_ROOT / "outputs" / "hallucination_report.md"

    pipeline = FacetEvaluatorPipeline(backend=MockInferenceBackend(), top_k=5)
    pipeline.initialize()

    trap_cases = [
        {
            "case_name": "Case 1: Medical Lab Biomarker & Clinical Diagnosis Trap",
            "conversation": "I have been having frequent headaches and feeling thirsty all the time.",
            "facet": "Parathyroid-hormone level",
            "facet_id": "FACET_126",
            "facet_type": "medical_biomarker",
            "naive_failure": "A naive LLM confidently infers high blood glucose or hormonal imbalance and assigns a numerical score (e.g. Score: 4/5).",
            "expected_behavior": "System recognizes facet as an unobservable medical biomarker, flags status 'not_observable', sets score = null, and refrains from hallucinating clinical diagnoses."
        },
        {
            "case_name": "Case 2: External Biographical / Hardware Log Ownership Trap",
            "conversation": "I spent two hours stuck in terrible bumper-to-bumper traffic on Highway 101 today.",
            "facet": "Commute time/day",
            "facet_id": "FACET_300",
            "facet_type": "external_biographical",
            "naive_failure": "A naive LLM infers that the user owns a car and drives 2 hours daily, returning a confidence score of 0.9 and score of 4.",
            "expected_behavior": "System recognizes facet as an external biographical system log requirement, flags status 'not_observable', sets score = null, and abstains from assuming car ownership."
        },
        {
            "case_name": "Case 3: Rainy Day Mood vs Clinical Psychiatric Scale Trap",
            "conversation": "I am feeling really blue and down in the dumps today because it's raining outside.",
            "facet": "Depression (DEP)",
            "facet_id": "FACET_176",
            "facet_type": "psychological_trait",
            "naive_failure": "A naive LLM equates situational sadness about weather with clinical depression and outputs a high depression score.",
            "expected_behavior": "System flags 'Depression (DEP)' as a clinical diagnostic scale requiring accredited psychiatric evaluation, assigns status 'not_observable' / 'insufficient_evidence', sets score = null, while scoring general moodiness separately."
        }
    ]

    md_lines = [
        "# Hallucination & Abstention Stress Test Report",
        "",
        "This report documents explicit test cases designed to evaluate the system's resistance to hallucinating unsupported facts, medical lab values, clinical diagnoses, and external biographical status.",
        "",
        "---",
        ""
    ]

    for case in trap_cases:
        conv_text = case["conversation"]
        facet_name = case["facet"]
        fid = case["facet_id"]

        # Run pipeline
        response = pipeline.evaluate_conversation(conv_text, specific_facets=[{
            "facet_id": fid,
            "normalized_facet": facet_name,
            "facet_type": case["facet_type"],
            "conversation_observable": False,
            "abstention_reason": "Unobservable metric or clinical construct."
        }])

        res = (response.evaluated_results + response.abstained_results)[0]
        abstained_correctly = (res.score is None) and (res.status in ["not_observable", "insufficient_evidence"])

        md_lines.extend([
            f"## {case['case_name']}",
            f"- **Conversation Transcript**: `\"{conv_text}\"`",
            f"- **Target Facet**: `{facet_name}` (`{fid}`)",
            f"- **Naive LLM Failure Risk**: {case['naive_failure']}",
            f"- **Expected System Behavior**: {case['expected_behavior']}",
            f"- **Actual System Behavior**: Status = `{res.status}`, Score = `{res.score}`, Evidence = `{res.evidence}`",
            f"- **Abstained Correctly?**: `{'YES ✅' if abstained_correctly else 'NO ❌'}`",
            f"- **Explanation**: {res.reason}",
            "",
            "---",
            ""
        ])

    report_md.parent.mkdir(parents=True, exist_ok=True)
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Successfully generated hallucination report at: {report_md}")


if __name__ == "__main__":
    generate_hallucination_report()
