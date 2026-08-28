"""
Script to run 5 representative test cases through the complete end-to-end production path.
Used to manually verify status, score, confidence, evidence/reason, and latency.
"""

import sys
import os
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator_pipeline import FacetEvaluatorPipeline
from src.scoring.inference_backend import RemoteInferenceClientBackend, MockInferenceBackend
from src.scoring.inference_client import InferenceClient
from src.api.config import get_settings


FIVE_REPRESENTATIVE_CASES = [
    {
        "id": "CASE_1_OBSERVABLE",
        "description": "Clearly observable behavior (Skydiving risk-taking)",
        "text": "I am taking a wild risk by going skydiving this weekend without a backup parachute!",
        "target_facet": "Risktaking"
    },
    {
        "id": "CASE_2_AMBIGUOUS",
        "description": "Ambiguous statement (Hesitation / decision delay)",
        "text": "I'm not completely sure if I should sign the offer letter today or wait a few more days.",
        "target_facet": "Hesitation"
    },
    {
        "id": "CASE_3_MEDICAL_UNOBSERVABLE",
        "description": "Unsupported medical biomarker (Blood pressure / dizziness)",
        "text": "I felt dizzy after my morning walk and my blood pressure reading was 130/85.",
        "target_facet": "Blood pressure level"
    },
    {
        "id": "CASE_4_QUOTED_OPINION",
        "description": "Quoted third-party statement (Hesitation claim rejected by speaker)",
        "text": "He told the team that I was exhibiting hesitation, but I completely disagree with his assessment.",
        "target_facet": "Hesitation"
    },
    {
        "id": "CASE_5_VAGUE_EMOTION",
        "description": "Vague emotional statement (Tiredness / general mood)",
        "text": "I feel a little bit tired today, so I think I will just stay home and read.",
        "target_facet": "Slothfulness"
    }
]


def run_5_case_validation():
    settings = get_settings()
    backend_mode = os.getenv("BACKEND_MODE", settings.backend_mode).lower()

    if backend_mode == "remote":
        client = InferenceClient(
            inference_url=settings.inference_url,
            model_name=settings.model_name,
            timeout=settings.inference_timeout
        )
        backend = RemoteInferenceClientBackend(client=client)
        mode_str = "REMOTE"
    else:
        backend = MockInferenceBackend()
        mode_str = "MOCK"

    print("==================================================================")
    print("  5-CASE REPRESENTATIVE VALIDATION SUITE")
    print("==================================================================")
    print(f"BACKEND MODE:       {mode_str} ({type(backend).__name__})")
    print(f"MODEL NAME:         {settings.model_name}")
    print(f"INFERENCE URL:      {settings.inference_url}")
    print("==================================================================\n")

    pipeline = FacetEvaluatorPipeline(backend=backend, top_k=30, batch_size=10)
    pipeline.initialize()

    results = []
    for c in FIVE_REPRESENTATIVE_CASES:
        t0 = time.time()
        print(f"[{c['id']}] {c['description']}")
        print(f"Input: \"{c['text']}\"")

        res = pipeline.evaluate_conversation(c["text"], top_k=30)
        elapsed_s = round(time.time() - t0, 2)

        all_res = res.evaluated_results + res.abstained_results
        
        # Find target facet result
        target_r = None
        for r in all_res:
            if c["target_facet"].lower() in r.facet.lower():
                target_r = r
                break

        if target_r is None and all_res:
            target_r = all_res[0]

        res_item = {
            "case_id": c["id"],
            "description": c["description"],
            "text": c["text"],
            "evaluated_facet": target_r.facet if target_r else c["target_facet"],
            "status": target_r.status if target_r else "insufficient_evidence",
            "score": target_r.score if target_r else None,
            "confidence": target_r.confidence if target_r else 0.0,
            "evidence": target_r.evidence if target_r else None,
            "reason": target_r.reason if target_r else "No candidate matched target facet.",
            "latency_s": elapsed_s
        }
        results.append(res_item)

        status_badge = f"[{res_item['status'].upper()}]"
        score_str = f"Score: {res_item['score']}" if res_item['score'] is not None else "Score: null"
        print(f"Result: {status_badge} | {score_str} | Confidence: {res_item['confidence']} | Latency: {elapsed_s}s")
        print(f"Evidence: \"{res_item['evidence']}\"")
        print(f"Reason: {res_item['reason']}\n" + "-" * 60)

    print("\n==================================================================")
    print("  5-CASE VALIDATION COMPLETE")
    print("==================================================================")


if __name__ == "__main__":
    run_5_case_validation()
