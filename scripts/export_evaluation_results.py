"""
Script to evaluate 20 Tricky Anti-Hallucination Scenarios and export structured JSON outputs.
"""

import sys
from pathlib import Path
import json
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator_pipeline import FacetEvaluatorPipeline
from src.scoring.inference_backend import HuggingFaceInferenceBackend, MockInferenceBackend
from scripts.run_evaluation_benchmark import TRICKY_BENCHMARK_SCENARIOS


def export_genuine_reports(use_gpu: bool = True):
    output_path = PROJECT_ROOT / "outputs" / "evaluation_results.json"
    
    if use_gpu and torch.cuda.is_available():
        print("CUDA GPU detected! Loading genuine Qwen2.5 open-weight LLM model...")
        backend = HuggingFaceInferenceBackend(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            load_in_4bit=True
        )
    else:
        print("CUDA GPU not available. Using local CPU testing backend...")
        backend = MockInferenceBackend()

    print("Initializing Facet Evaluator Pipeline...")
    pipeline = FacetEvaluatorPipeline(backend=backend, top_k=10, batch_size=5)
    pipeline.initialize()

    all_evaluation_outputs = {}

    for i, sc in enumerate(TRICKY_BENCHMARK_SCENARIOS, 1):
        sid = sc["id"]
        text = sc["conversation_text"]
        print(f"[{i}/20] Evaluating {sid}: '{text}'...")

        response = pipeline.evaluate_conversation(text)

        all_evaluation_outputs[sid] = {
            "scenario_description": sc["description"],
            "conversation_text": text,
            "total_candidates_retrieved": response.total_candidates_retrieved,
            "evaluated_results": [r.model_dump() for r in response.evaluated_results],
            "abstained_results": [r.model_dump() for r in response.abstained_results]
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_evaluation_outputs, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully saved 20 Tricky Scenario evaluation results to:\n  {output_path.resolve()}")


if __name__ == "__main__":
    export_genuine_reports()
