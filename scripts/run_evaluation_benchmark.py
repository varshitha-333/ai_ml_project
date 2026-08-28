"""
Benchmark Evaluation Script testing 20 Tricky Anti-Hallucination & Abstention Scenarios.
"""

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator_pipeline import FacetEvaluatorPipeline
from src.scoring.inference_backend import MockInferenceBackend


TRICKY_BENCHMARK_SCENARIOS = [
    {
        "id": "SCENARIO_001",
        "description": "Medical Symptom Query (Diabetes/Blood test hallucination trap)",
        "conversation_text": "I have been having frequent headaches and feeling thirsty all the time."
    },
    {
        "id": "SCENARIO_002",
        "description": "External Biographical Car Ownership Trap (Traffic statement)",
        "conversation_text": "I spent two hours stuck in terrible bumper-to-bumper traffic on Highway 101 today."
    },
    {
        "id": "SCENARIO_003",
        "description": "Emotional Mood vs Clinical Construct (Depression scale trap)",
        "conversation_text": "I am feeling really blue and down in the dumps today because it's raining."
    },
    {
        "id": "SCENARIO_004",
        "description": "Metaphorical Job Risk & Decisiveness",
        "conversation_text": "I took a huge leap of faith and submitted my resignation letter without having another job lined up!"
    },
    {
        "id": "SCENARIO_005",
        "description": "Passive-Aggressive Interpersonal Posture",
        "conversation_text": "Fine, do whatever you want. It's not like my opinion ever mattered to anyone here anyway."
    },
    {
        "id": "SCENARIO_006",
        "description": "Spiritual Retreat vs Esoteric I Ching Hexagram Trap",
        "conversation_text": "I spent the weekend quietly meditating at a peaceful retreat in the mountains."
    },
    {
        "id": "SCENARIO_007",
        "description": "Mental Math Claim vs Standardized Cognitive Test Trap",
        "conversation_text": "I can calculate 15% tip in my head in two seconds flat!"
    },
    {
        "id": "SCENARIO_008",
        "description": "Digital Nomad Lifestyle vs External Passport Log Trap",
        "conversation_text": "I've been working remotely from coffee shops in Bali and Bangkok for the last six months."
    },
    {
        "id": "SCENARIO_009",
        "description": "Financial Risk & Cryptocurrency Impulsivity",
        "conversation_text": "I put my entire life savings into a high-volatility meme cryptocurrency yesterday."
    },
    {
        "id": "SCENARIO_010",
        "description": "Extreme Brevity & Laconic Directness",
        "conversation_text": "No."
    },
    {
        "id": "SCENARIO_011",
        "description": "Overprotectiveness vs Home Security Hardware Log Trap",
        "conversation_text": "I installed five security cameras, three locks, and I track my daughter's GPS location 24/7."
    },
    {
        "id": "SCENARIO_012",
        "description": "Sarcasm, Acidity & Civility",
        "conversation_text": "Oh sure, I'd just LOVE to stay past midnight fixing your typos for free again!"
    },
    {
        "id": "SCENARIO_013",
        "description": "Religious Self-Reflection vs Memorization Count Trap",
        "conversation_text": "I read scripture every single morning before starting my day."
    },
    {
        "id": "SCENARIO_014",
        "description": "Dietary Habit Claim vs Biological Macronutrient Ratio Trap",
        "conversation_text": "I haven't eaten processed food or sugar in over three years."
    },
    {
        "id": "SCENARIO_015",
        "description": "High Hesitation & Cognitive Deliberation",
        "conversation_text": "Um... well... I guess maybe we could... wait, let me think... maybe option B?"
    },
    {
        "id": "SCENARIO_016",
        "description": "Chivalry & Warmheartedness",
        "conversation_text": "I held the elevator door for an elderly stranger and carried their heavy groceries up the stairs."
    },
    {
        "id": "SCENARIO_017",
        "description": "Direct Confrontation vs Non-Conformity",
        "conversation_text": "I directly confronted my manager about the unfair budget cuts during our team meeting."
    },
    {
        "id": "SCENARIO_018",
        "description": "Emotional Burnout & Exhaustion",
        "conversation_text": "I am completely exhausted, emotionally drained, and I can barely force myself to open my laptop."
    },
    {
        "id": "SCENARIO_019",
        "description": "Creative DIY Activity vs Pet Tracking Sensor Trap",
        "conversation_text": "I built a custom agility obstacle course in my backyard for my Golden Retriever."
    },
    {
        "id": "SCENARIO_020",
        "description": "Hypothetical Counterfactual Threat Posture",
        "conversation_text": "If I were ever in a robbery, I would probably freeze and give them all my money."
    }
]


def run_benchmark():
    print("==================================================================")
    print("    FACET EVALUATOR - 20 TRICKY ANTI-HALLUCINATION BENCHMARK      ")
    print("==================================================================")

    pipeline = FacetEvaluatorPipeline(backend=MockInferenceBackend(), top_k=10)
    pipeline.initialize()

    for i, scenario in enumerate(TRICKY_BENCHMARK_SCENARIOS, 1):
        print(f"\n[{i}/20] {scenario['id']}: {scenario['description']}")
        print(f"  Conversation: \"{scenario['conversation_text']}\"")

        response = pipeline.evaluate_conversation(scenario['conversation_text'])

        print(f"  Retrieved Candidates: {response.total_candidates_retrieved}")
        for res in response.evaluated_results + response.abstained_results:
            status_symbol = "✅" if res.status == "scored" else ("🚫" if res.status == "not_observable" else "⚠️")
            score_str = f"Score: {res.score}" if res.score is not None else "Score: null (Abstained)"
            print(f"    {status_symbol} [{res.status.upper()}] {res.facet} ({res.facet_id}) -> {score_str}")
            if res.evidence:
                print(f"       Evidence: \"{res.evidence}\"")
            print(f"       Reason: {res.reason}")

    print("\n==================================================================")
    print(" 20 Tricky Anti-Hallucination Scenarios Evaluated Successfully!   ")
    print("==================================================================")


if __name__ == "__main__":
    run_benchmark()
