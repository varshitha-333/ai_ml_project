"""
Test extract_json_results on the exact raw truncated strings provided by the user.
"""

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scratch.debug_qwen_cases import extract_json_results

def test_user_inputs():
    user_test_001_raw = '''[
    {"facet_id": "FACET_001", "facet_name": "Risktaking", "score": 4.0, "status": "scored", "confidence": "high", "reasoning": "The text mentions signing up for a solo skydiving trip despite no prior experience, indicating a willingness to take risks."},
    {"facet_id": "FACET_015", "facet_name": "Adventure-Seeking Behavior", "score": 4.0, "status": "scored", "confidence": "high", "reasoning": "Engaging in a new and potentially dangerous activity like skydiving suggests an adventurous spirit."},
    {"facet_id": "FACET_002", "facet_name": "Naivety", "score": 2.0, "status": "scored", "confidence": "medium", "reasoning": "While the text does not explicitly show naivety, attempting a new and risky activity without prior experience could be seen as somewhat naive."},
    {"facet_id": "FACET_306", "facet_name": "Use of Nature as a Stress Reliever", "score": 0.0, "status": "abstained", "confidence": "low", "reasoning": "There is no mention of using nature as a stress reliever in the text."},
    {"facet_id": "FACET_3'''

    user_test_002_raw = '''[
    {
        "facet_id": "FACET_364",
        "facet_name": "Trust in others",
        "score": 4.0,
        "status": "scored",
        "confidence": "high",
        "reasoning": "The text suggests that the person trusts the stranger's promise to double their money, indicating a high level of trust."
    },
    {
        "facet_id": "FACET_001",
        "facet_name": "Risktaking",
        "score": 3.0,
        "status": "scored",
        "confidence": "medium",
        "reasoning": "Believing a stranger's promise to double one's money involves taking a risk, which is indicative of risk-taking behavior."
    },
    {
        "facet_id": "FACET_331",
        "facet_name": "Desire to influence others",
        "score": 0.0,
        "status": "abstained",
        "confidence": "low",
        "reasoning": "There is no evidence in the text that the person is trying to influence the stranger or anyone else."
    },
    {
        "facet_id": "FACET_151",
        "facet_name": "Contribution to Group Goals",
        "score": 0.0,
        "status": "abstained",
        "confidence": "low",
        "reason'''

    print("======================================================================")
    print("TESTING EXTRACT_JSON_RESULTS ON USER'S EXACT TRUNCATED OUTPUTS")
    print("======================================================================")

    for name, raw in [("TEST_001 Truncated Input", user_test_001_raw), ("TEST_002 Truncated Input", user_test_002_raw)]:
        parsed, state = extract_json_results(raw)
        print(f"\n--- {name} ---")
        print(f"State: {state} | Recovered Items: {len(parsed)}")
        for item in parsed:
            print(f"  - [{item.get('facet_id')}] {item.get('facet_name')}: score={item.get('score')}, status={item.get('status')}")

    print("\n======================================================================")

if __name__ == "__main__":
    test_user_inputs()
