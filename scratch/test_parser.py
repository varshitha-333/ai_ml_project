"""
Unit Test Suite for extract_json_results in scratch/run_solution1_qwen_30.py.
Verifies parsing accuracy across all 4 LLM output formats.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scratch.run_solution1_qwen_30 import extract_json_results

def test_parser():
    print("======================================================================")
    print("TESTING EXTRACT_JSON_RESULTS PARSER ACCURACY")
    print("======================================================================")

    test_cases = [
        ("Format 1: Raw JSON Array", '[{"facet_id": "FACET_001", "score": 4.0, "status": "scored"}]'),
        ("Format 2: Markdown Codeblock", '```json\n[{"facet_id": "FACET_001", "score": 4.0, "status": "scored"}]\n```'),
        ("Format 3: Object Wrapper", '{"results": [{"facet_id": "FACET_001", "score": 4.0, "status": "scored"}]}'),
        ("Format 4: Text Preamble + Array", 'Here is the JSON evaluation:\n[{"facet_id": "FACET_001", "score": 4.0, "status": "scored"}]')
    ]

    all_passed = True
    for label, raw_text in test_cases:
        res = extract_json_results(raw_text)
        is_valid = isinstance(res, list) and len(res) > 0 and res[0].get("score") == 4.0
        status_str = "[PASS]" if is_valid else "[FAIL]"
        print(f"{label:<35} | {status_str} | Parsed: {res}")
        if not is_valid:
            all_passed = False

    print("======================================================================")
    if all_passed:
        print("ALL 4 PARSER TEST CASES PASSED PERFECTLY!")
    else:
        print("SOME PARSER TEST CASES FAILED.")
    print("======================================================================")

if __name__ == "__main__":
    test_parser()
