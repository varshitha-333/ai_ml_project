"""
Pytest Unit & Regression Test Suite for Qwen Harness (JSON Extraction, Facet Resolution, Latency).
"""

import pytest
import json
from scratch.debug_qwen_cases import (
    extract_json_results,
    normalize_facet_name,
    validate_result_schema
)


def test_json_array():
    raw_input = '[{"facet_id": "FACET_001", "score": 5.0, "status": "scored"}]'
    parsed, state = extract_json_results(raw_input)
    assert state == "valid_array"
    assert len(parsed) == 1
    assert parsed[0]["facet_id"] == "FACET_001"
    assert parsed[0]["score"] == 5.0


def test_wrapped_results_object():
    raw_input = '{"results": [{"facet_id": "FACET_001", "score": 5.0, "status": "scored"}]}'
    parsed, state = extract_json_results(raw_input)
    assert len(parsed) == 1
    assert parsed[0]["facet_id"] == "FACET_001"
    assert parsed[0]["score"] == 5.0


def test_markdown_json():
    raw_input = '```json\n[{"facet_id": "FACET_001", "score": 5.0, "status": "scored"}]\n```'
    parsed, state = extract_json_results(raw_input)
    assert len(parsed) == 1
    assert parsed[0]["score"] == 5.0


def test_json_with_surrounding_text():
    raw_input = 'Here is the JSON output:\n[{"facet_id": "FACET_001", "score": 5.0, "status": "scored"}]\nHope this helps.'
    parsed, state = extract_json_results(raw_input)
    assert len(parsed) == 1
    assert parsed[0]["score"] == 5.0


def test_single_object():
    raw_input = '{"facet_id": "FACET_001", "score": 5.0, "status": "scored"}'
    parsed, state = extract_json_results(raw_input)
    assert state == "single_object"
    assert len(parsed) == 1
    assert parsed[0]["facet_id"] == "FACET_001"


def test_malformed_json():
    raw_input = '[{"facet_id": "FACET_001", "score": 5.0, status: broken'
    parsed, state = extract_json_results(raw_input)
    assert state == "parse_error"
    assert parsed == []


def test_truncated_response():
    raw_input = '[\n    {"facet_id": "FACET_001", "facet_name": "Risktaking", "score": 4.0, "status": "scored", "confidence": "high", "reasoning": "Skydiving"},\n    {"facet_id": "FACET_015", "facet_name": "Adventure-Seeking Behavior", "score": 4.0, "status": "scored", "confidence": "high", "reasoning": "Skydiving"},\n    {"facet_id": "FACET_3'
    parsed, state = extract_json_results(raw_input)
    assert state == "truncated_recovery"
    assert len(parsed) == 2
    assert parsed[0]["facet_id"] == "FACET_001"
    assert parsed[1]["facet_id"] == "FACET_015"


def test_empty_response():
    raw_input = ""
    parsed, state = extract_json_results(raw_input)
    assert state == "empty_output"
    assert parsed == []


def test_trailing_colon_match():
    assert normalize_facet_name("Democratic Leadership:") == "Democratic Leadership"
    assert normalize_facet_name("  HonestyHumility:  ") == "HonestyHumility"
    assert normalize_facet_name("Hesitation") == "Hesitation"


def test_schema_validation():
    parsed_items = [{"facet_id": "FACET_001", "score": "4.5", "status": "SCORED"}]
    validated, state = validate_result_schema(parsed_items, {"FACET_001"})
    assert state == "valid"
    assert len(validated) == 1
    assert validated[0]["score"] == 4.5
    assert validated[0]["status"] == "scored"
