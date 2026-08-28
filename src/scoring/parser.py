"""
Robust JSON Extractor and Pydantic Validator with Syntax Repair, Refusal Recovery,
and Duplicate Deduplication for LLM Scoring Responses.
"""

import json
import re
from typing import List, Dict, Any, Tuple
from pydantic import ValidationError
from src.scoring.schemas import FacetScoreResult


class RobustJSONParser:
    """
    Parser for LLM json output featuring codeblock extraction, trailing comma syntax repair,
    refusal string detection, float score rounding, deduplication, and fallback handling.
    """

    REFUSAL_KEYWORDS = [
        "cannot fulfill", "cannot answer", "unable to evaluate",
        "as an ai", "as a language model", "sorry, but i cannot"
    ]

    def extract_json_text(self, text: str) -> str:
        """
        Extracts JSON content from raw markdown or codeblock wrappers.
        """
        text = text.strip()
        
        # Check for codeblocks
        codeblock_match = re.search(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', text, re.DOTALL)
        if codeblock_match:
            return codeblock_match.group(1).strip()

        # Check for bracket boundaries
        bracket_match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if bracket_match:
            return bracket_match.group(1).strip()

        return text

    def clean_json_syntax(self, json_str: str) -> str:
        """
        Repairs common LLM JSON syntax errors such as trailing commas and unclosed arrays.
        """
        json_str = json_str.strip()
        # Remove trailing commas before closing brackets
        json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
        
        # Repair unclosed array if cut off mid-output
        if json_str.startswith("[") and not json_str.endswith("]"):
            last_brace = json_str.rfind("}")
            if last_brace != -1:
                json_str = json_str[:last_brace + 1] + "]"
        return json_str

    def is_refusal_response(self, raw_text: str) -> bool:
        """
        Checks if the LLM output is a refusal string.
        """
        low = raw_text.lower()
        return any(rk in low for rk in self.REFUSAL_KEYWORDS)

    def parse_and_validate_batch(
        self,
        raw_llm_output: str,
        expected_candidates: List[Dict[str, Any]]
    ) -> Tuple[List[FacetScoreResult], List[str]]:
        """
        Parses raw LLM text into validated FacetScoreResult objects.
        Falls back safely to insufficient_evidence abstentions for any missing/malformed items.
        """
        logs = []
        parsed_results: List[FacetScoreResult] = []
        candidate_map = {c["facet_id"]: c for c in expected_candidates}
        seen_facet_ids = set()

        # Handle explicit refusal strings
        if self.is_refusal_response(raw_llm_output):
            logs.append("LLM returned a refusal response string. Routing all candidates to abstention fallback.")
            for c in expected_candidates:
                parsed_results.append(FacetScoreResult(
                    facet_id=c["facet_id"],
                    facet=c.get("normalized_facet", c.get("facet", "Unknown")),
                    status="insufficient_evidence",
                    score=None,
                    confidence=0.85,
                    evidence=None,
                    reason="Model refused or was unable to provide a response for this dialogue snippet."
                ))
            return parsed_results, logs

        json_text = self.extract_json_text(raw_llm_output)
        cleaned_json = self.clean_json_syntax(json_text)

        raw_items = []
        try:
            raw_items = json.loads(cleaned_json)
            if isinstance(raw_items, dict):
                raw_items = [raw_items]
        except json.JSONDecodeError as err:
            logs.append(f"JSONDecodeError: {err}. Attempting regex item extraction...")
            # Extract individual JSON objects safely
            matches = re.finditer(r'\{(?:\s*"[a-zA-Z0-9_]+"\s*:\s*(?:"(?:\\.|[^"\\])*"|true|false|null|-?\d+(?:\.\d+)?)\s*,?\s*)+\}', cleaned_json)
            for m in matches:
                try:
                    raw_items.append(json.loads(self.clean_json_syntax(m.group(0))))
                except Exception:
                    pass

        # Validate parsed items against Pydantic schema
        validated_by_id = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            
            fid = item.get("facet_id")
            if not fid or fid not in candidate_map:
                continue
            
            # Prevent duplicate facet IDs
            if fid in seen_facet_ids:
                logs.append(f"Duplicate facet_id '{fid}' detected in response. Preserving first valid item.")
                continue

            # Float score repair (e.g., 4.0 -> 4, 4.7 -> 5)
            raw_score = item.get("score")
            if isinstance(raw_score, float):
                item["score"] = int(round(raw_score))

            # Status standardization
            raw_status = str(item.get("status", "")).lower()
            if raw_status not in ["scored", "insufficient_evidence", "not_observable", "invalid_facet"]:
                item["status"] = "insufficient_evidence"

            # Force evidence & score consistency
            if item["status"] != "scored":
                item["score"] = None
                item["evidence"] = None

            try:
                valid_result = FacetScoreResult(**item)
                validated_by_id[fid] = valid_result
                seen_facet_ids.add(fid)
            except ValidationError as ve:
                logs.append(f"Pydantic ValidationError for facet_id '{fid}': {ve}")

        # Ensure all expected candidates have a result
        for c in expected_candidates:
            fid = c["facet_id"]
            if fid in validated_by_id:
                parsed_results.append(validated_by_id[fid])
            else:
                logs.append(f"Candidate '{fid}' missing from valid LLM items. Applying default abstention.")
                parsed_results.append(FacetScoreResult(
                    facet_id=fid,
                    facet=c.get("normalized_facet", c.get("facet", "Unknown")),
                    status="insufficient_evidence",
                    score=None,
                    confidence=0.85,
                    evidence=None,
                    reason=f"Conversational text does not express sufficient evidence for trait '{c.get('normalized_facet', fid)}'."
                ))

        return parsed_results, logs


def parse_and_validate_scores(raw_llm_output: str, expected_candidates: List[Dict[str, Any]]) -> Tuple[List[FacetScoreResult], List[str]]:
    parser = RobustJSONParser()
    return parser.parse_and_validate_batch(raw_llm_output, expected_candidates)
