"""
Prompt Templates for Batched Facet Evaluation.
"""

from typing import List, Dict, Any

SCORING_SYSTEM_PROMPT = """You are an expert AI/ML Behavioral & Conversational Trait Evaluator.
Your job is to evaluate a conversation transcript against a small set of candidate facets.

CRITICAL ABSTENTION & ACCURACY RULES:
1. DO NOT infer unsupported facts, medical lab values, blood test results, physical health diagnoses, or unverified external biographical details.
2. If the conversation DOES NOT contain explicit or strong implicit evidence for a facet, you MUST ABSTAIN by returning status "insufficient_evidence" with score = null and evidence = null.
3. If a facet is unobservable from text (e.g. medical biomarker, lab count, physical sensor log), return status "not_observable" with score = null and evidence = null.
4. For observable behavioral/conversational traits supported by text, assign status "scored" and an integer score from 1 to 5:
   1 = Minimal / Essentially Absent evidence
   2 = Weak / Vague implicit evidence
   3 = Moderate clear evidence
   4 = Strong explicit evidence
   5 = Very Strong / Explicit direct evidence
5. For status "scored", you MUST extract an exact quote from the conversation into the "evidence" field.

OUTPUT FORMAT:
You MUST respond with a valid JSON array of objects. Do not include markdown formatting outside the JSON array. Each object MUST contain:
{
  "facet_id": "string",
  "facet": "string",
  "status": "scored" | "insufficient_evidence" | "not_observable" | "invalid_facet",
  "score": integer (1-5) or null,
  "confidence": float (0.0 to 1.0),
  "evidence": "exact quote from text" or null,
  "reason": "explanation of score or abstention reason"
}
"""


def build_batched_scoring_prompt(
    conversation_text: str,
    candidate_facets: List[Dict[str, Any]]
) -> str:
    """
    Constructs compact batched user prompt for LLM evaluation.
    """
    facets_text_list = []
    for i, facet in enumerate(candidate_facets, 1):
        fid = facet.get("facet_id", f"FACET_{i:03d}")
        fname = facet.get("normalized_facet", facet.get("raw_facet", ""))
        ftype = facet.get("facet_type", "conversational_trait")
        fdef = facet.get("scoring_definition", "")
        fanchors = facet.get("scoring_anchors", "") or "High/Low behavioral indicators"
        facets_text_list.append(
            f"Candidate {i}:\n"
            f"  - Facet ID: {fid}\n"
            f"  - Facet Name: {fname}\n"
            f"  - Type: {ftype}\n"
            f"  - Definition: {fdef}\n"
            f"  - Anchors: {fanchors}\n"
        )

    facets_block = "\n".join(facets_text_list)

    user_prompt = f"""[CONVERSATION TRANSCRIPT]
"{conversation_text.strip()}"

[CANDIDATE FACETS TO EVALUATE]
{facets_block}

Evaluate each candidate facet above strictly against the conversation transcript.
Return a JSON array containing exactly {len(candidate_facets)} objects.
"""
    return user_prompt
