"""
Script to print detailed retrieval diagnostics for the 10 real test cases.
Inspects BM25 rank/score, Dense rank/score, and RRF rank/score for each query.
"""

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.pipeline import RetrievalPipeline

TEST_10_CASES = [
    {"id": 1, "text": "I am taking a wild risk going skydiving this weekend without a backup parachute!", "expected": "Risktaking"},
    {"id": 2, "text": "I felt dizzy after my morning walk and my blood pressure reading was 130/85.", "expected": "Blood pressure level"},
    {"id": 3, "text": "I LOVE staying past midnight fixing your typos for free because I care about perfection.", "expected": "Acidity"},
    {"id": 4, "text": "He told the team I was exhibiting hesitation, but I completely disagree with his statement.", "expected": "Hesitation"},
    {"id": 5, "text": "I handed in my resignation letter today without having another job lined up.", "expected": "Risktaking"},
    {"id": 6, "text": "My knees were knocking in sheer terror as I stepped onto the stage.", "expected": "Fearfulness"},
    {"id": 7, "text": "Could you please pass me the salt?", "expected": "Civility"},
    {"id": 8, "text": "I have been feeling super tired and con flojera all afternoon.", "expected": "Slothfulness"},
    {"id": 9, "text": "I feel blue and down in the dumps today.", "expected": "Moroseness"},
    {"id": 10, "text": "I spent six years studying advanced statistical reasoning in Python.", "expected": "Statistical Reasoning"}
]


def run_retrieval_diagnostics():
    print("==================================================================")
    print("  10-CASE DETAILED RETRIEVAL DIAGNOSTIC SUITE")
    print("==================================================================")

    pipeline = RetrievalPipeline(top_k=30)
    pipeline.initialize()

    for item in TEST_10_CASES:
        cid = item["id"]
        text = item["text"]
        exp = item["expected"]

        bm25_res = pipeline.retriever.bm25_indexer.search(text, top_k=30)
        dense_res = pipeline.retriever.dense_indexer.search(text, top_k=30)
        rrf_candidates = pipeline.retrieve(text, top_k=10)

        # Ranks
        bm25_rank = next((r_idx for r_idx, (doc, sc) in enumerate(bm25_res, 1) if exp.lower() in doc.get("normalized_facet", "").lower()), None)
        dense_rank = next((r_idx for r_idx, (doc, sc) in enumerate(dense_res, 1) if exp.lower() in doc.get("normalized_facet", "").lower()), None)
        rrf_rank = next((r_idx for r_idx, doc in enumerate(rrf_candidates, 1) if exp.lower() in doc.get("normalized_facet", "").lower()), None)

        print(f"\n[CASE {cid}] Query: \"{text}\"")
        print(f"Target Facet: '{exp}' | RRF Rank: {rrf_rank or 'NOT IN TOP 10'} | BM25 Rank: {bm25_rank or 'N/A'} | Dense Rank: {dense_rank or 'N/A'}")
        print("Top 5 RRF Candidates:")
        for idx, c in enumerate(rrf_candidates[:5], 1):
            fname = c.get("normalized_facet", c.get("raw_facet"))
            print(f"   {idx}. {fname}")

    print("\n==================================================================")
    print("  RETRIEVAL DIAGNOSTICS COMPLETE")
    print("==================================================================")


if __name__ == "__main__":
    run_retrieval_diagnostics()
