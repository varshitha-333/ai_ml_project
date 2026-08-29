"""
Debug Script: Dumps un-truncated raw Qwen output for TEST_001 and TEST_002
and tests extract_json_results to diagnose the exact parsing failure.
"""

import sys
from pathlib import Path
import json
import time
import urllib.request
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scratch.debug_qwen_cases import extract_json_results, BM25Indexer, DenseVectorIndexer, normalize_facet_name

COLAB_URL = "https://salvaging-ardently-late.ngrok-free.dev/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def dump_raw_output():
    catalog_path = PROJECT_ROOT / "data" / "processed" / "enriched_facets.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog_docs = json.load(f)

    observable_docs = [d for d in catalog_docs if d.get("conversation_observable", True) is True]
    bm25 = BM25Indexer().fit(observable_docs)
    dense = DenseVectorIndexer().fit(observable_docs)

    test_cases = [
        ("TEST_001", "I signed up for a solo skydiving trip even though I had never tried it before."),
        ("TEST_002", "He immediately believed the stranger who promised to double his money.")
    ]

    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for cid, text in test_cases:
        print(f"\n==================================================")
        print(f"FETCHING RAW OUTPUT FOR {cid}...")
        print(f"==================================================")

        res_bm25 = bm25.search(text, top_k=10)
        res_dense = dense.search(text, top_k=10)

        rrf_scores = {}
        doc_map = {}
        for r, (d, _) in enumerate(res_bm25, 1):
            fid = d["facet_id"]; doc_map[fid] = d; rrf_scores[fid] = rrf_scores.get(fid, 0.0) + (1.0 / (60.0 + r))
        for r, (d, _) in enumerate(res_dense, 1):
            fid = d["facet_id"]; doc_map[fid] = d; rrf_scores[fid] = rrf_scores.get(fid, 0.0) + (1.0 / (60.0 + r))

        sorted_fids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top10_cands = [doc_map[fid] for fid in sorted_fids[:10]]

        facet_lines = [f"- [{d['facet_id']}] {normalize_facet_name(d['normalized_facet'])}: {d.get('definition', d.get('scoring_definition', ''))}" for d in top10_cands]
        facets_text = "\n".join(facet_lines)

        system_prompt = (
            "You are a psychological facet evaluator. Assess if the user's text exhibits any of the given candidate traits.\n"
            "For matching traits, set status to 'scored' and assign a score from 1.0 to 5.0.\n"
            "For unmatched traits, set status to 'abstained' and score to 0.0.\n"
            "Return ONLY a raw JSON array of objects with keys: facet_id, facet_name, score, status, confidence, reasoning."
        )

        user_prompt = f"CONVERSATION TEXT:\n\"{text}\"\n\nCANDIDATE FACETS:\n{facets_text}\n\nOutput JSON array:"

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 1024
        }

        try:
            json_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                COLAB_URL,
                data=json_data,
                headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0", "ngrok-skip-browser-warning": "true"},
                method="POST"
            )
            t0 = time.time()
            with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
                t_ms = (time.time() - t0) * 1000
                resp_body = json.loads(resp.read().decode("utf-8"))
                raw_content = resp_body["choices"][0]["message"]["content"]
                
                print(f"\n--- RAW UN-TRUNCATED QWEN OUTPUT ({round(t_ms)}ms) ---")
                print(repr(raw_content))
                print("--- END RAW OUTPUT ---\n")

                parsed, state = extract_json_results(raw_content)
                print(f"PARSER RESULT: state='{state}', item_count={len(parsed)}")
                print(f"PARSED ITEMS: {json.dumps(parsed, indent=2)}")

        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    dump_raw_output()
