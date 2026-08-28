import json
import pandas as pd

df = pd.read_csv("facet_evaluation_test_set_150.csv")
facets = json.load(open("data/processed/enriched_facets.json", encoding="utf-8"))

cat_by_norm = {f["normalized_facet"].strip().lower(): f for f in facets}
cat_by_raw = {f["raw_facet"].strip().lower(): f for f in facets}

match_count = 0
unmatched = []

for idx, row in df.iterrows():
    ef = str(row["expected_facet"]).strip()
    ef_lower = ef.lower()
    ef_clean = ef.rstrip(":").strip().lower()

    found = cat_by_norm.get(ef_lower) or cat_by_raw.get(ef_lower) or cat_by_norm.get(ef_clean) or cat_by_raw.get(ef_clean)

    if found:
        match_count += 1
    else:
        unmatched.append((row["test_id"], ef))

print(f"Matched Facets: {match_count} / {len(df)}")
print(f"Unmatched Facets Count: {len(unmatched)}")
print("Sample Unmatched Facets:")
for tid, ef in unmatched[:15]:
    print(f"  - [{tid}] '{ef}'")
