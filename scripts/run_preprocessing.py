"""
Execution script for running the facet preprocessing pipeline.
"""

import sys
from pathlib import Path
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import load_raw_facets, preprocess_facets_dataframe


def main():
    raw_path = PROJECT_ROOT / "data" / "raw" / "Facets Assignment.csv"
    processed_csv = PROJECT_ROOT / "data" / "processed" / "enriched_facets.csv"
    processed_json = PROJECT_ROOT / "data" / "processed" / "enriched_facets.json"

    print(f"Loading raw facets dataset from: {raw_path}")
    raw_df = load_raw_facets(raw_path)

    print("Running preprocessing & taxonomy classification pipeline...")
    enriched_df, stats = preprocess_facets_dataframe(raw_df)

    # Ensure processed directory exists
    processed_csv.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV and JSON outputs
    enriched_df.to_csv(processed_csv, index=False, encoding="utf-8")
    print(f"Successfully saved enriched CSV to: {processed_csv}")

    records = enriched_df.to_dict(orient="records")
    with open(processed_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Successfully saved enriched JSON to: {processed_json}")

    print("\n--- PREPROCESSING SUMMARY ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
