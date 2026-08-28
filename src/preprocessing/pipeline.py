"""
Preprocessing Pipeline for Facets Dataset Enrichment.

Loads raw CSV files, applies cleaning and normalization, runs taxonomy classification,
generates unique facet IDs, and produces an enriched pandas DataFrame.
"""

from pathlib import Path
from typing import Dict, Union, Tuple
import pandas as pd

from src.preprocessing.cleaner import clean_facet_text, detect_quality_flags
from src.preprocessing.taxonomy import classify_facet


def load_raw_facets(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads raw CSV dataset with fallback encoding handling.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {path}")

    # Attempt utf-8 first, fallback to cp1252/latin1 if encoding issues arise
    for enc in ["utf-8", "cp1252", "latin1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            if "Facets" in df.columns or "facets" in df.columns:
                col_name = "Facets" if "Facets" in df.columns else "facets"
                df = df.rename(columns={col_name: "raw_facet"})
                return df
        except Exception:
            continue

    raise ValueError(f"Unable to read CSV file {path} with standard encodings.")


def preprocess_facets_dataframe(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Applies cleaning, quality flagging, taxonomy classification, and schema formatting
    to the input DataFrame.

    Returns:
        Tuple[pd.DataFrame, Dict[str, int]]: (enriched_dataframe, stats_summary)
    """
    records = []
    seen_raw = set()
    seen_normalized = set()
    
    duplicate_count = 0
    malformed_count = 0
    header_count = 0
    observable_count = 0

    for idx, row in raw_df.iterrows():
        raw_val = row["raw_facet"]
        raw_str = str(raw_val) if pd.notna(raw_val) else ""

        # Clean text
        clean_meta = clean_facet_text(raw_str)
        norm_val = clean_meta["normalized_facet"]

        # Duplicate check
        is_dup = (raw_str in seen_raw) or (norm_val in seen_normalized and bool(norm_val))
        if is_dup:
            duplicate_count += 1

        seen_raw.add(raw_str)
        if norm_val:
            seen_normalized.add(norm_val)

        # Quality Flag
        quality_flag = detect_quality_flags(clean_meta, is_duplicate=is_dup)
        if quality_flag == "blank" or quality_flag == "malformed":
            malformed_count += 1
        elif quality_flag == "header_row":
            header_count += 1

        # Classify Taxonomy
        taxonomy_meta = classify_facet(raw_str, norm_val, clean_meta)
        if taxonomy_meta["conversation_observable"]:
            observable_count += 1

        facet_id = f"FACET_{idx + 1:03d}"

        records.append({
            "facet_id": facet_id,
            "raw_facet": raw_str,
            "normalized_facet": norm_val,
            "facet_type": taxonomy_meta["facet_type"],
            "conversation_observable": taxonomy_meta["conversation_observable"],
            "sensitivity": taxonomy_meta["sensitivity"],
            "scoring_definition": taxonomy_meta["scoring_definition"],
            "scoring_anchors": taxonomy_meta["scoring_anchors"],
            "abstention_reason": taxonomy_meta["abstention_reason"],
            "quality_flag": quality_flag,
        })

    enriched_df = pd.DataFrame(records)

    stats = {
        "total_rows": len(raw_df),
        "valid_rows": len(enriched_df[enriched_df["quality_flag"] == "valid"]),
        "header_rows": header_count,
        "malformed_rows": malformed_count,
        "duplicates": duplicate_count,
        "observable_rows": observable_count,
        "unobservable_rows": len(raw_df) - observable_count
    }

    return enriched_df, stats
