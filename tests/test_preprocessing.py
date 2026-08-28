"""
Unit & Integration Tests for Facet Preprocessing & Taxonomy Classification.
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.cleaner import (
    clean_facet_text,
    detect_quality_flags,
    strip_numeric_prefix,
    strip_trailing_colon,
    reformat_camelcase,
    repair_encoding_anomalies
)
from src.preprocessing.taxonomy import classify_facet
from src.preprocessing.pipeline import preprocess_facets_dataframe, load_raw_facets


def test_raw_vs_normalized_preservation():
    raw_str = "800. Sufi practice: Sufi retreat attendance count:"
    meta = clean_facet_text(raw_str)
    
    assert raw_str == "800. Sufi practice: Sufi retreat attendance count:"
    assert meta["normalized_facet"] == "Sufi practice: Sufi retreat attendance count"
    assert meta["has_numeric_prefix"] is True
    assert meta["is_header"] is True


def test_trailing_colon_header_detection():
    raw_str = "Democratic Leadership:"
    meta = clean_facet_text(raw_str)
    quality_flag = detect_quality_flags(meta)
    taxonomy = classify_facet(raw_str, meta["normalized_facet"], meta)

    assert meta["is_header"] is True
    assert quality_flag == "header_row"
    assert taxonomy["facet_type"] == "header_label"
    assert taxonomy["conversation_observable"] is False
    assert "Category header" in taxonomy["abstention_reason"]


def test_numeric_prefix_stripping():
    raw_str = "644. Spiritual virtue: Humility practice index"
    text, stripped = strip_numeric_prefix(raw_str)

    assert stripped is True
    assert text == "Spiritual virtue: Humility practice index"


def test_encoding_repair():
    raw_str = "516. Religious coping  Negative"
    repaired, is_repaired = repair_encoding_anomalies(raw_str)

    assert is_repaired is True
    assert "–" in repaired


def test_camelcase_reformatting():
    raw_str = "SelfEsteem"
    reformatted, is_ref = reformat_camelcase(raw_str)

    assert is_ref is True
    assert reformatted == "Self Esteem"


def test_medical_biomarker_classification():
    raw_str = "Parathyroid-hormone level"
    meta = clean_facet_text(raw_str)
    taxonomy = classify_facet(raw_str, meta["normalized_facet"], meta)

    assert taxonomy["facet_type"] == "medical_biomarker"
    assert taxonomy["conversation_observable"] is False
    assert taxonomy["sensitivity"] == "clinical_medical"
    assert "laboratory" in taxonomy["abstention_reason"].lower()


def test_external_biographical_classification():
    raw_str = "Passport-stamps count"
    meta = clean_facet_text(raw_str)
    taxonomy = classify_facet(raw_str, meta["normalized_facet"], meta)

    assert taxonomy["facet_type"] == "external_biographical"
    assert taxonomy["conversation_observable"] is False
    assert taxonomy["sensitivity"] == "sensitive_personal"


def test_conversational_trait_classification():
    raw_str = "Hesitation"
    meta = clean_facet_text(raw_str)
    taxonomy = classify_facet(raw_str, meta["normalized_facet"], meta)

    assert taxonomy["facet_type"] == "conversational_trait"
    assert taxonomy["conversation_observable"] is True
    assert taxonomy["scoring_anchors"] is not None
    assert taxonomy["abstention_reason"] is None


def test_pipeline_end_to_end():
    raw_data = pd.DataFrame({
        "raw_facet": [
            "Hesitation",
            "Democratic Leadership:",
            "800. Sufi practice: Sufi retreat attendance count",
            "Parathyroid-hormone level",
            "SelfEsteem"
        ]
    })

    enriched_df, stats = preprocess_facets_dataframe(raw_data)

    assert len(enriched_df) == 5
    assert stats["total_rows"] == 5
    assert stats["header_rows"] == 1
    assert stats["observable_rows"] == 2  # Hesitation, SelfEsteem
    assert "facet_id" in enriched_df.columns
    assert "scoring_definition" in enriched_df.columns
    assert enriched_df.iloc[0]["facet_id"] == "FACET_001"
