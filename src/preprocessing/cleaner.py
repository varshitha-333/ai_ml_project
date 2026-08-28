"""
Facet Text Cleaner and Quality Flag Detector.

Provides pure functions for cleaning raw facet strings, stripping extraneous
formatting artifacts, repairing encoding anomalies, and identifying quality flags.
"""

import re
from typing import Dict, List, Tuple


def repair_encoding_anomalies(text: str) -> Tuple[str, bool]:
    """
    Detects and repairs common character encoding anomalies (mojibake, replacement characters).

    Returns:
        Tuple[str, bool]: (repaired_text, is_repaired_flag)
    """
    if not text:
        return text, False

    original = text
    repaired = text

    # Common unicode replacement / mojibake patterns in dataset
    # En-dash replacement
    repaired = re.sub(r'[\ufffd\x80\x93]+', '–', repaired)
    repaired = repaired.replace('  ', ' – ')
    repaired = repaired.replace(' – ', ' – ')
    
    # Specific known corrupted strings in raw CSV
    repaired = repaired.replace('Bah spiritual metric: Ridvn', "Bahá'í spiritual metric: Ridván")
    repaired = repaired.replace('Bahá’\xad', "Bahá'í")
    repaired = repaired.replace('Religious coping  Negative', 'Religious coping – Negative')
    repaired = repaired.replace('Eightfold Path  Right Intention', 'Eightfold Path – Right Intention')
    repaired = repaired.replace('Cultural Intelligence  Behavioral', 'Cultural Intelligence – Behavioral')
    repaired = repaired.replace('Openness  Artistic', 'Openness – Artistic')

    is_repaired = (repaired != original) or bool(re.search(r'[^\x00-\x7F]', original))
    return repaired, is_repaired


def strip_numeric_prefix(text: str) -> Tuple[str, bool]:
    """
    Strips catalog index prefixes like '800. Sufi practice' -> 'Sufi practice'.

    Returns:
        Tuple[str, bool]: (stripped_text, was_prefix_removed)
    """
    match = re.match(r'^\s*\d{3,4}\.\s*(.+)$', text)
    if match:
        return match.group(1).strip(), True
    return text.strip(), False


def strip_trailing_colon(text: str) -> Tuple[str, bool]:
    """
    Strips trailing colons from category/header entries like 'Democratic Leadership:' -> 'Democratic Leadership'.

    Returns:
        Tuple[str, bool]: (stripped_text, was_colon_removed)
    """
    if text.endswith(':'):
        return text[:-1].strip(), True
    return text, False


def reformat_camelcase(text: str) -> Tuple[str, bool]:
    """
    Splits unspaced camelCase strings like 'SelfEsteem' -> 'Self Esteem', 'HonestyHumility' -> 'Honesty Humility'.

    Returns:
        Tuple[str, bool]: (reformatted_text, was_reformatted)
    """
    # Known concatenated items in the dataset
    known_camelcase = {
        'SelfEsteem': 'Self Esteem',
        'HonestyHumility': 'Honesty Humility',
        'Selfcontrol': 'Self Control',
        'SelfDirectedness': 'Self Directedness'
    }
    
    if text in known_camelcase:
        return known_camelcase[text], True

    # Generic camelCase split if applicable (lowercase followed by uppercase without space)
    split_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    if split_text != text:
        return split_text, True
    return text, False


def clean_facet_text(raw_text: str) -> Dict[str, object]:
    """
    Full cleaning pipeline for a single raw facet text entry.

    Returns dictionary with normalized text and transformation metadata.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {
            "normalized_facet": "",
            "is_blank": True,
            "is_header": False,
            "has_numeric_prefix": False,
            "encoding_repaired": False,
            "camelcase_reformatted": False
        }

    text = raw_text.strip()
    
    # Check header
    text_no_colon, is_header = strip_trailing_colon(text)
    
    # Strip numeric prefix
    text_no_prefix, has_prefix = strip_numeric_prefix(text_no_colon)
    
    # Repair encoding
    repaired_text, encoding_repaired = repair_encoding_anomalies(text_no_prefix)
    
    # Reformat camelCase
    normalized_text, camelcase_reformatted = reformat_camelcase(repaired_text)
    
    return {
        "normalized_facet": normalized_text,
        "is_blank": False,
        "is_header": is_header,
        "has_numeric_prefix": has_prefix,
        "encoding_repaired": encoding_repaired,
        "camelcase_reformatted": camelcase_reformatted
    }


def detect_quality_flags(clean_meta: Dict[str, object], is_duplicate: bool = False) -> str:
    """
    Assigns a primary data quality flag based on cleaning metadata.
    """
    if clean_meta.get("is_blank"):
        return "blank"
    if is_duplicate:
        return "duplicate"
    if clean_meta.get("is_header"):
        return "header_row"
    if clean_meta.get("has_numeric_prefix"):
        return "number_prefix_stripped"
    if clean_meta.get("encoding_repaired"):
        return "encoding_repaired"
    if clean_meta.get("camelcase_reformatted"):
        return "camelcase_reformatted"
    return "valid"
