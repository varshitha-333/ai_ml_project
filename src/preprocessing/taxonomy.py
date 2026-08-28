"""
Taxonomy Classification Engine for Facets Catalog.

Implements rule-based, conservative classification of raw and normalized facets into distinct
types, assigning observability flags, privacy/sensitivity tiers, scoring definitions,
scoring anchors, and explicit abstention reasons.
"""

import json
import re
from typing import Dict, Any, Optional


# Keywords for taxonomy classification
MEDICAL_KEYWORDS = [
    r'\blevel\b', r'\bhormone\b', r'basophil count', r'blood count', r'cell count',
    r'\bgene\b', r'\bpolygenic\b', r'\bdiagnosis\b', r'\bmacronutrient\b', r'\bapnea\b',
    r'\bmetabolic\b', r'\bserotonin\b', r'\bimmune-response\b', r'\bfsh\b',
    r'\bparathyroid\b', r'\bchromatin\b', r'chronic pain', r'caffeine sensitivity'
]

EXTERNAL_BIOGRAPHICAL_KEYWORDS = [
    r'hours/week', r'km/week', r'time/day', r'time outdoors', r'sessions / year',
    r'sessions/year', r'cycles per year', r'months', r'stamps', r'subscriber',
    r'contributions', r'visits/year', r'years', r'%\b', r'commute', r'snacking',
    r'dietary', r'transport', r'security-system', r'food sourcing', r'passport',
    r'drug-use history', r'cloud-backup', r'peer-to-peer', r'open-source',
    r'museum visits', r'choir participation', r'pet-enrichment', r'dance rehearsal',
    r'robotics-interaction', r'public-transport', r'caffeine intake', r'wake-time consistency',
    r'count\b', r'frequency\b', r'hours\b'
]

ESOTERIC_SPIRITUAL_KEYWORDS = [
    r'i ching', r'hexagram', r'astrology', r'rising sign', r'sufi', r'quran',
    r'bahá', r'bah[aá]', r'reiki', r'gnostic', r'jewish', r'sukkot', r'hindu',
    r'yoga', r'vrata', r'bhagavad-gita', r'zohar', r'seerah', r'sikh', r'kirtan',
    r'tiferet', r'channeling', r'shabbat', r'dhikr', r'aura-color', r'satya',
    r'holiness', r'pilgrimage', r'spiritual'
]

COGNITIVE_SKILL_KEYWORDS = [
    r'numerical reasoning', r'spatial perception', r'alphanumeric', r'psychomotor',
    r'working memory', r'executive-function', r'iq\b', r'intelligence quotient',
    r'mental arithmetic', r'spelling accuracy', r'sequential memory', r'auditory memory',
    r'divided attention', r'material properties', r'anatomy knowledge', r'numeric filing',
    r'alphabetical filing'
]

CLINICAL_PSYCHOLOGICAL_KEYWORDS = [
    r'depression', r'hypomania', r'hysteria', r'psychoticism', r'neuroticism',
    r'burnout', r'identity diffusion', r'ego dissolution', r'physical-violence exposure',
    r'introjection', r'acculturative stress', r'operant-learning'
]


def classify_facet(
    raw_facet: str,
    normalized_facet: str,
    clean_meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Classifies a normalized facet into taxonomy types and returns structured metadata.

    Returns:
        Dict[str, Any] containing:
            - facet_type (str)
            - conversation_observable (bool)
            - sensitivity (str)
            - scoring_definition (str)
            - scoring_anchors (Optional[str])
            - abstention_reason (Optional[str])
    """
    if clean_meta.get("is_blank"):
        return {
            "facet_type": "malformed_invalid",
            "conversation_observable": False,
            "sensitivity": "low",
            "scoring_definition": "Malformed or empty facet record.",
            "scoring_anchors": None,
            "abstention_reason": "Empty or non-string facet entry."
        }

    if clean_meta.get("is_header"):
        return {
            "facet_type": "header_label",
            "conversation_observable": False,
            "sensitivity": "low",
            "scoring_definition": f"Category header label: '{normalized_facet}'. Not an evaluable individual facet.",
            "scoring_anchors": None,
            "abstention_reason": "Category header or structural section label, not an individual scoreable trait."
        }

    norm_lower = normalized_facet.lower()

    # 1. Esoteric / Spiritual / Ritual
    for pattern in ESOTERIC_SPIRITUAL_KEYWORDS:
        if re.search(pattern, norm_lower):
            return {
                "facet_type": "esoteric_spiritual",
                "conversation_observable": False,
                "sensitivity": "sensitive_personal",
                "scoring_definition": f"Esoteric/spiritual practice or belief metric: '{normalized_facet}'.",
                "scoring_anchors": None,
                "abstention_reason": "Esoteric, ritual, or spiritual metric not objectively evaluable from standard dialogue text."
            }

    # 2. Medical / Health / Lab Biomarker
    for pattern in MEDICAL_KEYWORDS:
        if re.search(pattern, norm_lower):
            return {
                "facet_type": "medical_biomarker",
                "conversation_observable": False,
                "sensitivity": "clinical_medical",
                "scoring_definition": f"Medical, biological, or physiological biomarker: '{normalized_facet}'.",
                "scoring_anchors": None,
                "abstention_reason": "Requires biological, laboratory, or physiological measurement; unobservable from text."
            }

    # 3. External / Biographical / System Logs
    for pattern in EXTERNAL_BIOGRAPHICAL_KEYWORDS:
        if re.search(pattern, norm_lower):
            return {
                "facet_type": "external_biographical",
                "conversation_observable": False,
                "sensitivity": "sensitive_personal",
                "scoring_definition": f"Objective historical, lifestyle, or biographical tracking metric: '{normalized_facet}'.",
                "scoring_anchors": None,
                "abstention_reason": "Requires objective external tracking, sensor data, or historical log verification."
            }

    # 4. Cognitive & Aptitude Skill Testing
    for pattern in COGNITIVE_SKILL_KEYWORDS:
        if re.search(pattern, norm_lower):
            return {
                "facet_type": "cognitive_skill_test",
                "conversation_observable": False,
                "sensitivity": "medium",
                "scoring_definition": f"Standardized cognitive or psychomotor test metric: '{normalized_facet}'.",
                "scoring_anchors": None,
                "abstention_reason": "Requires a standardized psychometric testing task or timed cognitive exam."
            }

    # 5. Clinical / Severe Psychological Trait
    for pattern in CLINICAL_PSYCHOLOGICAL_KEYWORDS:
        if re.search(pattern, norm_lower):
            return {
                "facet_type": "psychological_trait",
                "conversation_observable": False,
                "sensitivity": "high",
                "scoring_definition": f"Clinical psychometric construct: '{normalized_facet}'.",
                "scoring_anchors": None,
                "abstention_reason": "Clinical diagnostic construct requiring accredited psychiatric/psychometric clinical evaluation."
            }

    # 6. Check general non-observable indicators (e.g. specific metadata metrics)
    if any(kw in norm_lower for kw in [' nationality', 'ratio:', 'temperature', 'presence', 'frequency', 'count', 'hours']):
        return {
            "facet_type": "external_biographical",
            "conversation_observable": False,
            "sensitivity": "sensitive_personal",
            "scoring_definition": f"External demographic or environment parameter: '{normalized_facet}'.",
            "scoring_anchors": None,
            "abstention_reason": "External environment or historical parameter not directly observable from conversation."
        }

    # 7. Conversational Observable Trait (Default for behavioral/interpersonal traits)
    anchors = json.dumps({
        "high": f"Demonstrates clear, consistent evidence of high {normalized_facet} through explicit statements, tone, or interaction pattern.",
        "low": f"Demonstrates explicit evidence of low {normalized_facet} or opposing behavioral tendencies in dialogue."
    })

    return {
        "facet_type": "conversational_trait",
        "conversation_observable": True,
        "sensitivity": "medium" if any(w in norm_lower for w in ['hostility', 'rebelliousness', 'dishonesty', 'impudence', 'hatefulness']) else "low",
        "scoring_definition": f"Evaluates conversational text for evidence of '{normalized_facet}' based on language style, tone, expressed sentiment, and interpersonal posture.",
        "scoring_anchors": anchors,
        "abstention_reason": None
    }
