"""
Script to generate dataset audit reports in JSON and Markdown formats.
"""

import sys
from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_audit_report():
    processed_csv = PROJECT_ROOT / "data" / "processed" / "enriched_facets.csv"
    report_json_path = PROJECT_ROOT / "outputs" / "audit_report.json"
    report_md_path = PROJECT_ROOT / "outputs" / "audit_report.md"

    if not processed_csv.exists():
        raise FileNotFoundError(f"Processed dataset not found at {processed_csv}. Run preprocessing script first.")

    df = pd.read_csv(processed_csv)

    total_rows = len(df)
    valid_rows = int((df["quality_flag"] == "valid").sum())
    header_rows = int((df["quality_flag"] == "header_row").sum())
    prefix_stripped_rows = int((df["quality_flag"] == "number_prefix_stripped").sum())
    encoding_repaired_rows = int((df["quality_flag"] == "encoding_repaired").sum())
    camelcase_reformatted_rows = int((df["quality_flag"] == "camelcase_reformatted").sum())
    malformed_rows = int(df["quality_flag"].isin(["blank", "malformed"]).sum())
    duplicate_rows = int((df["quality_flag"] == "duplicate").sum())

    observable_count = int(df["conversation_observable"].sum())
    unobservable_count = total_rows - observable_count

    facet_type_dist = df["facet_type"].value_counts().to_dict()
    sensitivity_dist = df["sensitivity"].value_counts().to_dict()
    quality_flag_dist = df["quality_flag"].value_counts().to_dict()

    report_data = {
        "dataset_metadata": {
            "source_file": "Facets Assignment.csv",
            "total_records": total_rows,
            "processed_records": total_rows
        },
        "data_quality_metrics": {
            "total_rows": total_rows,
            "valid_rows": valid_rows,
            "header_rows": header_rows,
            "number_prefix_stripped": prefix_stripped_rows,
            "encoding_repaired": encoding_repaired_rows,
            "camelcase_reformatted": camelcase_reformatted_rows,
            "malformed_suspicious_rows": malformed_rows,
            "duplicates": duplicate_rows
        },
        "observability_metrics": {
            "conversation_observable_count": observable_count,
            "conversation_observable_pct": round(observable_count / total_rows * 100, 2),
            "unobservable_count": unobservable_count,
            "unobservable_pct": round(unobservable_count / total_rows * 100, 2)
        },
        "distribution_metrics": {
            "facet_type_distribution": facet_type_dist,
            "sensitivity_distribution": sensitivity_dist,
            "quality_flag_distribution": quality_flag_dist
        }
    }

    # Save JSON Report
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Audit report JSON saved to: {report_json_path}")

    # Generate Markdown Report
    md_content = f"""# Facet Catalog Dataset Audit Report

## 1. Summary Metrics
- **Total Records Ingested**: `{total_rows}`
- **Valid Clean Facets**: `{valid_rows}` ({round(valid_rows/total_rows*100, 1)}%)
- **Category Header / Section Labels**: `{header_rows}`
- **Numeric Catalog Index Prefixes Stripped**: `{prefix_stripped_rows}`
- **Encoding Anomaly Repairs**: `{encoding_repaired_rows}`
- **CamelCase Words Reformatted**: `{camelcase_reformatted_rows}`
- **Malformed / Suspicious Rows**: `{malformed_rows}`
- **Duplicate Entries**: `{duplicate_rows}`

---

## 2. Observability Analysis
| Status | Count | Percentage | Primary Handling Strategy |
| :--- | :---: | :---: | :--- |
| **Conversation Observable** | `{observable_count}` | `{round(observable_count/total_rows*100, 1)}%` | Candidate for LLM retrieval and score evaluation in Part 2/3. |
| **Unobservable (Abstain)** | `{unobservable_count}` | `{round(unobservable_count/total_rows*100, 1)}%` | Explicitly abstained with defined `abstention_reason`. |

---

## 3. Facet Type Taxonomy Distribution
| Facet Type | Count | Observability | Sensitivity Tiers |
| :--- | :---: | :---: | :--- |
| `conversational_trait` | `{facet_type_dist.get('conversational_trait', 0)}` | Observable | Low / Medium |
| `esoteric_spiritual` | `{facet_type_dist.get('esoteric_spiritual', 0)}` | Unobservable | Sensitive Personal |
| `external_biographical` | `{facet_type_dist.get('external_biographical', 0)}` | Unobservable | Sensitive Personal |
| `header_label` | `{facet_type_dist.get('header_label', 0)}` | Unobservable | Low |
| `medical_biomarker` | `{facet_type_dist.get('medical_biomarker', 0)}` | Unobservable | Clinical Medical |
| `cognitive_skill_test` | `{facet_type_dist.get('cognitive_skill_test', 0)}` | Unobservable | Medium |
| `psychological_trait` | `{facet_type_dist.get('psychological_trait', 0)}` | Unobservable | High |

---

## 4. Privacy & Sensitivity Tiers
| Sensitivity Tier | Count | Description |
| :--- | :---: | :--- |
| `low` | `{sensitivity_dist.get('low', 0)}` | Standard observable behavioral traits or non-sensitive labels. |
| `sensitive_personal` | `{sensitivity_dist.get('sensitive_personal', 0)}` | Personal lifestyle logs, travel metrics, or religious practices. |
| `clinical_medical` | `{sensitivity_dist.get('clinical_medical', 0)}` | Protected health metrics, lab blood counts, genetic markers. |
| `medium` | `{sensitivity_dist.get('medium', 0)}` | Aptitude test scores or sensitive behavioral attributes. |
| `high` | `{sensitivity_dist.get('high', 0)}` | Clinical psychometric diagnostic constructs (e.g. MMPI subscales). |

---

## 5. Sample Enriched Entries

### Sample Observable Trait (`conversational_trait`)
- **Raw Facet**: `Hesitation`
- **Normalized Facet**: `Hesitation`
- **Facet Type**: `conversational_trait`
- **Conversation Observable**: `True`
- **Quality Flag**: `valid`
- **Scoring Anchors**: High (+1) = Clear evidence of hesitation; Low (-1) = High decisiveness.

### Sample Unobservable Medical Biomarker (`medical_biomarker`)
- **Raw Facet**: `Parathyroid-hormone level`
- **Normalized Facet**: `Parathyroid-hormone level`
- **Facet Type**: `medical_biomarker`
- **Conversation Observable**: `False`
- **Abstention Reason**: `Requires biological, laboratory, or physiological measurement; unobservable from text.`

### Sample Category Header Row (`header_label`)
- **Raw Facet**: `Democratic Leadership:`
- **Normalized Facet**: `Democratic Leadership`
- **Facet Type**: `header_label`
- **Conversation Observable**: `False`
- **Quality Flag**: `header_row`
- **Abstention Reason**: `Category header or structural section label, not an individual scoreable trait.`
"""

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Audit report Markdown saved to: {report_md_path}")


if __name__ == "__main__":
    generate_audit_report()
