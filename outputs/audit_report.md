# Facet Catalog Dataset Audit Report

## 1. Summary Metrics
- **Total Records Ingested**: `399`
- **Valid Clean Facets**: `333` (83.5%)
- **Category Header / Section Labels**: `30`
- **Numeric Catalog Index Prefixes Stripped**: `31`
- **Encoding Anomaly Repairs**: `2`
- **CamelCase Words Reformatted**: `3`
- **Malformed / Suspicious Rows**: `0`
- **Duplicate Entries**: `0`

---

## 2. Observability Analysis
| Status | Count | Percentage | Primary Handling Strategy |
| :--- | :---: | :---: | :--- |
| **Conversation Observable** | `253` | `63.4%` | Candidate for LLM retrieval and score evaluation in Part 2/3. |
| **Unobservable (Abstain)** | `146` | `36.6%` | Explicitly abstained with defined `abstention_reason`. |

---

## 3. Facet Type Taxonomy Distribution
| Facet Type | Count | Observability | Sensitivity Tiers |
| :--- | :---: | :---: | :--- |
| `conversational_trait` | `253` | Observable | Low / Medium |
| `esoteric_spiritual` | `32` | Unobservable | Sensitive Personal |
| `external_biographical` | `37` | Unobservable | Sensitive Personal |
| `header_label` | `30` | Unobservable | Low |
| `medical_biomarker` | `20` | Unobservable | Clinical Medical |
| `cognitive_skill_test` | `15` | Unobservable | Medium |
| `psychological_trait` | `12` | Unobservable | High |

---

## 4. Privacy & Sensitivity Tiers
| Sensitivity Tier | Count | Description |
| :--- | :---: | :--- |
| `low` | `278` | Standard observable behavioral traits or non-sensitive labels. |
| `sensitive_personal` | `69` | Personal lifestyle logs, travel metrics, or religious practices. |
| `clinical_medical` | `20` | Protected health metrics, lab blood counts, genetic markers. |
| `medium` | `20` | Aptitude test scores or sensitive behavioral attributes. |
| `high` | `12` | Clinical psychometric diagnostic constructs (e.g. MMPI subscales). |

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
