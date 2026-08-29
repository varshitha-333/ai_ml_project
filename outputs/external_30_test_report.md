# REAL QWEN 30-CASE EXTERNAL VALIDATION REPORT

**BACKEND MODE**: `REMOTE`  
**MODEL**: `Qwen/Qwen2.5-7B-Instruct`  
**RETRIEVAL K**: `10`  
**TOTAL CASES**: `30`  
**STATUS ACCURACY**: `20.0%` (6/30)  
**SCORED FACETS**: `6`  
**ABSTAINED FACETS**: `24`  
**FALSE SCORING RATE**: `0.0%` (Zero Hallucination Target)  
**AVERAGE LATENCY**: `26107.02 ms`

---

## 📊 Per-Case Evaluation Results

| Case ID | Expected Trait | Expected Status | Predicted Status | Score | Verdict | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `TEST_001` | `Risktaking` | `scored` | `scored` | `5` | ✅ PASS | `54910.31 ms` |
| `TEST_002` | `Naivety` | `scored` | `scored` | `3` | ✅ PASS | `41732.64 ms` |
| `TEST_003` | `Democratic Leadership:` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `26343.02 ms` |
| `TEST_004` | `Hesitation` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `27263.57 ms` |
| `TEST_005` | `Discontentment` | `scored` | `scored` | `4` | ✅ PASS | `32686.3 ms` |
| `TEST_006` | `Overprotectiveness` | `scored` | `scored` | `3` | ✅ PASS | `27229.15 ms` |
| `TEST_007` | `Merriness` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `27536.3 ms` |
| `TEST_008` | `Emotionalism` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `26770.83 ms` |
| `TEST_009` | `Self-improvement` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `26890.57 ms` |
| `TEST_010` | `Statistical Reasoning` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `26774.17 ms` |
| `TEST_011` | `Assertiveness and control in relationships` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `26469.44 ms` |
| `TEST_012` | `Cunningness` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `26437.32 ms` |
| `TEST_013` | `Adventure-Seeking Behavior` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `26711.01 ms` |
| `TEST_014` | `Compassion Fatigue` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `26638.18 ms` |
| `TEST_015` | `Moroseness` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `26811.28 ms` |
| `TEST_016` | `Specialist` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `26407.68 ms` |
| `TEST_017` | `Aloofness` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `26846.79 ms` |
| `TEST_018` | `Genuine` | `scored` | `scored` | `5` | ✅ PASS | `26615.56 ms` |
| `TEST_019` | `Determinedness` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `26657.49 ms` |
| `TEST_020` | `HonestyHumility:` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `26375.1 ms` |
| `TEST_021` | `Relationship Building Themes:` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `26746.19 ms` |
| `TEST_022` | `Pure Challenge` | `scored` | `inference_error` | `null` | ❌ FAIL | `3879.54 ms` |
| `TEST_023` | `Numerical Reasoning Subcomponents:` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `2936.86 ms` |
| `TEST_024` | `Openness` | `scored` | `inference_error` | `null` | ❌ FAIL | `3450.99 ms` |
| `TEST_025` | `SelfEsteem` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `28428.1 ms` |
| `TEST_026` | `Comparing alphanumeric data` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `27340.15 ms` |
| `TEST_027` | `Big-heartedness` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `26531.17 ms` |
| `TEST_028` | `Compulsive activities` | `scored` | `insufficient_evidence` | `null` | ❌ FAIL | `26716.03 ms` |
| `TEST_029` | `Disrespect` | `scored` | `scored` | `3` | ✅ PASS | `26600.3 ms` |
| `TEST_030` | `FSH level` | `not_observable` | `insufficient_evidence` | `null` | ❌ FAIL | `26474.54 ms` |
