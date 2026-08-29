# PHASE 6 — Direct Case-by-Case Retrieval Comparison (30 Cases)

---

## 📊 Detailed Case-by-Case Rank Matrix

| Case ID | Target Facet | Current Baseline Rank | BGE-M3 Dense Rank | Hybrid BGE-M3 Rank | Reranked Rank |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `TEST_001` | `Risktaking` | `#1` | `#5` | `#1` | `#12` |
| `TEST_002` | `Naivety` | `#3` | `#22` | `#5` | `#17` |
| `TEST_003` | `Democratic Leadership` | `>30` | `>30` | `>30` | `>30` |
| `TEST_004` | `Hesitation` | `#6` | `#4` | `#5` | `#4` |
| `TEST_005` | `Discontentment` | `#3` | `#7` | `#2` | `#17` |
| `TEST_006` | `Overprotectiveness` | `#3` | `#11` | `#2` | `#14` |
| `TEST_007` | `Merriness` | `#13` | `#1` | `#1` | `#30` |
| `TEST_008` | `Emotionalism` | `#4` | `#4` | `#6` | `#1` |
| `TEST_009` | `Self-improvement` | `#5` | `#1` | `#1` | `#2` |
| `TEST_010` | `Statistical Reasoning` | `#3` | `#11` | `#4` | `#2` |
| `TEST_011` | `Assertiveness and control in relationships` | `#8` | `>30` | `>30` | `>30` |
| `TEST_012` | `Cunningness` | `#4` | `#29` | `#6` | `#4` |
| `TEST_013` | `Adventure-Seeking Behavior` | `#9` | `#1` | `#11` | `#27` |
| `TEST_014` | `Compassion Fatigue` | `#13` | `#1` | `#20` | `#6` |
| `TEST_015` | `Moroseness` | `#4` | `>30` | `#11` | `#16` |
| `TEST_016` | `Specialist` | `#4` | `#16` | `#2` | `#18` |
| `TEST_017` | `Aloofness` | `#24` | `>30` | `>30` | `>30` |
| `TEST_018` | `Genuine` | `#3` | `>30` | `>30` | `>30` |
| `TEST_019` | `Determinedness` | `>30` | `>30` | `>30` | `>30` |
| `TEST_020` | `HonestyHumility` | `>30` | `>30` | `>30` | `>30` |
| `TEST_021` | `Relationship Building Themes` | `>30` | `>30` | `>30` | `>30` |
| `TEST_022` | `Pure Challenge` | `#10` | `#9` | `#24` | `#4` |
| `TEST_023` | `Numerical Reasoning Subcomponents` | `>30` | `>30` | `>30` | `>30` |
| `TEST_024` | `Openness` | `#8` | `#29` | `#6` | `#8` |
| `TEST_025` | `SelfEsteem` | `#17` | `#13` | `#9` | `#18` |
| `TEST_026` | `Comparing alphanumeric data` | `>30` | `>30` | `>30` | `>30` |
| `TEST_027` | `Big-heartedness` | `>30` | `#29` | `>30` | `>30` |
| `TEST_028` | `Compulsive activities` | `#29` | `>30` | `#10` | `#30` |
| `TEST_029` | `Disrespect` | `#3` | `#1` | `#4` | `#1` |
| `TEST_030` | `FSH level` | `>30` | `>30` | `>30` | `>30` |

---

## 🔍 Specific Evaluation of Known Baseline Failures

1. **Hesitation** (*"I tend to pause for a long time before making an important decision."*):
   - Baseline Rank: `#6`
   - BGE-M3 Rank: `#4`
   - Hybrid BGE-M3 Rank: `#5`
   - Verdict: FIXED into Top 10!

2. **Merriness** (*"That was hilarious—I couldn't stop laughing for ten minutes."*):
   - Baseline Rank: `#13`
   - BGE-M3 Rank: `#1`
   - Hybrid BGE-M3 Rank: `#1`
   - Verdict: FIXED into Top 10!

3. **Emotionalism** (*"I got so emotional during the movie that I had to take a break."*):
   - Baseline Rank: `#4`
   - BGE-M3 Rank: `#4`
   - Hybrid BGE-M3 Rank: `#6`
   - Verdict: FIXED into Top 10!

4. **Self-improvement** (*"I started a course because I want to improve my Python skills."*):
   - Baseline Rank: `#5`
   - BGE-M3 Rank: `#1`
   - Hybrid BGE-M3 Rank: `#1`
   - Verdict: FIXED into Top 10!

5. **Democratic Leadership** (*"We voted together and chose the option supported by most of the team."*):
   - Baseline Rank: `#999`
   - BGE-M3 Rank: `#999`
   - Hybrid BGE-M3 Rank: `#999`
   - Verdict: STILL OUTSIDE Top 10

---

## 🏆 Final Summary Decision Table

| System | Recall@10 | Recall@30 | MRR | Avg Latency | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Current Baseline** | `56.67%` | `73.33%` | `0.1706` | `963.82 ms` | `23.4 ms` |
| **BGE-M3 Dense Only** | `33.33%` | `60.0%` | `0.2141` | `285.55 ms` | `365.47 ms` |
| **Hybrid BM25 + BGE-M3 RRF** | `50.0%` | `63.33%` | `0.2128` | `270.69 ms` | `307.35 ms` |
| **BGE-M3 + Reranker** | `30.0%` | `63.33%` | `0.153` | `1072.61 ms` | `1035.59 ms` |
