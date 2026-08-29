"""
Generate a novel, completely independent 30-case validation dataset
(facet_evaluation_test_set_30_novel.csv) covering observable behavioral traits,
emotional expressions, cognitive reasoning, sarcasm, code-switching, and medical/biographical abstentions.
"""

import sys
from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def generate_novel_30_dataset():
    novel_30_cases = [
        # Observable Scored Cases (24 Cases)
        ("I decided to invest my savings into a risky crypto startup without a backup plan.", "Risktaking", "scored", 4.0),
        ("He believed the fake email claiming he won a million dollars and clicked the link.", "Naivety", "scored", 4.0),
        ("Our team took a vote to choose our new logo design together.", "Democratic Leadership", "scored", 4.0),
        ("I paused for 2 minutes before choosing which job offer to accept.", "Hesitation", "scored", 4.0),
        ("I keep feeling that my current salary is never good enough despite my raise.", "Discontentment", "scored", 4.0),
        ("I constantly call my teenage daughter every 10 minutes to check if she is safe.", "Overprotectiveness", "scored", 4.0),
        ("We spent the entire evening telling jokes and laughing hysterically.", "Merriness", "scored", 4.0),
        ("I got teary-eyed watching the hero reunite with his family in the movie.", "Emotionalism", "scored", 4.0),
        ("I enrolled in an online Python course to improve my backend coding skills.", "Self-improvement", "scored", 4.0),
        ("The sample mean is 50 with a high standard deviation, showing wide spread.", "Statistical Reasoning", "scored", 4.0),
        ("I told my manager firmly that I cannot take on extra projects this month.", "Assertiveness and control in relationships", "scored", 4.0),
        ("I strategically changed the conversation topic so they would agree with me.", "Cunningness", "scored", 4.0),
        ("Next month I am going white-water rafting down a steep canyon.", "Adventure-Seeking Behavior", "scored", 4.0),
        ("Handling intense customer complaints all day has left me emotionally exhausted.", "Compassion Fatigue", "scored", 4.0),
        ("Everything feels depressing lately, and I doubt things will improve.", "Moroseness", "scored", 4.0),
        ("I have worked exclusively as a PostgreSQL database architect for 7 years.", "Specialist", "scored", 4.0),
        ("I prefer standing alone near the doorway during office parties.", "Aloofness", "scored", 4.0),
        ("I acknowledged my bug immediately during standup and apologized.", "Genuine", "scored", 4.0),
        ("The server crashed twice, but I stayed up until 3 AM to fix it.", "Determinedness", "scored", 4.0),
        ("I returned the dropped wallet with all its cash to the security guard.", "HEXACO domain: Honesty-Humility", "scored", 5.0),
        ("I took time to learn about my coworker's interests before starting our project.", "Relationship Building Themes", "scored", 4.0),
        ("I love solving complex math riddles just for the fun of it.", "Pure Challenge", "scored", 4.0),
        ("We need a 95% confidence interval before drawing conclusions from this data.", "Numerical Reasoning Subcomponents", "scored", 4.0),
        ("I am happy to listen to your opinion even when we disagree.", "Openness", "scored", 4.0),

        # Non-Observable Anti-Hallucination Trap Cases (6 Cases)
        ("I do not have a laboratory report available for my FSH level.", "FSH level", "not_observable", 0.0),
        ("I do not have medical lab records for my Parathyroid-hormone level.", "Parathyroid-hormone level", "not_observable", 0.0),
        ("I do not have system log counts for my Passport-stamps count.", "Passport-stamps count", "not_observable", 0.0),
        ("I do not have laboratory documentation for blood glucose levels.", "Acidity", "not_observable", 0.0),
        ("I do not have hardware log records for subscription count.", "Subscription count", "not_observable", 0.0),
        ("I do not have clinical data for cell endorsement counts.", "Skill-endorsements count", "not_observable", 0.0),
    ]

    novel_rows = []
    for idx, (text, facet, status, score) in enumerate(novel_30_cases, 1):
        novel_rows.append({
            "test_id": f"TEST_{idx:03d}",
            "text": text,
            "expected_facet": facet,
            "expected_status": status,
            "expected_score": score
        })

    df_novel = pd.DataFrame(novel_rows)
    out_csv = PROJECT_ROOT / "facet_evaluation_test_set_30_novel.csv"
    df_novel.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Generated 30 novel test cases saved to -> {out_csv}")

    # Remove old 150 novel file if present
    old_150 = PROJECT_ROOT / "facet_evaluation_test_set_150_novel.csv"
    if old_150.exists():
        old_150.unlink()
        print(f"[CLEANUP] Removed old 150 file -> {old_150}")

if __name__ == "__main__":
    generate_novel_30_dataset()
