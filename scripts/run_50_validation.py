"""
Script for Final End-to-End 50-Case External Validation Experiment.
Evaluates exactly 50 curated test cases from facet_evaluation_test_set_50.csv.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_150_validation import run_150_validation

if __name__ == "__main__":
    run_150_validation()
