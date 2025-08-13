#!/usr/bin/env python3
"""
Quick Sample Evaluation Script
Runs evaluation on just 100 questions for fast results
"""

import subprocess
import sys

def main():
    print("Starting sample evaluation with 100 questions...")
    
    # Run the evaluation with sample size
    cmd = [
        "python3", "scale_up_evaluation.py",
        "--max_questions", "100",
        "--workers", "2",  # Use fewer workers for sample
        "--csv_file", "results/retrieved_contexts.csv",
        "--qa_file", "questions_answers.xlsx"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("Sample evaluation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
