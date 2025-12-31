#!/usr/bin/env python3
"""
Filter CSV file to keep only rows where som_contains_answer is True.

Usage:
    python filter_questions.py <path_to_csv>
"""

import sys
import pandas as pd


def filter_questions_by_som_contains_answer(csv_path):
    """
    Read CSV file and save filtered rows where som_contains_answer is True.
    
    Args:
        csv_path (str): Path to the input CSV file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = {'question', 'answer', 'som_contains_answer'}
        if not required_columns.issubset(df.columns):
            print(f"Error: CSV must contain columns: {required_columns}")
            print(f"Found columns: {set(df.columns)}")
            sys.exit(1)
        
        # Filter rows where som_contains_answer is True
        filtered_df = df[df['som_contains_answer'] == True]
        
        # Save to output file
        output_path = 'filtered_questions_som.csv'
        filtered_df.to_csv(output_path, index=False)
        
        print(f"Successfully filtered {len(filtered_df)} rows from {len(df)} total rows")
        print(f"Output saved to: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python filter_questions.py <path_to_csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    filter_questions_by_som_contains_answer(csv_path)
