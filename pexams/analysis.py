import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from collections import Counter
import logging
from typing import Optional, List
from tabulate import tabulate

def parse_q_list(q_str: Optional[str]) -> List[int]:
    """Converts a comma-separated string of question numbers to a sorted list of unique integers."""
    if not q_str:
        return []
    try:
        return sorted(list(set(int(q.strip()) for q in q_str.split(',') if q.strip().isdigit())))
    except ValueError:
        logging.warning(f"Invalid format for question list string: '{q_str}'. Expected comma-separated numbers. Returning empty list.")
        return []

def analyze_results(csv_filepath, max_score, output_dir=".", void_questions_str: Optional[str] = None, void_questions_nicely_str: Optional[str] = None):
    """
    Analyzes exam results from a CSV file, scales scores to 0-10, 
    plots score distribution, and shows statistics.
    Allows for voiding questions or voiding them 'nicely' (only if incorrect/unanswered).
    """
    if not os.path.exists(csv_filepath):
        logging.error(f"Error: CSV file not found at {csv_filepath}")
        return

    try:
        df = pd.read_csv(csv_filepath)
        logging.info(f"Successfully loaded {csv_filepath}")
    except Exception as e:
        logging.error(f"Error reading CSV file {csv_filepath}: {e}")
        return

    if 'score' not in df.columns:
        logging.error(f"Error: 'score' column not found in {csv_filepath}.")
        return

    df['score_numeric'] = pd.to_numeric(df['score'], errors='coerce')
    
    original_rows = len(df)
    df.dropna(subset=['score_numeric'], inplace=True)
    if len(df) < original_rows:
        logging.warning(f"Dropped {original_rows - len(df)} rows due to non-numeric 'score' values.")

    if df.empty:
        logging.error("No valid numeric data in 'score' column after cleaning.")
        return
        
    # For pexams, the score is already the count of correct answers.
    # We need to know the penalty for incorrect answers to adjust for voiding.
    # Assuming a penalty of -1/3 for now, as it's a common case.
    # This part is more complex than in rexams because we don't have per-question points.
    # A simplification: for voided questions, we assume they give 1 point if correct.
    # We don't have information about incorrect answers to add back penalties.
    # This is a limitation of the current pexams CSV format.
    # Let's proceed with a simplified voiding logic.
    
    logging.warning("Simplified 'void' logic is being used. It assumes each question is worth 1 point and does not handle negative marking for voiding.")

    void_q_list = parse_q_list(void_questions_str)
    
    # We can't implement 'void_nicely' without per-question results in the CSV.
    if void_questions_nicely_str:
        logging.warning("'void_nicely' is not supported with the current CSV format from pexams. Ignoring.")

    adjustments_made = bool(void_q_list)
    
    df['score_adjusted'] = df['score_numeric'].copy()
    max_score_adjusted = float(max_score)

    if adjustments_made:
        logging.info(f"Voiding questions: {void_q_list}. Max score will be reduced.")
        # We can't adjust student scores without knowing which they got right.
        # The best we can do is adjust the max score.
        max_score_adjusted -= len(void_q_list)
        logging.info(f"Adjusted max score is now: {max_score_adjusted}")

    df['mark'] = (df['score_adjusted'] / max_score_adjusted) * 10 if max_score_adjusted > 0 else 0
    df['mark_clipped'] = np.clip(df['mark'], 0, 10)

    print("\n--- Descriptive Statistics for Marks (0-10 scale) ---")
    stats = df['mark_clipped'].describe()
    print(stats)
    
    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    df['mark_binned_for_plot'] = np.floor(df['mark_clipped'].fillna(0) + 0.5).astype(int)
    score_counts = Counter(df['mark_binned_for_plot'])
    all_possible_scores = np.arange(0, 11)
    frequencies = [score_counts.get(s, 0) for s in all_possible_scores]

    plt.bar(all_possible_scores, frequencies, width=1.0, edgecolor='black', align='center', color='skyblue')

    ax.set_title(f'Distribution of Exam Marks (Scaled to 0-10 from Max Raw: {max_score_adjusted})', fontsize=15)
    ax.set_xlabel('Mark (0-10 Scale)', fontsize=12)
    ax.set_ylabel('Number of Students', fontsize=12)
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_xlim(-0.5, 10.5)

    if max(frequencies, default=0) > 0:
        ax.set_ylim(top=max(frequencies) * 1.1)
    else:
        ax.set_ylim(top=1)

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    mean_mark = df['mark_clipped'].mean()
    median_mark = df['mark_clipped'].median()
    ax.axvline(mean_mark, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_mark:.2f}')
    ax.axvline(median_mark, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_mark:.2f}')
    ax.legend()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    plot_filename = os.path.join(output_dir, "mark_distribution_0_10.png")
    try:
        plt.savefig(plot_filename)
        logging.info(f"\nPlot saved to {os.path.abspath(plot_filename)}")
    except Exception as e:
        logging.error(f"Error saving plot: {e}")

    # --- Print Student Marks ---
    print("\n--- Student Marks (0-10 Scale) ---")
    
    results_to_print_df = df[['student_id', 'student_name', 'mark_clipped']].copy()
    results_to_print_df.rename(columns={'mark_clipped': 'mark'}, inplace=True)
    
    # Save to a new CSV
    final_csv_path = os.path.join(output_dir, "final_marks.csv")
    results_to_print_df.to_csv(final_csv_path, index=False)
    logging.info(f"Final marks saved to {os.path.abspath(final_csv_path)}")
    
    # Print to console
    print(tabulate(results_to_print_df, headers='keys', tablefmt='psql', floatfmt=".2f"))
