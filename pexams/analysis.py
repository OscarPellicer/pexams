import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from collections import Counter
import logging
from typing import Optional, List
from tabulate import tabulate
from matplotlib.patches import Patch

def _plot_answer_distribution(df, solutions_per_model, output_dir):
    """
    Plots the distribution of answers for each question in a single grouped bar chart,
    normalized to model 1's answer order.
    """
    # Assuming the first model key is the reference (e.g., "1")
    ref_model_key = sorted(solutions_per_model.keys())[0]
    ref_solutions = solutions_per_model[ref_model_key]
    
    # Create a mapping from option text to the reference index for each question
    option_text_to_ref_idx = {}
    for q_id, q_data in ref_solutions.items():
        option_text_to_ref_idx[q_id] = {opt['text']: i for i, opt in enumerate(q_data['options'])}

    # Translate all student answers to the reference model's option indexing
    all_answers_translated = []
    for _, row in df.iterrows():
        model_id = str(row['model_id'])
        if model_id not in solutions_per_model:
            continue
        
        current_model_solutions = solutions_per_model[model_id]
        
        for q_num_str, ans_char in row.items():
            if not q_num_str.startswith('answer_'):
                continue
            
            q_id = int(q_num_str.split('_')[1])
            if q_id not in current_model_solutions or not isinstance(ans_char, str) or ans_char == 'NA':
                continue

            # Convert character answer to index (A=0, B=1, ...)
            ans_idx = ord(ans_char) - ord('A')
            
            # Get the text of the option the student chose
            try:
                chosen_option_text = current_model_solutions[q_id]['options'][ans_idx]['text']
            except IndexError:
                continue

            # Find the corresponding index in the reference model
            if q_id in option_text_to_ref_idx and chosen_option_text in option_text_to_ref_idx[q_id]:
                ref_idx = option_text_to_ref_idx[q_id][chosen_option_text]
                all_answers_translated.append({'question_id': q_id, 'ref_answer_idx': ref_idx})

    if not all_answers_translated:
        logging.warning("Could not generate answer distribution plot: No valid translated answers found.")
        return

    translated_df = pd.DataFrame(all_answers_translated)
    
    question_ids = sorted(ref_solutions.keys())
    num_questions = len(question_ids)
    
    max_num_options = 0
    if ref_solutions:
        max_num_options = max(len(q_data['options']) for q_data in ref_solutions.values())

    answer_counts_by_q = {
        q_id: translated_df[translated_df['question_id'] == q_id]['ref_answer_idx'].value_counts()
        for q_id in question_ids
    }

    fig, ax = plt.subplots(figsize=(max(15, num_questions * 2), 8))
    x = np.arange(num_questions)
    width = 0.8 / max_num_options if max_num_options > 0 else 0.8

    for i in range(max_num_options):
        counts = [answer_counts_by_q[q_id].get(i, 0) for q_id in question_ids]
        offset = (i - (max_num_options - 1) / 2) * width
        
        colors = []
        for q_id in question_ids:
            correct_idx = ref_solutions[q_id]['correct_answer_index']
            # Only add a bar if this option index is valid for the question
            if i < len(ref_solutions[q_id]['options']):
                colors.append('green' if i == correct_idx else 'red')
            else:
                # This is a placeholder, this bar won't be plotted
                colors.append('none')

        # Filter positions, counts, and colors for valid options
        valid_positions = [x[j] + offset for j, q_id in enumerate(question_ids) if i < len(ref_solutions[q_id]['options'])]
        valid_counts = [counts[j] for j, q_id in enumerate(question_ids) if i < len(ref_solutions[q_id]['options'])]
        valid_colors = [c for c in colors if c != 'none']
        
        if valid_positions:
            ax.bar(valid_positions, valid_counts, width, label=f'Option {chr(ord("A") + i)}', color=valid_colors)

    ax.set_title('Answer Distribution per Question', fontsize=16)
    ax.set_xlabel('Question ID', fontsize=12)
    ax.set_ylabel('Number of Students', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Q{q_id}' for q_id in question_ids])
    
    max_count = 0
    if all_answers_translated:
        max_count = translated_df.groupby('question_id')['ref_answer_idx'].count().max()

    ax.set_yticks(np.arange(0, max_count + 2, 1))
    
    # Custom legend for colors
    legend_elements = [Patch(facecolor='green', label='Correct Answer'),
                       Patch(facecolor='red', label='Incorrect Answer')]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plot_filename = os.path.join(output_dir, "answer_distribution.png")
    try:
        plt.savefig(plot_filename)
        logging.info(f"Answer distribution plot saved to {os.path.abspath(plot_filename)}")
    except Exception as e:
        logging.error(f"Error saving answer distribution plot: {e}")

def parse_q_list(q_str: Optional[str]) -> List[int]:
    """Converts a comma-separated string of question numbers to a sorted list of unique integers."""
    if not q_str:
        return []
    try:
        return sorted(list(set(int(q.strip()) for q in q_str.split(',') if q.strip().isdigit())))
    except ValueError:
        logging.warning(f"Invalid format for question list string: '{q_str}'. Expected comma-separated numbers. Returning empty list.")
        return []

def analyze_results(csv_filepath, max_score, output_dir=".", void_questions_str: Optional[str] = None, void_questions_nicely_str: Optional[str] = None, solutions_per_model=None):
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

    if solutions_per_model:
        _plot_answer_distribution(df, solutions_per_model, output_dir)
        
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
