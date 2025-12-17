import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from collections import Counter
import logging
from typing import Optional, List, Dict
from tabulate import tabulate
from matplotlib.patches import Patch
import textwrap

from pexams import utils

def _truncate_text(text, width=60):
    if not text: return ""
    # Use textwrap.shorten to truncate
    return textwrap.shorten(str(text), width=width, placeholder="...")

def _plot_answer_distribution(df, solutions_per_model, output_dir):
    """
    Plots the distribution of answers for each question in horizontal bar charts,
    normalized to a reference model's answer order. Creates multiple plot files
    if there are many questions.
    """
    if not solutions_per_model:
        logging.warning("Cannot generate answer distribution plot: solutions_per_model is empty.")
        return

    # Assuming the first model key is the reference (e.g., "1")
    ref_model_key = sorted(solutions_per_model.keys())[0]
    ref_solutions = solutions_per_model[ref_model_key]
    
    # Create a mapping from option text to the reference index for each question
    option_text_to_ref_idx = {}
    for q_id, q_data in ref_solutions.items():
        if 'options' in q_data:
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
            if q_id not in current_model_solutions or not isinstance(ans_char, str):
                continue
            
            if ans_char == 'NA':
                all_answers_translated.append({'question_id': q_id, 'ref_answer_idx': 'NA'})
                continue

            # Convert character answer to index (A=0, B=1, ...)
            ans_idx = ord(ans_char) - ord('A')
            
            # Get the text of the option the student chose
            try:
                if 'options' in current_model_solutions[q_id] and ans_idx < len(current_model_solutions[q_id]['options']):
                    chosen_option_text = current_model_solutions[q_id]['options'][ans_idx]['text']
                else:
                    continue
            except (IndexError, KeyError):
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
    
    QUESTIONS_PER_PAGE = 6
    COLS = 2
    ROWS = int(np.ceil(QUESTIONS_PER_PAGE / COLS))
    
    num_pages = int(np.ceil(num_questions / QUESTIONS_PER_PAGE))

    for page_idx in range(num_pages):
        start_idx = page_idx * QUESTIONS_PER_PAGE
        end_idx = start_idx + QUESTIONS_PER_PAGE
        page_question_ids = question_ids[start_idx:end_idx]
        
        if not page_question_ids:
            continue
            
        # Dynamically calculate required rows for this page to avoid empty space
        num_q_on_page = len(page_question_ids)
        rows_needed = int(np.ceil(num_q_on_page / COLS))

        fig, axes = plt.subplots(rows_needed, COLS, figsize=(16, 2.5 * rows_needed))
        # Flatten axes for easy iteration, handle case of 1x1 or 1D array
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
            
        for i, q_id in enumerate(page_question_ids):
            ax = axes_flat[i]
            
            q_data = ref_solutions[q_id]
            q_text = q_data.get('text', f'Question {q_id}')
            options = q_data.get('options', [])
            correct_idx = q_data.get('correct_answer_index')
            
            # Get counts for this question
            q_counts = translated_df[translated_df['question_id'] == q_id]['ref_answer_idx'].value_counts()
            
            # Prepare data for plotting
            bar_labels = []
            widths = []
            colors = []
            
            # Iterate through options + NA
            for opt_idx, opt in enumerate(options):
                opt_text = _truncate_text(opt['text'], width=60) # Shorten more aggressively
                bar_labels.append(f"{chr(ord('A') + opt_idx)}) {opt_text}")
                widths.append(q_counts.get(opt_idx, 0))
                # Softer colors
                colors.append('#77DD77' if opt_idx == correct_idx else '#FF6961') 
                
            # Add NA
            bar_labels.append("NA")
            widths.append(q_counts.get('NA', 0))
            colors.append('#CFCFC4') # Softer gray for NA
            
            # Horizontal Bar Plot
            y_pos = np.arange(len(bar_labels))
            
            # Add small epsilon (0.05) to widths for visualization so 0-count bars are visible (showing color)
            plot_widths = [w + 0.5 for w in widths]
            
            # No spacing between bars (height=1.0)
            rects = ax.barh(y_pos, plot_widths, align='center', color=colors, height=1.0)
            
            # Remove default Y axis labels (we'll put text inside)
            ax.set_yticks([]) 
            ax.invert_yaxis()  # Labels read top-to-bottom
            
            # Remove all spines except bottom
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            
            # Set X axis (0 to N students)
            total_students = len(df)
            
            # Ensure xlim covers the max width (which might be > total_students due to +0.2)
            max_width = max(plot_widths) if plot_widths else 0
            ax.set_xlim(0, max(total_students, max_width))
            
            # Use default tick locations, but maybe limit range?
            # User asked: "keep the x axis ticks at default"
            # We just leave ax.set_xticks() alone.
            
            # No x-label
            ax.set_xlabel('')
            
            ax.set_title(_truncate_text(f"{q_id}. {q_text}", width=100), fontsize=12, loc='left', pad=5)
            
            # Add text inside bars
            for rect, label, count in zip(rects, bar_labels, widths):               
                # Use offset points to place text consistently regardless of x-axis scale
                ax.annotate(label, 
                            xy=(0, rect.get_y() + rect.get_height()/2),
                            xytext=(5, 0), # 5 points padding from left
                            textcoords="offset points",
                            ha='left', va='center', fontsize=12, color='black')

            
        # Hide unused subplots completely
        for j in range(len(page_question_ids), len(axes_flat)):
            axes_flat[j].axis('off')
            
        plt.tight_layout()
        
        suffix = f"_p{page_idx+1}" if num_pages > 1 else ""
        plot_filename = os.path.join(output_dir, f"answer_distribution{suffix}.png")
        try:
            plt.savefig(plot_filename)
            logging.info(f"Answer distribution plot saved to {os.path.abspath(plot_filename)}")
        except Exception as e:
            logging.error(f"Error saving answer distribution plot: {e}")
        plt.close(fig)

def parse_q_list(q_str: Optional[str]) -> List[int]:
    """Converts a comma-separated string of question numbers to a sorted list of unique integers."""
    if not q_str:
        return []
    try:
        return sorted(list(set(int(q.strip()) for q in q_str.split(',') if q.strip().isdigit())))
    except ValueError:
        logging.warning(f"Invalid format for question list string: '{q_str}'. Expected comma-separated numbers. Returning empty list.")
        return []

def analyze_results(
    csv_filepath,
    output_dir=".",
    exam_dir: Optional[str] = None,
    solutions_per_model: Optional[Dict] = None,
    max_score: Optional[int] = None,
    void_questions_str: Optional[str] = None, 
    void_questions_nicely_str: Optional[str] = None
):
    """
    Analyzes exam results from a CSV file, scales scores to 0-10, 
    plots score distribution, and shows statistics.
    Allows for voiding questions or voiding them 'nicely' (only if incorrect/unanswered).
    
    You can provide either (solutions_per_model AND max_score) OR (exam_dir).
    """
    
    if solutions_per_model is None or max_score is None:
        if exam_dir:
            logging.info(f"Loading solutions from {exam_dir} for analysis...")
            solutions_per_model, _, max_score_loaded = utils.load_solutions(exam_dir)
            if max_score is None:
                max_score = max_score_loaded
        else:
            logging.error("Cannot perform analysis: solutions_per_model/max_score or exam_dir must be provided.")
            return

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
        logging.error("Error: 'score' column not found in CSV. Cannot perform analysis.")
        return

    void_q_list = parse_q_list(void_questions_str)
    void_q_nicely_list = parse_q_list(void_questions_nicely_str)

    if void_q_list:
        logging.info(f"Voiding questions (will be removed for all students): {void_q_list}")
    if void_q_nicely_list:
        logging.info(f"Voiding questions nicely (removed only if incorrect or not answered): {void_q_nicely_list}")

    # --- Recalculate scores based on voiding rules ---
    adjusted_scores = []
    adjusted_max_scores = []

    for _, row in df.iterrows():
        model_id = str(row['model_id'])
        if model_id not in solutions_per_model:
            adjusted_scores.append(0)
            adjusted_max_scores.append(max_score)
            continue

        model_solutions = solutions_per_model[model_id]
        student_score = 0
        student_max_score = 0
        
        q_ids = sorted(model_solutions.keys())

        for q_id in q_ids:
            # Question is completely voided for everyone
            if q_id in void_q_list:
                continue

            answer_col = f'answer_{q_id}'
            student_answer_char = row.get(answer_col)
            
            # Retrieve correct answer index
            # Check if using the full dump dict or simplified one
            sol_data = model_solutions[q_id]
            if isinstance(sol_data, dict):
                correct_answer_idx = sol_data.get('correct_answer_index')
            else:
                correct_answer_idx = sol_data
                
            if correct_answer_idx is None:
                continue # Skip questions without a correct answer (e.g., surveys)

            correct_answer_char = chr(ord('A') + correct_answer_idx)
            is_correct = (student_answer_char == correct_answer_char)

            # Question is voided nicely
            if q_id in void_q_nicely_list:
                if is_correct:
                    student_score += 1
                    student_max_score += 1
                # If incorrect, it doesn't count towards student's score or max score
            
            # Regular question
            else:
                if is_correct:
                    student_score += 1
                student_max_score += 1
        
        adjusted_scores.append(student_score)
        adjusted_max_scores.append(student_max_score)

    df['score_adjusted'] = adjusted_scores
    df['max_score_adjusted'] = adjusted_max_scores
    
    # --- Plot answer distribution before calculating final marks ---
    if solutions_per_model:
        _plot_answer_distribution(df, solutions_per_model, output_dir)
        
    df['mark'] = (df['score_adjusted'] / df['max_score_adjusted'].replace(0, 1)) * 10
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

    ax.set_title(f'Distribution of Exam Marks (Scaled to 0-10)', fontsize=15)
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
