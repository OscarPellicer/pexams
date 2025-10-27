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
    Plots the distribution of answers for each question in grouped bar charts,
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
    
    QUESTIONS_PER_PLOT = 20  # Max questions per plot file
    num_plots = int(np.ceil(num_questions / QUESTIONS_PER_PLOT))

    for plot_idx in range(num_plots):
        start_idx = plot_idx * QUESTIONS_PER_PLOT
        end_idx = start_idx + QUESTIONS_PER_PLOT
        plot_question_ids = question_ids[start_idx:end_idx]
        
        num_q_in_plot = len(plot_question_ids)
        if num_q_in_plot == 0:
            continue

        # Adjust figure width based on number of questions in this subplot
        fig_width = max(10, min(20, num_q_in_plot * 1.8))
        fig, ax = plt.subplots(figsize=(fig_width, 8))

        answer_counts_by_q = {
            q_id: translated_df[translated_df['question_id'] == q_id]['ref_answer_idx'].value_counts()
            for q_id in plot_question_ids
        }

        max_num_options = 0
        if ref_solutions:
            max_num_options = max(len(ref_solutions[q_id].get('options', [])) for q_id in plot_question_ids)

        num_bars_per_group = max_num_options + 1  # +1 for "NA"
        group_width = 0.8
        bar_width = group_width / num_bars_per_group
        
        x = np.arange(num_q_in_plot)

        for i in range(max_num_options + 1): # Loop through options + NA
            is_na_bar = (i == max_num_options)
            
            if is_na_bar:
                counts = [answer_counts_by_q.get(q_id, pd.Series()).get('NA', 0) for q_id in plot_question_ids]
            else:
                counts = [answer_counts_by_q.get(q_id, pd.Series()).get(i, 0) for q_id in plot_question_ids]

            offset = (i - (num_bars_per_group - 1) / 2) * bar_width
            
            colors = []
            if is_na_bar:
                colors = ['#7f7f7f'] * num_q_in_plot # Gray for NA
            else:
                for q_id in plot_question_ids:
                    correct_idx = ref_solutions[q_id].get('correct_answer_index')
                    if 'options' in ref_solutions[q_id] and i < len(ref_solutions[q_id]['options']):
                        colors.append('#2ca02c' if i == correct_idx else '#d62728')
                    else:
                        colors.append('none')

            valid_positions = [x[j] + offset for j, c in enumerate(colors) if c != 'none']
            valid_counts = [counts[j] for j, c in enumerate(colors) if c != 'none']
            valid_colors = [c for c in colors if c != 'none']
            
            if is_na_bar: # Override for NA bar
                valid_positions = x + offset
                valid_counts = counts
                valid_colors = colors

            if len(valid_positions) > 0:
                rects = ax.bar(valid_positions, valid_counts, bar_width * 0.95, color=valid_colors)
                
                # Add text labels on top of the bars
                label_text = "NA" if is_na_bar else chr(ord('A') + i)
                labels = [label_text if count > 0 else "" for count in valid_counts]
                ax.bar_label(rects, labels=labels, padding=3, fontsize=8, color='black')

        q_start, q_end = plot_question_ids[0], plot_question_ids[-1]
        ax.set_title(f'Answer Distribution for Questions {q_start}-{q_end}', fontsize=16)
        ax.set_xlabel('Question ID', fontsize=12)
        ax.set_ylabel('Number of Students', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Q{q_id}' for q_id in plot_question_ids])
        
        max_count = translated_df[translated_df['question_id'].isin(plot_question_ids)].groupby('question_id').size().max() if not translated_df.empty else 0

        if max_count > 0:
            ax.set_ylim(top=max_count * 1.15)
            if max_count <= 20:
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        legend_elements = [
            Patch(facecolor='#2ca02c', edgecolor='black', label='Correct Answer'),
            Patch(facecolor='#d62728', edgecolor='black', label='Incorrect Answer'),
            Patch(facecolor='#7f7f7f', edgecolor='black', label='Not Answered (NA)')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize='small')

        plt.tight_layout()
        
        suffix = f"_q{q_start}-{q_end}" if num_plots > 1 else ""
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
    max_score, 
    solutions_per_model, 
    output_dir=".", 
    void_questions_str: Optional[str] = None, 
    void_questions_nicely_str: Optional[str] = None
):
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
            
            correct_answer_idx = model_solutions[q_id]['correct_answer_index']
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
