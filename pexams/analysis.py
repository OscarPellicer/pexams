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
from playwright.sync_api import sync_playwright
import markdown

from pexams import utils

def _truncate_text(text, width=60):
    if not text: return ""
    # Use textwrap.shorten to truncate
    return textwrap.shorten(str(text), width=width, placeholder="...")

def _get_translated_answers(df, solutions_per_model):
    """
    Translates all student answers to the reference model's option indexing.
    Returns a DataFrame with columns ['question_id', 'ref_answer_idx'].
    """
    if not solutions_per_model:
        return pd.DataFrame()

    ref_model_key = sorted(solutions_per_model.keys())[0]
    ref_solutions = solutions_per_model[ref_model_key]
    
    option_text_to_ref_idx = {}
    for q_id, q_data in ref_solutions.items():
        if 'options' in q_data:
            option_text_to_ref_idx[q_id] = {opt['text']: i for i, opt in enumerate(q_data['options'])}

    all_answers_translated = []
    for _, row in df.iterrows():
        model_id = str(row['model_id'])
        if model_id not in solutions_per_model:
            continue
        
        current_model_solutions = solutions_per_model[model_id]
        
        for q_num_str, ans_char in row.items():
            if not q_num_str.startswith('answer_'):
                continue
            
            try:
                parts = q_num_str.split('_')
                if len(parts) < 2: continue
                q_id = int(parts[1])
            except ValueError:
                continue

            # Check if answer is NA (string 'NA' or NaN/None)
            # We explicitly check for 'NA' string first, then for nulls
            is_na = (ans_char == 'NA') or pd.isna(ans_char)

            if not is_na and not isinstance(ans_char, str):
                continue
            
            if q_id not in current_model_solutions:
                continue
            
            if is_na:
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

    return pd.DataFrame(all_answers_translated)

def _save_answer_stats_csv(translated_df, solutions_per_model, output_dir):
    """Saves answer statistics to a CSV, including original IDs if available."""
    if translated_df.empty or not solutions_per_model:
        return

    ref_model_key = sorted(solutions_per_model.keys())[0]
    ref_solutions = solutions_per_model[ref_model_key]
    
    stats_data = []
    
    # Get all unique questions from reference
    for q_id in sorted(ref_solutions.keys()):
        q_data = ref_solutions[q_id]
        
        # Determine original ID
        original_id = q_data.get('original_id', q_id)
        if original_id is None: 
            original_id = q_id
            
        q_text = _truncate_text(q_data.get('text', ''), width=100)
        
        # Count answers
        q_counts = translated_df[translated_df['question_id'] == q_id]['ref_answer_idx'].value_counts()
        
        row = {
            'original_id': original_id,
            'exam_q_id': q_id,
            'question_text': q_text,
            'total_answers': int(q_counts.sum()),
            'NA_count': int(q_counts.get('NA', 0))
        }
        
        options = q_data.get('options', [])
        for i, opt in enumerate(options):
            label = chr(ord('A') + i)
            count = q_counts.get(i, 0)
            row[f'option_{label}_count'] = int(count)
            # Add option text for clarity
            row[f'option_{label}_text'] = _truncate_text(opt['text'], width=50)
            
        stats_data.append(row)
        
    if not stats_data:
        return
        
    stats_df = pd.DataFrame(stats_data)
    output_path = os.path.join(output_dir, "question_stats.csv")
    stats_df.to_csv(output_path, index=False)
    logging.info(f"Question statistics saved to {os.path.abspath(output_path)}")

def _generate_stats_pdf(df, solutions_per_model, output_dir, mark_plot_path=None):
    """
    Generates a PDF report with answer statistics using HTML and Playwright.
    Also saves the stats CSV.
    """
    if not solutions_per_model:
        return

    logging.info("Generating PDF statistics report...")
    translated_df = _get_translated_answers(df, solutions_per_model)
    if translated_df.empty:
        logging.warning("No translated answers found for report.")
        return

    # Save CSV stats (legacy support / raw data)
    _save_answer_stats_csv(translated_df, solutions_per_model, output_dir)

    ref_model_key = sorted(solutions_per_model.keys())[0]
    ref_solutions = solutions_per_model[ref_model_key]
    
    total_students = len(df)
    
    # Calculate stats
    marks = df['mark_clipped']
    stats = {
        'total_students': total_students,
        'mean': marks.mean(),
        'median': marks.median(),
        'std': marks.std(),
        'min': marks.min(),
        'max': marks.max(),
        'pass_count': len(marks[marks >= 5.0]),
        'pass_rate': (len(marks[marks >= 5.0]) / total_students * 100) if total_students > 0 else 0
    }

    # Markdown extension config
    extensions = [
        'pymdownx.arithmatex',
        'pymdownx.inlinehilite',
        'fenced_code',
        'codehilite'
    ]
    extension_configs = {
        'pymdownx.arithmatex': {'generic': True}
    }

    def render_md(text):
        if not text: return ""
        html = markdown.markdown(str(text).replace('\n', ' <br> '), extensions=extensions, extension_configs=extension_configs).strip()
        if html.startswith("<p>"):
            html = html[3:-4]
        return html

    # CSS for the report
    css = """
    body { 
        font-family: 'Open Sans', 'Segoe UI', Tahoma, sans-serif; 
        margin: 0; 
        padding: 20px 30px; 
        color: #333; 
        background-color: #fff; 
        font-size: 13px;
    }
    h1 { text-align: left; color: #2c3e50; margin-bottom: 25px; font-size: 24px; }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 15px;
        margin-bottom: 30px;
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
    }
    .stat-item {
        display: flex;
        flex-direction: column;
    }
    .stat-label {
        font-size: 11px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-value {
        font-size: 18px;
        font-weight: 600;
        color: #2c3e50;
    }
    
    .section-title { 
        border-bottom: 2px solid #2c3e50; 
        padding-bottom: 8px; 
        margin-top: 30px; 
        margin-bottom: 15px; 
        color: #2c3e50; 
        font-size: 18px;
        text-align: left;
    }
    
    .distribution-section { text-align: center; margin-bottom: 40px; page-break-after: always; }
    .distribution-img { max-width: 80%; height: auto; border: 1px solid #ddd; padding: 5px; border-radius: 4px; }
    
    .question-block { 
        margin-bottom: 15px; 
        background: #fff; 
        border: 1px solid #e0e0e0; 
        border-radius: 6px; 
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        page-break-inside: avoid;
    }
    .question-header { 
        font-weight: 600; 
        font-size: 1.05em; 
        margin-bottom: 8px; 
        color: #2c3e50;
    }
    
    .options-list { display: flex; flex-direction: column; gap: 5px; }
    
    .option-row { 
        position: relative; 
        border: 1px solid #eee; 
        border-radius: 4px; 
        overflow: hidden; 
        background-color: #f9f9f9;
        font-size: 0.95em;
    }
    
    .option-bg { 
        position: absolute; 
        top: 0; left: 0; bottom: 0; 
        z-index: 0; 
        height: 100%;
    }
    
    /* Colors for bars */
    .bg-correct { background-color: #d4edda; border-right: 1px solid #c3e6cb; } /* Light Green */
    .bg-incorrect { background-color: #f8d7da; border-right: 1px solid #f5c6cb; } /* Light Red */
    .bg-na { background-color: #e2e3e5; border-right: 1px solid #d6d8db; } /* Light Gray */
    
    .option-content { 
        position: relative; 
        z-index: 1; 
        padding: 8px 12px; 
        display: flex; 
        justify-content: space-between; 
        align-items: center;
        min-height: 24px;
    }
    
    .option-text-container { display: flex; align-items: center; gap: 10px; flex: 1; }
    .option-label { font-weight: bold; min-width: 20px; color: #555; }
    .option-text { flex: 1; }
    
    .badge { 
        font-size: 0.7em; 
        padding: 2px 6px; 
        border-radius: 4px; 
        font-weight: bold;
        text-transform: uppercase;
        margin-left: 10px;
        white-space: nowrap;
    }
    .badge-correct { background-color: #28a745; color: white; }
    
    .stats { 
        font-family: 'Consolas', monospace; 
        font-weight: bold; 
        color: #555; 
        white-space: nowrap; 
        margin-left: 15px;
        font-size: 0.9em;
    }
    
    /* Image handling in questions */
    img { max-width: 100%; height: auto; }
    """
    
    html_parts = []
    html_parts.append(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Exam Statistics Report</title>
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,300..800;1,300..800&display=swap">
        <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <style>{css}</style>
    </head>
    <body>
        <h1>Exam Statistics Report</h1>
        
        <div class="stats-grid">
            <div class="stat-item">
                <span class="stat-label">Total Students</span>
                <span class="stat-value">{stats['total_students']}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Mean Score</span>
                <span class="stat-value">{stats['mean']:.2f}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Median Score</span>
                <span class="stat-value">{stats['median']:.2f}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Std Dev</span>
                <span class="stat-value">{stats['std']:.2f}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Min / Max</span>
                <span class="stat-value">{stats['min']:.1f} / {stats['max']:.1f}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Pass Rate (≥5.0)</span>
                <span class="stat-value">{stats['pass_rate']:.1f}%</span>
            </div>
        </div>
    """)
    
    # Mark Distribution Plot
    if mark_plot_path and os.path.exists(mark_plot_path):
        abs_plot_path = os.path.abspath(mark_plot_path).replace('\\', '/')
        html_parts.append(f"""
        <div class="distribution-section">
            <h2 class="section-title">Score Distribution</h2>
            <img src="file:///{abs_plot_path}" class="distribution-img" alt="Score Distribution">
        </div>
        """)
    
    html_parts.append("<h2 class='section-title'>Question Analysis</h2>")
    
    for q_id in sorted(ref_solutions.keys()):
        q_data = ref_solutions[q_id]
        q_raw_text = q_data.get('text', f'Question {q_id}')
        
        # Render markdown for question
        q_html_text = render_md(q_raw_text)
        
        options = q_data.get('options', [])
        correct_idx = q_data.get('correct_answer_index')
        
        q_counts = translated_df[translated_df['question_id'] == q_id]['ref_answer_idx'].value_counts()
        
        html_parts.append(f"""
        <div class="question-block">
            <div class="question-header">{q_id}. {q_html_text}</div>
            <div class="options-list">
        """)
        
        for i, opt in enumerate(options):
            label = chr(ord('A') + i)
            opt_raw_text = opt['text']
            
            # Render markdown for option
            opt_html_text = render_md(opt_raw_text)
            
            count = q_counts.get(i, 0)
            percent = (count / total_students) * 100 if total_students > 0 else 0
            
            is_correct = (i == correct_idx)
            bg_class = "bg-correct" if is_correct else "bg-incorrect"
            
            badge_html = '<span class="badge badge-correct">Correct</span>' if is_correct else ""
            
            html_parts.append(f"""
            <div class="option-row">
                <div class="option-bg {bg_class}" style="width: {percent}%;"></div>
                <div class="option-content">
                    <div class="option-text-container">
                        <span class="option-label">{label})</span>
                        <div class="option-text">{opt_html_text}</div>
                        {badge_html}
                    </div>
                    <span class="stats">{count} ({percent:.1f}%)</span>
                </div>
            </div>
            """)
            
        # Always add NA bar, even if count is 0, for consistency
        na_count = q_counts.get('NA', 0)
        percent = (na_count / total_students) * 100 if total_students > 0 else 0
        
        html_parts.append(f"""
        <div class="option-row">
            <div class="option-bg bg-na" style="width: {percent}%;"></div>
            <div class="option-content">
                <div class="option-text-container">
                    <span class="option-label">NA</span>
                    <div class="option-text">No Answer</div>
                </div>
                <span class="stats">{na_count} ({percent:.1f}%)</span>
            </div>
        </div>
        """)
            
        html_parts.append("</div></div>") # Close options-list and question-block
        
    html_parts.append("</body></html>")
    
    full_html = "\n".join(html_parts)
    
    html_path = os.path.join(output_dir, "stats_report.html")
    pdf_path = os.path.join(output_dir, "stats_report.pdf")
    
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(full_html)
        logging.info(f"Generated HTML report: {html_path}")
    except Exception as e:
        logging.error(f"Error saving HTML report: {e}")
        return

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            abs_html_path = os.path.abspath(html_path).replace('\\', '/')
            page.goto(f"file:///{abs_html_path}", wait_until="networkidle")
            
            # Wait for MathJax to finish rendering.
            page.evaluate("() => MathJax.typesetPromise()")
            page.wait_for_timeout(1000)

            page.pdf(path=pdf_path, format="A4", print_background=True, 
                     margin={"top": "15mm", "bottom": "15mm", "left": "15mm", "right": "15mm"})
            browser.close()
        logging.info(f"Statistics report saved to {os.path.abspath(pdf_path)}")
    except Exception as e:
        logging.error(f"Failed to convert report to PDF: {e}")

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
    void_questions_nicely_str: Optional[str] = None,
    penalty: float = 0.0
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
    correct_counts = []
    incorrect_counts = []
    na_counts = []

    for _, row in df.iterrows():
        model_id = str(row['model_id'])
        if model_id not in solutions_per_model:
            adjusted_scores.append(0)
            adjusted_max_scores.append(max_score)
            correct_counts.append(0)
            incorrect_counts.append(0)
            na_counts.append(0)
            continue

        model_solutions = solutions_per_model[model_id]
        student_score = 0
        student_max_score = 0
        student_correct = 0
        student_incorrect = 0
        student_na = 0
        
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
            is_answered = (student_answer_char != 'NA' and isinstance(student_answer_char, str))

            # Question is voided nicely
            if q_id in void_q_nicely_list:
                if is_correct:
                    student_score += 1
                    student_max_score += 1
                    student_correct += 1
                # If incorrect, it doesn't count towards student's score or max score
            
            # Regular question
            else:
                student_max_score += 1
                if is_correct:
                    student_score += 1
                    student_correct += 1
                elif is_answered:
                    if penalty > 0:
                         student_score -= penalty
                    student_incorrect += 1
                else:
                    student_na += 1
        
        adjusted_scores.append(student_score)
        adjusted_max_scores.append(student_max_score)
        correct_counts.append(student_correct)
        incorrect_counts.append(student_incorrect)
        na_counts.append(student_na)

    df['score_adjusted'] = adjusted_scores
    df['max_score_adjusted'] = adjusted_max_scores
    df['correct_count'] = correct_counts
    df['incorrect_count'] = incorrect_counts
    df['na_count'] = na_counts
    
    # (Old plotting function call removed here)
        
    df['mark'] = (df['score_adjusted'] / df['max_score_adjusted'].replace(0, 1)) * 10
    df['mark_clipped'] = np.clip(df['mark'], 0, 10)

    print("\n--- Descriptive Statistics for Marks (0-10 scale) ---")
    stats = df['mark_clipped'].describe()
    print(stats)
    
    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4))

    df['mark_binned_for_plot'] = np.floor(df['mark_clipped'].fillna(0) + 0.5).astype(int)
    score_counts = Counter(df['mark_binned_for_plot'])
    all_possible_scores = np.arange(0, 11)
    frequencies = [score_counts.get(s, 0) for s in all_possible_scores]

    plt.bar(all_possible_scores, frequencies, width=1.0, edgecolor='black', align='center', color='skyblue')

    ax.set_title(f'Distribution of Exam Marks (Scaled to 0-10)', fontsize=14, loc='left')
    ax.set_xlabel('Mark (0-10 Scale)', fontsize=11)
    ax.set_ylabel('Number of Students', fontsize=11)
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
    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    plot_filename = os.path.join(output_dir, "mark_distribution_0_10.png")
    try:
        plt.savefig(plot_filename, dpi=300)
        logging.info(f"\nPlot saved to {os.path.abspath(plot_filename)}")
    except Exception as e:
        logging.error(f"Error saving plot: {e}")
    plt.close(fig)

    # --- Print Student Marks ---
    print("\n--- Student Marks (0-10 Scale) ---")
    
    results_to_print_df = df[['student_id', 'student_name', 'score_adjusted', 'max_score_adjusted', 'correct_count', 'incorrect_count', 'na_count', 'mark_clipped']].copy()
    results_to_print_df.rename(columns={'mark_clipped': 'mark', 'score_adjusted': 'score', 'max_score_adjusted': 'max_score', 'correct_count': 'correct', 'incorrect_count': 'incorrect', 'na_count': 'NA'}, inplace=True)
    
    # Save to a new CSV
    final_csv_path = os.path.join(output_dir, "final_marks.csv")
    results_to_print_df.to_csv(final_csv_path, index=False)
    logging.info(f"Final marks saved to {os.path.abspath(final_csv_path)}")
    
    # Print to console
    results_to_print_df.index = range(1, len(results_to_print_df) + 1)
    print(tabulate(results_to_print_df, headers='keys', tablefmt='psql', floatfmt=".2f"))

    # --- Generate PDF Stats Report ---
    if solutions_per_model:
        _generate_stats_pdf(df, solutions_per_model, output_dir, mark_plot_path=plot_filename)
