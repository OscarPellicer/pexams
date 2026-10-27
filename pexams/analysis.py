"""Analysis of corrected exams: marks recomputed with voided questions and penalties, mark
distribution plot, ``final_marks.csv``, per-question statistics and the PDF statistics report."""
import logging
import os
import textwrap
from collections import Counter
from typing import Dict, List, Optional

import markdown
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tabulate import tabulate  # noqa: E402

from pexams import utils  # noqa: E402
from pexams.pdf_render import html_document, html_to_pdf  # noqa: E402

PASS_MARK = 5.0


_REPORT_CSS = """
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


def _summary_stats(marks: pd.Series) -> dict:
    total_students = len(marks)
    passed = len(marks[marks >= PASS_MARK])
    return {
        'total_students': total_students,
        'mean': marks.mean(),
        'median': marks.median(),
        'std': marks.std(),
        'min': marks.min(),
        'max': marks.max(),
        'pass_count': passed,
        'pass_rate': (passed / total_students * 100) if total_students > 0 else 0,
    }


def _render_inline_md(text) -> str:
    """Markdown of a question or option as inline HTML (math kept for MathJax)."""
    if not text:
        return ""
    html = markdown.markdown(str(text).replace('\n', ' <br> '),
                             extensions=['pymdownx.arithmatex', 'pymdownx.inlinehilite', 'fenced_code', 'codehilite'],
                             extension_configs={'pymdownx.arithmatex': {'generic': True}}).strip()
    if html.startswith("<p>"):
        html = html[3:-4]
    return html


def _stats_grid_html(stats: dict) -> str:
    items = [
        ("Total Students", stats['total_students']),
        ("Mean Score", f"{stats['mean']:.2f}"),
        ("Median Score", f"{stats['median']:.2f}"),
        ("Std Dev", f"{stats['std']:.2f}"),
        ("Min / Max", f"{stats['min']:.1f} / {stats['max']:.1f}"),
        (f"Pass Rate (≥{PASS_MARK:.1f})", f"{stats['pass_rate']:.1f}%"),
    ]
    cells = "".join(f"""
            <div class="stat-item">
                <span class="stat-label">{label}</span>
                <span class="stat-value">{value}</span>
            </div>""" for label, value in items)
    return f'<div class="stats-grid">{cells}\n        </div>'


def _option_row_html(label: str, text_html: str, count: int, total_students: int, bg_class: str, badge_html: str = "") -> str:
    percent = (count / total_students) * 100 if total_students > 0 else 0
    return f"""
            <div class="option-row">
                <div class="option-bg {bg_class}" style="width: {percent}%;"></div>
                <div class="option-content">
                    <div class="option-text-container">
                        <span class="option-label">{label}</span>
                        <div class="option-text">{text_html}</div>
                        {badge_html}
                    </div>
                    <span class="stats">{count} ({percent:.1f}%)</span>
                </div>
            </div>
            """


def _question_block_html(q_id, q_data: dict, q_counts: pd.Series, total_students: int) -> str:
    """One question of the report: each option with the share of students who chose it, and the
    unanswered ones (always shown, for consistency)."""
    correct_idx = q_data.get('correct_answer_index')
    rows = []
    for i, opt in enumerate(q_data.get('options', [])):
        is_correct = (i == correct_idx)
        rows.append(_option_row_html(
            f"{chr(ord('A') + i)})", _render_inline_md(opt['text']), q_counts.get(i, 0), total_students,
            "bg-correct" if is_correct else "bg-incorrect",
            '<span class="badge badge-correct">Correct</span>' if is_correct else ""))
    rows.append(_option_row_html("NA", "No Answer", q_counts.get('NA', 0), total_students, "bg-na"))
    return f"""
        <div class="question-block">
            <div class="question-header">{q_id}. {_render_inline_md(q_data.get('text', f'Question {q_id}'))}</div>
            <div class="options-list">
        {"".join(rows)}</div></div>"""


def _report_body_html(df: pd.DataFrame, translated_df: pd.DataFrame, ref_solutions: dict, mark_plot_path=None) -> str:
    total_students = len(df)
    parts = ["<h1>Exam Statistics Report</h1>", _stats_grid_html(_summary_stats(df['mark_clipped']))]
    if mark_plot_path and os.path.exists(mark_plot_path):
        parts.append(f"""
        <div class="distribution-section">
            <h2 class="section-title">Score Distribution</h2>
            <img src="file:///{os.path.abspath(mark_plot_path).replace(chr(92), '/')}" class="distribution-img" alt="Score Distribution">
        </div>
        """)
    parts.append("<h2 class='section-title'>Question Analysis</h2>")
    for q_id in sorted(ref_solutions.keys()):
        q_counts = translated_df[translated_df['question_id'] == q_id]['ref_answer_idx'].value_counts()
        parts.append(_question_block_html(q_id, ref_solutions[q_id], q_counts, total_students))
    return "\n".join(parts)


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

    ref_solutions = solutions_per_model[sorted(solutions_per_model.keys())[0]]
    full_html = html_document(_report_body_html(df, translated_df, ref_solutions, mark_plot_path),
                              title="Exam Statistics Report", css=_REPORT_CSS)

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
        html_to_pdf(html_path, pdf_path, margin={"top": "15mm", "bottom": "15mm", "left": "15mm", "right": "15mm"})
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

UNVOIDED_SCANS_DIR = ".unvoided"  # inside scanned_pages: sheets as drawn by a correction without .raw sheets


def redraw_answer_sheets(output_dir: str, df: pd.DataFrame, solutions_per_model: Dict,
                         voided_questions: List[int], lang: str = "en") -> int:
    """Redraw each student's annotated answer sheet (``<output_dir>/scanned_pages/<id>.png``) from
    what ``correction_results.csv`` says now: the answers (so a manual fix of a misread answer shows
    up), the ID and name read, and a grey line labelled e.g. "ANULADA" across voided questions.

    It draws on the unannotated sheet saved by the correction (``scanned_pages/.raw/``). Sheets
    corrected before that existed only get the void marks, drawn on a copy of the original kept in
    ``scanned_pages/.unvoided/``. Needs the full question data per model (``solutions_per_model`` as
    from ``utils.load_solutions``) to rebuild the sheet layout. Returns the number of sheets written.
    """
    import re
    import shutil

    import cv2

    from pexams.correct_exams import PX_PER_MM, RAW_SHEETS_DIR, annotate_answer_sheet, draw_void_marks
    from pexams.layout import get_answer_sheet_layout
    from pexams.schemas import PexamQuestion
    from pexams.translations import LANG_STRINGS

    scans_dir = os.path.join(output_dir, "scanned_pages")
    if not os.path.isdir(scans_dir):
        return 0
    raw_dir, originals_dir = os.path.join(scans_dir, RAW_SHEETS_DIR), os.path.join(scans_dir, UNVOIDED_SCANS_DIR)
    label = LANG_STRINGS.get(lang, {}).get("voided") or "VOID"
    models, written = {}, 0

    def model_data(model_id):
        if model_id not in models:
            dumps = [q for _, q in sorted((solutions_per_model or {}).get(model_id, {}).items()) if isinstance(q, dict)]
            try:
                questions = [PexamQuestion.model_validate(q) for q in dumps]
                models[model_id] = (get_answer_sheet_layout(questions) if questions else None,
                                    {q.id: q.correct_answer_index for q in questions if q.correct_answer_index is not None})
            except Exception as e:
                logging.warning("Cannot rebuild the answer sheet of model %s to redraw it: %s", model_id, e)
                models[model_id] = (None, {})
        return models[model_id]

    for _, row in df.iterrows():
        student_id = str(row.get("student_id", ""))
        names = [n for n in dict.fromkeys([student_id, re.sub(r"[^A-Za-z0-9_-]+", "_", student_id).strip("_")]) if n]
        sheet_layout, solutions = model_data(str(row.get("model_id", "")))

        raw = next((os.path.join(raw_dir, f"{n}.png") for n in names if os.path.exists(os.path.join(raw_dir, f"{n}.png"))), None)
        if raw and sheet_layout is not None:
            image = cv2.imread(raw)
            if image is None:
                continue
            answers = {}
            for q_id in solutions:
                value = str(row.get(f"answer_{q_id}", "") or "").strip().upper()
                answers[q_id] = ord(value) - ord("A") if len(value) == 1 and value.isalpha() else -1
            ocr_id = row.get("ocr_student_id", "")
            annotated = annotate_answer_sheet(
                image, sheet_layout, model_id=str(row.get("model_id", "")),
                student_name=str(row.get("ocr_student_name", "") or "") if pd.notna(row.get("ocr_student_name", "")) else "",
                id_text=str(ocr_id) if pd.notna(ocr_id) and str(ocr_id) else student_id,
                answers=answers, solutions=solutions, voided=voided_questions, void_label=label)
            cv2.imwrite(os.path.join(scans_dir, os.path.basename(raw)), annotated)
            written += 1
            continue

        # Older corrections (no .raw sheet): only the void marks, on a kept copy of the original.
        path = next((os.path.join(scans_dir, f"{n}.png") for n in names
                     if os.path.exists(os.path.join(scans_dir, f"{n}.png"))
                     or os.path.exists(os.path.join(originals_dir, f"{n}.png"))), None)
        if path is None or sheet_layout is None:
            continue
        original = os.path.join(originals_dir, os.path.basename(path))
        if not voided_questions:
            if os.path.exists(original):
                shutil.copyfile(original, path)
                written += 1
            continue
        os.makedirs(originals_dir, exist_ok=True)
        if not os.path.exists(original):
            shutil.copyfile(path, original)
        image = cv2.imread(original)
        if image is None:
            continue
        draw_void_marks(image, sheet_layout, voided_questions, label, PX_PER_MM)
        cv2.imwrite(path, image)
        written += 1
    return written


# Kept for callers of the first version (voids only).
mark_voided_questions_on_scans = redraw_answer_sheets


def _score_student(row, model_solutions: dict, void_q_list: List[int], void_q_nicely_list: List[int],
                   penalty: float):
    """Score, maximum score, and counts of correct, incorrect and unanswered questions of one
    student, with voided questions (removed for everyone, or "nicely": only when not correct)
    and the penalty (a fraction of each question's points) per wrong answer."""
    score = max_score = 0
    correct = incorrect = na = 0
    for q_id in sorted(model_solutions.keys()):
        if q_id in void_q_list:
            continue
        student_answer_char = row.get(f'answer_{q_id}')
        # Full question dumps (analysis) or bare correct-answer indices.
        sol_data = model_solutions[q_id]
        if isinstance(sol_data, dict):
            correct_answer_idx = sol_data.get('correct_answer_index')
            question_points = float(sol_data.get('points', 1.0) or 1.0)
        else:
            correct_answer_idx = sol_data
            question_points = 1.0
        if correct_answer_idx is None:
            continue  # Skip questions without a correct answer (e.g., surveys)

        is_correct = (student_answer_char == chr(ord('A') + correct_answer_idx))
        is_answered = (student_answer_char != 'NA' and isinstance(student_answer_char, str))
        if q_id in void_q_nicely_list:
            # Counts only if correct; otherwise it is left out of the student's score and maximum.
            if is_correct:
                score += question_points
                max_score += question_points
                correct += 1
            continue
        max_score += question_points
        if is_correct:
            score += question_points
            correct += 1
        elif is_answered:
            if penalty > 0:
                # The penalty is a fraction of the question's value, so it scales with
                # per-question points (e.g. --mc-total-points) instead of being absolute.
                score -= penalty * question_points
            incorrect += 1
        else:
            na += 1
    return score, max_score, correct, incorrect, na


def _add_marks(df: pd.DataFrame, solutions_per_model: Dict, max_score: float, void_q_list: List[int],
               void_q_nicely_list: List[int], penalty: float) -> None:
    """Add the recomputed scores, counts and 0-10 marks (``mark``, ``mark_clipped``) to ``df``."""
    rows = []
    for _, row in df.iterrows():
        model_id = str(row['model_id'])
        if model_id not in solutions_per_model:
            rows.append((0, max_score, 0, 0, 0))
        else:
            rows.append(_score_student(row, solutions_per_model[model_id], void_q_list, void_q_nicely_list, penalty))
    columns = ['score_adjusted', 'max_score_adjusted', 'correct_count', 'incorrect_count', 'na_count']
    for index, column in enumerate(columns):
        df[column] = [r[index] for r in rows]
    df['mark'] = (df['score_adjusted'] / df['max_score_adjusted'].replace(0, 1)) * 10
    df['mark_clipped'] = np.clip(df['mark'], 0, 10)


def _plot_mark_distribution(df: pd.DataFrame, output_dir: str) -> str:
    """Bar chart of the rounded 0-10 marks with their mean and median; returns the PNG path."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4))

    df['mark_binned_for_plot'] = np.floor(df['mark_clipped'].fillna(0) + 0.5).astype(int)
    score_counts = Counter(df['mark_binned_for_plot'])
    all_possible_scores = np.arange(0, 11)
    frequencies = [score_counts.get(s, 0) for s in all_possible_scores]
    plt.bar(all_possible_scores, frequencies, width=1.0, edgecolor='black', align='center', color='skyblue')

    ax.set_title('Distribution of Exam Marks (Scaled to 0-10)', fontsize=14, loc='left')
    ax.set_xlabel('Mark (0-10 Scale)', fontsize=11)
    ax.set_ylabel('Number of Students', fontsize=11)
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(top=max(frequencies) * 1.1 if max(frequencies, default=0) > 0 else 1)
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
    return plot_filename


def _save_and_print_final_marks(df: pd.DataFrame, output_dir: str) -> None:
    print("\n--- Student Marks (0-10 Scale) ---")
    marks_df = df[['student_id', 'student_name', 'score_adjusted', 'max_score_adjusted', 'correct_count',
                   'incorrect_count', 'na_count', 'mark_clipped']].copy()
    marks_df.rename(columns={'mark_clipped': 'mark', 'score_adjusted': 'score', 'max_score_adjusted': 'max_score',
                             'correct_count': 'correct', 'incorrect_count': 'incorrect', 'na_count': 'NA'}, inplace=True)
    final_csv_path = os.path.join(output_dir, "final_marks.csv")
    marks_df.to_csv(final_csv_path, index=False)
    logging.info(f"Final marks saved to {os.path.abspath(final_csv_path)}")
    marks_df.index = range(1, len(marks_df) + 1)
    print(tabulate(marks_df, headers='keys', tablefmt='psql', floatfmt=".2f"))


def analyze_results(
    csv_filepath,
    output_dir=".",
    exam_dir: Optional[str] = None,
    solutions_per_model: Optional[Dict] = None,
    max_score: Optional[float] = None,
    void_questions_str: Optional[str] = None,
    void_questions_nicely_str: Optional[str] = None,
    penalty: float = 0.0,
    lang: str = "en",
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
    _add_marks(df, solutions_per_model, max_score, void_q_list, void_q_nicely_list, penalty)

    # Keep the annotated answer sheets (shown to students) consistent with the voids.
    sheets = redraw_answer_sheets(output_dir, df, solutions_per_model,
                                  sorted(set(void_q_list) | set(void_q_nicely_list)), lang=lang)
    if sheets:
        logging.info("Redrew %d annotated answer sheet(s) from correction_results.csv.", sheets)

    print("\n--- Descriptive Statistics for Marks (0-10 scale) ---")
    print(df['mark_clipped'].describe())

    plot_filename = _plot_mark_distribution(df, output_dir)
    _save_and_print_final_marks(df, output_dir)
    if solutions_per_model:
        _generate_stats_pdf(df, solutions_per_model, output_dir, mark_plot_path=plot_filename)
