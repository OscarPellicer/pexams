"""Integration tests for the full generate / correct / analyze pipeline
and for overflow edge cases.

These tests are run automatically by the CLI commands:

    pexams test [--output-dir DIR]           # runs test_full_pipeline
    pexams test-overflow [--output-dir DIR]  # runs test_overflow_*

They can also be run directly with pytest:

    pytest tests/test_integration.py -v
    pytest tests/test_integration.py -v --output-dir ./my_output
"""

import os

import pandas as pd
import pytest

from pexams import analysis, correct_exams, generate_exams, layout, utils
from pexams.grades import fill_marks_in_file
from pexams.io import (
    gift_converter,
    md_converter,
    moodle_xml_converter,
    rexams_converter,
    wooclap_converter,
)
from pexams.io.loader import load_and_prepare_questions


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_questions():
    """Load bundled sample questions once per test session."""
    questions = load_and_prepare_questions("sample_test.md")
    assert questions, "Failed to load sample_test.md from assets."
    return questions


# ---------------------------------------------------------------------------
# Full pipeline test  (pexams test)
# ---------------------------------------------------------------------------


def test_full_pipeline(output_dir, sample_questions):
    """Full generate / correct / analyze cycle using bundled sample data."""
    questions = sample_questions

    # 1. Exports ---------------------------------------------------------------
    export_map = {
        "rexams":  lambda d: rexams_converter.prepare_for_rexams(questions, d),
        "wooclap": lambda d: wooclap_converter.convert_to_wooclap(questions, os.path.join(d, "w.csv")),
        "gift":    lambda d: gift_converter.convert_to_gift(questions, os.path.join(d, "g.gift")),
        "md":      lambda d: md_converter.save_questions_to_md(questions, os.path.join(d, "q.md")),
        "moodle":  lambda d: moodle_xml_converter.convert_to_moodle_xml(questions, os.path.join(d, "m.xml")),
    }
    for fmt, run_export in export_map.items():
        out = os.path.join(output_dir, f"export_{fmt}")
        os.makedirs(out, exist_ok=True)
        run_export(out)

    # 2. Generate exams with simulated scans -----------------------------------
    utils.set_seeds(seed_questions=None, seed_answers=42)
    exam_dir = os.path.join(output_dir, "exam_output")
    generate_exams.generate_exams(
        questions=questions,
        output_dir=exam_dir,
        num_models=2,
        generate_fakes=4,
        columns=2,
        exam_title="CI Test Exam",
        exam_course="Test Course",
        exam_date="2025-01-01",
        lang="es",
        generate_references=True,
        font_size="10pt",
        total_students=11,
        extra_model_templates=1,
        custom_header=(
            "**Instructions:** Incorrect answers will be penalized by **-0.25 points**.\n"
            "Why do mathematicians confuse Halloween and Christmas? Because $OCT 31 = DEC 25$."
        ),
    )

    # 3. Correct ---------------------------------------------------------------
    correction_dir = os.path.join(output_dir, "correction_results")
    os.makedirs(correction_dir, exist_ok=True)

    solutions_full, solutions_simple, max_score = utils.load_solutions(exam_dir)
    assert solutions_simple, "Failed to load solutions from the generated exam directory."

    success = correct_exams.correct_exams(
        input_path=os.path.join(exam_dir, "simulated_scans"),
        solutions_per_model=solutions_simple,
        output_dir=correction_dir,
        questions_dir=exam_dir,
    )
    assert success, "Exam correction step failed."

    # 4. Analysis --------------------------------------------------------------
    results_csv = os.path.join(correction_dir, "correction_results.csv")
    assert os.path.exists(results_csv), "correction_results.csv was not produced by the corrector."

    analysis.analyze_results(
        csv_filepath=results_csv,
        max_score=max_score,
        output_dir=correction_dir,
        solutions_per_model=solutions_full,
        void_questions_str="1",
        void_questions_nicely_str="2",
    )

    final_marks_path = os.path.join(correction_dir, "final_marks.csv")
    assert os.path.exists(final_marks_path), "final_marks.csv was not produced by the analysis."

    # 5. Fuzzy match & mark filling --------------------------------------------
    df = pd.read_csv(results_csv)
    valid_ids = [str(x) for x in df["student_id"] if "unknown" not in str(x).lower()]
    assert valid_ids, "No valid student IDs found in correction_results.csv."

    # Prefer a student with score > 0 so the mark assertion is meaningful.
    scored_ids = [str(r["student_id"]) for _, r in df.iterrows()
                  if r.get("score", 0) > 0 and "unknown" not in str(r["student_id"]).lower()]
    target_id = scored_ids[0] if scored_ids else valid_ids[0]
    fuzzy_id = target_id[:-1] + ("A" if target_id[-1] != "A" else "B")

    input_csv_path = os.path.join(output_dir, "students_input.csv")
    with open(input_csv_path, "w", encoding="utf-8") as f:
        f.write("student_id,name,mark\n")
        f.write(f"{fuzzy_id},Test Student,0\n")

    fill_marks_in_file(input_csv_path, "student_id", "mark", results_csv, fuzzy_threshold=80)

    # fill_marks_in_file writes to <name>_with_marks.csv, not the original file.
    output_csv_path = (
        os.path.splitext(input_csv_path)[0] + "_with_marks" + os.path.splitext(input_csv_path)[1]
    )
    assert os.path.exists(output_csv_path), "fill_marks_in_file did not produce an output file."
    df_out = pd.read_csv(output_csv_path)
    assert df_out.iloc[0]["mark"] > 0, (
        f"Fuzzy match returned mark=0 for '{fuzzy_id}' (target '{target_id}'). "
        "Check OCR quality or fuzzy threshold."
    )

    # 6. Rerun analysis after manual correction --------------------------------
    # fill_marks_in_file rewrites final_marks.csv replacing the OCR id (target_id)
    # with the roster id (fuzzy_id), so we search for either.
    df_marks = pd.read_csv(final_marks_path)
    row_old = df_marks[df_marks["student_id"].astype(str).isin([target_id, fuzzy_id])]
    assert not row_old.empty, (
        f"Student {target_id!r} (or {fuzzy_id!r}) not found in final_marks.csv."
    )
    old_score = row_old.iloc[0]["score"]

    # correction_results.csv always retains the original OCR student IDs.
    df = pd.read_csv(results_csv)
    row_idx = df.index[df["student_id"].astype(str) == target_id].tolist()[0]
    model_id = str(df.at[row_idx, "model_id"])
    target_q = 3

    assert model_id in solutions_simple and target_q in solutions_simple[model_id], (
        f"Q{target_q} not found in solutions for model {model_id!r}."
    )

    q_correct_char = chr(ord("A") + solutions_simple[model_id][target_q])
    current_answer = str(df.at[row_idx, f"answer_{target_q}"])

    if current_answer == q_correct_char:
        new_answer = "B" if q_correct_char == "A" else "A"
        expected_delta = -1
    else:
        new_answer = q_correct_char
        expected_delta = 1

    df.at[row_idx, f"answer_{target_q}"] = new_answer
    df.to_csv(results_csv, index=False)

    # analyze_results regenerates final_marks.csv from correction_results.csv,
    # so the student will appear under their OCR id (target_id) again.
    analysis.analyze_results(
        csv_filepath=results_csv,
        max_score=max_score,
        output_dir=correction_dir,
        solutions_per_model=solutions_full,
        void_questions_str="1",
        void_questions_nicely_str="2",
    )

    df_marks_new = pd.read_csv(final_marks_path)
    new_score = df_marks_new[
        df_marks_new["student_id"].astype(str) == target_id
    ].iloc[0]["score"]

    assert new_score == old_score + expected_delta, (
        f"Manual correction verification failed: "
        f"expected {old_score + expected_delta}, got {new_score}."
    )


# ---------------------------------------------------------------------------
# Overflow tests  (pexams test-overflow)
# ---------------------------------------------------------------------------

_LONG_TEXT = (
    "This is a very long text that should overflow the available space "
    "in the header fields of the answer sheet template. " * 5
)


@pytest.mark.overflow
def test_overflow_text(output_dir, sample_questions):
    """Long strings in header fields should not crash PDF generation."""
    generate_exams.generate_exams(
        questions=sample_questions,
        output_dir=os.path.join(output_dir, "overflow_text"),
        num_models=1,
        exam_title=_LONG_TEXT,
        exam_course=_LONG_TEXT,
        exam_date=_LONG_TEXT,
        columns=1,
        generate_fakes=0,
    )


@pytest.mark.overflow
def test_overflow_question_count(output_dir, sample_questions):
    """Exceeding MAX_QUESTIONS must raise ValueError."""
    max_q = layout.MAX_QUESTIONS
    many_questions = []
    while len(many_questions) <= max_q:
        for q in sample_questions:
            q_copy = q.model_copy(deep=True)
            q_copy.id = len(many_questions) + 1
            many_questions.append(q_copy)

    with pytest.raises(ValueError):
        generate_exams.generate_exams(
            questions=many_questions,
            output_dir=os.path.join(output_dir, "overflow_questions"),
            num_models=1,
        )


@pytest.mark.overflow
def test_overflow_custom_header(output_dir, sample_questions):
    """A very long custom header should not crash PDF generation."""
    generate_exams.generate_exams(
        questions=sample_questions,
        output_dir=os.path.join(output_dir, "overflow_header"),
        num_models=1,
        columns=1,
        generate_fakes=0,
        custom_header=_LONG_TEXT,
    )
