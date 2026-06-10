import argparse
import logging
import os
import sys
import random

# Attempt to import pandas for data handling
try:
    import pandas as pd
except ImportError:
    pd = None

from pexams import correct_exams
from pexams import generate_exams
from pexams import analysis
from pexams import utils
from pexams.io import md_converter, rexams_converter, wooclap_converter, gift_converter, moodle_xml_converter
from pexams.io.loader import load_and_prepare_questions
from pexams.grades import fill_marks_in_file
from pexams.feedback_zip import create_moodle_feedback_csv, create_moodle_feedback_zip


def _find_tests_dir():
    """Locate the tests/ directory whether running from project root or installed."""
    candidate = os.path.join(os.getcwd(), "tests")
    if os.path.isdir(candidate):
        return candidate
    candidate = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests")
    if os.path.isdir(candidate):
        return candidate
    return None


def _run_pytest(output_dir, k_filter=None):
    try:
        import pytest
    except ImportError:
        logging.error("pytest is required to run tests. Install it with: pip install pytest")
        return

    tests_dir = _find_tests_dir()
    if tests_dir is None:
        logging.error(
            "Could not find the tests/ directory. "
            "Run pexams from the project root, or install the package in editable mode."
        )
        return

    pytest_args = [tests_dir, "-v", f"--output-dir={output_dir}"]
    if k_filter:
        pytest_args += ["-k", k_filter]

    # Shut down any logging handlers set up by the CLI before handing off to
    # pytest, which will reinitialise its own I/O.  Without this the parent
    # process's stream handles become stale inside pytest's captured output
    # context, producing a flood of OSError: [WinError 6] messages.
    logging.shutdown()
    sys.exit(pytest.main(pytest_args))


def main():
    """Main CLI entry point for the pexams library."""
    
    parser = argparse.ArgumentParser(description="Pexams: Generate and correct exams using Python, Playwright, and OpenCV.")
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Correction Command ---
    correct_parser = subparsers.add_parser( "correct", formatter_class=argparse.RawTextHelpFormatter, 
        help="Correct scanned exam answer sheets from a PDF file or a folder of images.")
    correct_parser.add_argument( "--input-path", type=str, required=False,
        help="Path to the single PDF file or a folder containing scanned answer sheets as PNG/JPG images.")
    correct_parser.add_argument( "--exam-dir", type=str, required=True,
        help="Path to the directory containing exam models and solutions (e.g., the output from 'generate').")
    correct_parser.add_argument( "--output-dir", type=str, required=True,
        help="Directory to save the correction results CSV and any debug images.")  
    correct_parser.add_argument( "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Set the logging level.")
    correct_parser.add_argument( "--void-questions", type=str, default=None,
            help="Comma-separated list of question numbers to remove from score calculation (e.g., '3,4').")
    correct_parser.add_argument( "--void-questions-nicely", type=str, default=None,
        help="Comma-separated list of question IDs to void 'nicely'. If correct, it counts. " \
        "If incorrect, it's removed from the total score calculation for that student.")
    correct_parser.add_argument( "--input-csv", type=str, required=False,
        help="Path to an input CSV/TSV/XLSX file to fill with marks.")
    correct_parser.add_argument( "--id-column", type=str, required=False,
        help="Column name in input-csv containing student IDs.")
    correct_parser.add_argument( "--mark-column", type=str, required=False,
        help="Column name in input-csv to fill with marks.")
    correct_parser.add_argument( "--name-column", type=str, required=False,
        help="Column name in input-csv containing student names (required if --simplify-csv is used).")
    correct_parser.add_argument( "--simplify-csv", action="store_true",
        help="If set, the output CSV will only contain the id, name, and mark columns.")
    correct_parser.add_argument( "--fuzzy-id-match", type=int, default=100,
        help="Fuzzy matching threshold (0-100) for student IDs.")
    correct_parser.add_argument( "--name-match-threshold", type=float, default=70.0,
        help="Fuzzy matching threshold (0-100) for matching OCR student names to --input-csv before analysis.")
    correct_parser.add_argument( "--use-llm-name-ocr", action="store_true",
        help="Use OpenRouter vision OCR for student names before roster matching. Default: local OCR only.")
    correct_parser.add_argument( "--openrouter-name-model", type=str, default="google/gemini-3-flash-preview",
        help="OpenRouter vision model for --use-llm-name-ocr. Default: google/gemini-3-flash-preview.")
    correct_parser.add_argument( "--penalty", type=float, default=0.0,
        help="Score penalty for wrong answers (positive float, e.g. 0.33333). Default is 0.0.")
    correct_parser.add_argument( "--input-encoding", type=str, default="utf-8",
        help="Encoding of the input CSV file (default: utf-8).")
    correct_parser.add_argument( "--input-sep", type=str, default=",",
        help="Separator for the input CSV file (default: comma ','). Use 'semi' for semicolon ';', "+\
            "'tab' for tab '\t', 'pipe' for pipe '|', or any other separator.")
    correct_parser.add_argument( "--output-decimal-sep", type=str, default=".",
        help="Decimal separator for the output marks (default: dot '.'). Use ',' for comma.")
    correct_parser.add_argument( "--only-analysis", action="store_true",    
        help="Skip image processing and run analysis on existing correction_results.csv.")

    # --- Test Command ---    
    test_parser = subparsers.add_parser( "test", formatter_class=argparse.RawTextHelpFormatter, 
        help="Run a full generate/correct cycle using the bundled sample files.")
    test_parser.add_argument( "--output-dir", type=str, default="./pexams_test_output",
        help="Directory to save the test output.")

    # --- Test Overflow Command ---
    test_overflow_parser = subparsers.add_parser( "test-overflow", formatter_class=argparse.RawTextHelpFormatter, 
        help="Run overflow tests for text and question limits.")
    test_overflow_parser.add_argument( "--output-dir", type=str, default="./pexams_test_overflow",
        help="Directory to save the overflow test output.")

    # --- Generation/Convert Command ---
    generate_parser = subparsers.add_parser( "generate", formatter_class=argparse.RawTextHelpFormatter, 
        help="Generate exams or export questions to other formats.")
    generate_parser.add_argument( "--input-file", type=str, required=True, 
    help="Path to the input file containing questions (Markdown .md or JSON).")
    generate_parser.add_argument( "--to", type=str, default="pexams", choices=["pexams", "rexams", "wooclap", "gift", "md", "moodle"],
        help="Output format. Default is 'pexams' (PDF generation).")
    generate_parser.add_argument( "--output-dir", type=str, required=True, help="Directory to save the output.")
    generate_parser.add_argument("--exam-title", type=str, default="Final Exam", help="Title of the exam.")
    generate_parser.add_argument("--exam-course", type=str, default=None, help="Course name for the exam.")
    generate_parser.add_argument("--exam-date", type=str, default=None, help="Date of the exam.")
    generate_parser.add_argument("--lang", type=str, default="en", help="Language for the answer sheet / output.")
    generate_parser.add_argument("--num-models", type=int, default=4, help="Number of different exam models to generate (pexams only).")
    generate_parser.add_argument("--columns", type=int, default=1, choices=[1, 2, 3], help="Number of columns (pexams only).")
    generate_parser.add_argument("--font-size", type=str, default="11pt", help="Base font size (pexams only).")
    generate_parser.add_argument("--total-students", type=int, default=0, help="Total number of students for mass PDF generation (pexams only).")
    generate_parser.add_argument("--extra-model-templates", type=int, default=0, help="Number of extra template sheets to generate per model (pexams only).")
    generate_parser.add_argument("--shuffle-questions", type=int, default=42, help="Seed for shuffling questions (default: 42). If not provided, questions are not shuffled (input order preserved).")
    generate_parser.add_argument("--shuffle-answers", type=int, default=42, help="Seed for shuffling answers (options) (default: 42). Must not be None, answers must be shuffled.")
    generate_parser.add_argument("--keep-html", action="store_true", help="Keep intermediate HTML files (pexams only).")
    generate_parser.add_argument("--generate-fakes", type=int, default=0, help="Generate simulated scans (pexams only).")
    generate_parser.add_argument("--generate-references", action="store_true", help="Generate reference scan (pexams only).")
    generate_parser.add_argument("--custom-header", type=str, default=None, help="Markdown string or path to .md file to insert before questions.")
    generate_parser.add_argument("--mc-total-points", type=float, default=None, help="Total points assigned to all multiple-choice questions. Cannot be combined with per-question non-default MC points.")
    generate_parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    
    # --- Correct-Online Command ---
    correct_online_parser = subparsers.add_parser(
        "correct-online",
        formatter_class=argparse.RawTextHelpFormatter,
        help="Ingest results from online platforms (Wooclap, Moodle) and generate a statistics report.",
    )
    correct_online_sub = correct_online_parser.add_subparsers(
        dest="platform", required=True, help="The platform from which results were exported."
    )

    # Shared arguments for all online-correction sub-subparsers
    _online_common = argparse.ArgumentParser(add_help=False)
    _online_common.add_argument(
        "--input-file", required=True,
        help="Path to the questions file (.md or .json) used when exporting to this platform.",
    )
    _online_common.add_argument(
        "--results", required=True,
        help="Path to the results file exported from the platform (CSV or XLSX).",
    )
    _online_common.add_argument(
        "--output-dir", required=True,
        help="Directory where correction_results.csv and stats_report.pdf will be saved.",
    )
    _online_common.add_argument(
        "--penalty", type=float, default=0.0,
        help="Score penalty deducted per wrong answer (default: 0.0).",
    )
    _online_common.add_argument(
        "--encoding", default="auto",
        help="Encoding of the results file, e.g. 'utf-8', 'latin1'. Default: auto-detect.",
    )
    _online_common.add_argument(
        "--sep", default="auto",
        help="CSV separator, e.g. ',' or ';'. Default: auto-detect.",
    )
    _online_common.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    # Wooclap sub-subparser
    wo_parser = correct_online_sub.add_parser(
        "wooclap", parents=[_online_common],
        help="Ingest Wooclap results (CSV/XLSX exported from 'Export to Excel').",
    )
    wo_parser.add_argument(
        "--fuzzy-threshold", type=int, default=80,
        help=(
            "Minimum similarity score (0–100) for fuzzy matching of question "
            "texts in column headers. Default: 80."
        ),
    )

    # Moodle sub-subparser
    mo_parser = correct_online_sub.add_parser(
        "moodle", parents=[_online_common],
        help="Ingest Moodle quiz results (CSV/XLSX downloaded from Results > Responses).",
    )
    mo_parser.add_argument(
        "--question-order", default=None,
        help=(
            "Comma-separated list of 1-based question indices that maps "
            "'Resposta 1' to questions[index-1], etc. "
            "Default: sequential order (1,2,3,...)."
        ),
    )

    # --- Moodle Feedback Zip Command ---
    feedback_zip_parser = subparsers.add_parser(
        "moodle-feedback-zip",
        formatter_class=argparse.RawTextHelpFormatter,
        help="Create a Moodle assignment feedback-files zip from corrected pexams scans and final marks.",
    )
    feedback_zip_parser.add_argument(
        "--moodle-csv", required=True,
        help="CSV/XLSX exported from the Moodle assignment grading table.",
    )
    feedback_zip_parser.add_argument(
        "--correction-dir", required=True,
        help="Directory containing final_marks.csv and scanned_pages/ from 'pexams correct'.",
    )
    feedback_zip_parser.add_argument(
        "--output-zip",
        help="Path of the Moodle feedback zip to create. Required for --feedback-mode zip/both.",
    )
    feedback_zip_parser.add_argument(
        "--output-csv",
        help="Path of the experimental Moodle feedback CSV to create. Default: <correction-dir>/moodle_feedback_base64.csv.",
    )
    feedback_zip_parser.add_argument(
        "--feedback-mode", default="zip", choices=["zip", "csv", "both"],
        help="Output mode: feedback-file zip, experimental base64 feedback CSV, or both. Default: zip.",
    )
    feedback_zip_parser.add_argument(
        "--id-column", default="Número ID",
        help="Student ID column in the Moodle export (default: 'Número ID').",
    )
    feedback_zip_parser.add_argument(
        "--participant-column", default="Identificador",
        help="Moodle participant identifier column (default: 'Identificador').",
    )
    feedback_zip_parser.add_argument(
        "--name-column", default="Nom complet",
        help="Student name column in the Moodle export (default: 'Nom complet').",
    )
    feedback_zip_parser.add_argument(
        "--mark-column", default="mark",
        help="Mark column in final_marks.csv (default: 'mark').",
    )
    feedback_zip_parser.add_argument(
        "--feedback-column", default="Comentaris de retroacció.",
        help="Feedback comments column for CSV mode (default: 'Comentaris de retroacció.').",
    )
    feedback_zip_parser.add_argument(
        "--grade-column", default="Qualificació",
        help="Grade column to fill in CSV mode. Pass an empty string to skip grade filling.",
    )
    feedback_zip_parser.add_argument(
        "--image-id-column", default="student_id",
        help="Student ID column in final_marks.csv and scanned_pages filenames (default: 'student_id').",
    )
    feedback_zip_parser.add_argument(
        "--images-dir", default=None,
        help="Directory with annotated PNG/JPG files. Default: <correction-dir>/scanned_pages.",
    )
    feedback_zip_parser.add_argument(
        "--marks-csv", default=None,
        help="Marks CSV to use. Default: <correction-dir>/final_marks.csv.",
    )
    feedback_zip_parser.add_argument(
        "--moodle-encoding", default="utf-8",
        help="Encoding of the Moodle CSV (default: utf-8).",
    )
    feedback_zip_parser.add_argument(
        "--moodle-sep", default=",",
        help="Separator for the Moodle CSV (default: comma). Use 'semi' for ';'.",
    )
    feedback_zip_parser.add_argument(
        "--max-mark", default="10",
        help="Text shown after the mark in the generated feedback header (default: 10).",
    )
    feedback_zip_parser.add_argument(
        "--feedback-file-format", default="pdf", choices=["pdf", "png"],
        help="File format to put inside the Moodle feedback ZIP. Default: pdf.",
    )
    feedback_zip_parser.add_argument(
        "--csv-decimal-sep", default=",",
        help="Decimal separator for marks written to the CSV grade column (default: comma).",
    )
    feedback_zip_parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace the output zip if it already exists.",
    )
    feedback_zip_parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level.",
    )

    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper() if hasattr(args, 'log_level') else 'INFO', logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.command == "test":
        _run_pytest(args.output_dir, k_filter="not overflow")

    elif args.command == "test-overflow":
        _run_pytest(args.output_dir, k_filter="overflow")

    elif args.command == "correct":
        if args.penalty < 0:
            logging.warning("Penalty cannot be negative (it is subtracted). Converting to positive.")
            args.penalty = abs(args.penalty)

        if not args.only_analysis:
             if not args.input_path:
                logging.error("the following arguments are required: --input-path (unless --only-analysis is used)")
                return
             if not os.path.exists(args.input_path):
                logging.error(f"Input path not found: {args.input_path}")
                return
             if not os.path.exists(args.exam_dir):
                logging.error(f"Exam directory not found: {args.exam_dir}")
                return

        
        if not os.path.isdir(args.exam_dir):
            logging.error(f"Exam directory not found: {args.exam_dir}")
            return
            
        solutions_full, solutions_simple, max_score = utils.load_solutions(args.exam_dir)
        if not solutions_simple:
            return

        os.makedirs(args.output_dir, exist_ok=True)
        input_sep =  ';' if args.input_sep == 'semi' else \
                     ',' if args.input_sep == 'comma' else \
                     '\t' if args.input_sep == 'tab' else \
                     '|' if args.input_sep == 'pipe' else \
                     args.input_sep
        
        if args.only_analysis:
             logging.info("Skipping image correction (--only-analysis). Using existing results.")
             correction_success = True
        else:
            correction_success = correct_exams.correct_exams(
                input_path=args.input_path,
                solutions_per_model=solutions_simple,
                output_dir=args.output_dir,
                questions_dir=args.exam_dir,
                roster_csv=args.input_csv,
                roster_id_column=args.id_column,
                roster_name_column=args.name_column,
                roster_encoding=args.input_encoding,
                roster_sep=input_sep,
                name_match_threshold=args.name_match_threshold,
                use_llm_name_ocr=args.use_llm_name_ocr,
                openrouter_name_model=args.openrouter_name_model,
            )
        
        if correction_success:
            logging.info("Correction finished. Starting analysis.")
            results_csv = os.path.join(args.output_dir, "correction_results.csv")
            if os.path.exists(results_csv):
                analysis.analyze_results(
                    csv_filepath=results_csv,
                    max_score=max_score,
                    output_dir=args.output_dir,
                    void_questions_str=args.void_questions,
                    solutions_per_model=solutions_full,
                    void_questions_nicely_str=args.void_questions_nicely,
                    penalty=args.penalty
                )
                
                # Input CSV Filling
                if args.input_csv:
                    if args.id_column and args.mark_column:
                         fill_marks_in_file(
                             args.input_csv, args.id_column, args.mark_column, results_csv, 
                             args.fuzzy_id_match, args.input_encoding, input_sep, args.output_decimal_sep,
                             name_col=args.name_column, simplify_csv=args.simplify_csv
                         )
                    else:
                        logging.warning("--input-csv provided but --id-column or --mark-column missing. Skipping mark filling.")
            else:
                logging.error(f"Analysis skipped: correction results file not found at {results_csv}")
    
    elif args.command == "generate":
        questions = load_and_prepare_questions(args.input_file)
        if questions is None:
            return

        if args.shuffle_answers is None:
            logging.error("--shuffle-answers cannot be None.")
            return

        # Set global seeds for all exporters
        utils.set_seeds(seed_questions=args.shuffle_questions, seed_answers=args.shuffle_answers)
        
        output_fmt = args.to
        out_dir = args.output_dir
        
        # Helper to warn about ignored arguments
        def check_arg(name, used_formats):
            if output_fmt not in used_formats and getattr(args, name) != parser.get_default(name):
                logging.warning(f"Argument '--{name}' is ignored for format '{output_fmt}'.")

        # Arguments specific to pexams
        pexams_args = ["num_models", "columns", "font_size", "total_students", "keep_html", "generate_fakes", "generate_references", "extra_model_templates", "custom_header", "mc_total_points"]
        for arg in pexams_args:
            check_arg(arg, ["pexams"])
            
        if output_fmt == "pexams":
            keep_html = args.keep_html or (hasattr(args, 'log_level') and args.log_level == 'DEBUG')
            generate_exams.generate_exams(
                questions=questions,
                output_dir=out_dir,
                num_models=args.num_models,
                exam_title=args.exam_title,
                exam_course=args.exam_course,
                exam_date=args.exam_date,
                columns=args.columns,
                lang=args.lang,
                keep_html=keep_html,
                font_size=args.font_size,
                generate_fakes=args.generate_fakes,
                generate_references=args.generate_references,
                total_students=args.total_students,
                extra_model_templates=args.extra_model_templates,
                custom_header=args.custom_header,
                markdown_asset_base_dir=os.path.dirname(os.path.abspath(args.input_file)),
                mc_total_points=args.mc_total_points,
            )
        else:
            # For non-pexams formats, we apply the shuffling here before passing to converter.
            # (Note: pexams format handles shuffling internally to support multiple models)
            
            # 1. Shuffle Questions (order)
            utils.shuffle_questions_list(questions)
            
            # 2. Shuffle Answers (options)
            for q in questions:
                utils.shuffle_options_for_question(q)
                
            if output_fmt == "rexams":
                rexams_converter.prepare_for_rexams(questions, out_dir)
            elif output_fmt == "wooclap":
                wooclap_file = os.path.join(out_dir, "wooclap_export.csv")
                wooclap_converter.convert_to_wooclap(questions, wooclap_file)
            elif output_fmt == "gift":
                gift_file = os.path.join(out_dir, "questions.gift")
                gift_converter.convert_to_gift(questions, gift_file)
            elif output_fmt == "md":
                md_file = os.path.join(out_dir, "questions.md")
                md_converter.save_questions_to_md(questions, md_file)
            elif output_fmt == "moodle":
                moodle_file = os.path.join(out_dir, "questions.xml")
                moodle_xml_converter.convert_to_moodle_xml(questions, moodle_file)
            else:
                logging.error(f"Unknown output format: {output_fmt}. Supported formats: pexams, rexams, wooclap, gift, md, moodle.")

    elif args.command == "correct-online":
        from pexams.io.online_results import parse_wooclap_results, parse_moodle_results

        questions = load_and_prepare_questions(args.input_file)
        if questions is None:
            return

        solutions_full, solutions_simple, max_score = utils.create_solutions_from_questions(questions)

        os.makedirs(args.output_dir, exist_ok=True)

        if args.platform == "wooclap":
            results_df = parse_wooclap_results(
                results_path=args.results,
                questions=questions,
                fuzzy_threshold=args.fuzzy_threshold / 100.0,
                encoding=args.encoding,
                sep=args.sep,
            )
        elif args.platform == "moodle":
            question_order = None
            if args.question_order:
                question_order = [int(x.strip()) - 1 for x in args.question_order.split(",")]
            results_df = parse_moodle_results(
                results_path=args.results,
                questions=questions,
                question_order=question_order,
                encoding=args.encoding,
                sep=args.sep,
            )
        else:
            logging.error("Unknown platform: %s", args.platform)
            return

        # Save correction_results.csv in the standard pexams format
        correction_csv = os.path.join(args.output_dir, "correction_results.csv")
        results_df.to_csv(correction_csv, index=False)
        print(f"Correction results saved to: {correction_csv}")

        # Run the standard pexams analysis + PDF report
        analysis.analyze_results(
            csv_filepath=correction_csv,
            max_score=max_score,
            output_dir=args.output_dir,
            solutions_per_model=solutions_full,
            penalty=args.penalty,
        )

    elif args.command == "moodle-feedback-zip":
        moodle_sep = ";" if args.moodle_sep == "semi" else args.moodle_sep
        outputs = []
        if args.feedback_mode in ("zip", "both"):
            if not args.output_zip:
                logging.error("--output-zip is required when --feedback-mode is zip or both.")
                return
            try:
                result = create_moodle_feedback_zip(
                    moodle_csv=args.moodle_csv,
                    correction_dir=args.correction_dir,
                    output_zip=args.output_zip,
                    id_column=args.id_column,
                    participant_column=args.participant_column,
                    name_column=args.name_column,
                    mark_column=args.mark_column,
                    images_dir=args.images_dir,
                    marks_csv=args.marks_csv,
                    moodle_encoding=args.moodle_encoding,
                    moodle_sep=moodle_sep,
                    max_mark=args.max_mark,
                    image_id_column=args.image_id_column,
                    feedback_file_format=args.feedback_file_format,
                    overwrite=args.overwrite,
                )
            except Exception as e:
                logging.error("Failed to create Moodle feedback zip: %s", e)
                return
            outputs.append(("zip", result.output_zip, result.manifest_csv, result))

        if args.feedback_mode in ("csv", "both"):
            output_csv = args.output_csv
            if not output_csv:
                output_csv = os.path.join(args.correction_dir, "moodle_feedback_base64.csv")
            grade_column = args.grade_column if args.grade_column else None
            try:
                result = create_moodle_feedback_csv(
                    moodle_csv=args.moodle_csv,
                    correction_dir=args.correction_dir,
                    output_csv=output_csv,
                    id_column=args.id_column,
                    participant_column=args.participant_column,
                    name_column=args.name_column,
                    feedback_column=args.feedback_column,
                    grade_column=grade_column,
                    mark_column=args.mark_column,
                    images_dir=args.images_dir,
                    marks_csv=args.marks_csv,
                    moodle_encoding=args.moodle_encoding,
                    moodle_sep=moodle_sep,
                    max_mark=args.max_mark,
                    image_id_column=args.image_id_column,
                    csv_decimal_sep=args.csv_decimal_sep,
                    overwrite=args.overwrite,
                )
            except Exception as e:
                logging.error("Failed to create Moodle feedback CSV: %s", e)
                return
            outputs.append(("csv", result.output_csv, result.manifest_csv, result))

        for kind, path, manifest, result in outputs:
            print(f"Moodle feedback {kind} saved to: {path}")
            print(f"Manifest saved to: {manifest}")
            print(
                f"Added {result.added} feedback item(s). "
                f"Unmatched Moodle rows: {result.unmatched_roster}. "
                f"Unmatched marks/images: {result.unmatched_marks}."
            )


if __name__ == "__main__":
    main()
