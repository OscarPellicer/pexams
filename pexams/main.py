"""Command-line interface of pexams.

Only argparse and the standard library are imported at module level, so that ``pexams --help``
is fast: each command imports OpenCV, Playwright, pandas or matplotlib when it runs.

``add_generate_pdf_arguments`` and ``add_correct_arguments`` declare the PDF-generation and
scan-correction flags; ptestgen reuses them (with its own defaults) for ``ptestgen export pexams``
and ``ptestgen correct pexams``, and ``run_correction`` runs the correction from parsed flags.
"""
import argparse
import logging
import os
import sys
from typing import Optional

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]
OUTPUT_FORMATS = ["pexams", "rexams", "wooclap", "gift", "md", "moodle"]
# The flags of add_generate_pdf_arguments, which only apply to --to pexams.
PDF_ONLY_ARGUMENTS = ["num_models", "columns", "font_size", "total_students", "keep_html", "generate_fakes",
                      "generate_references", "extra_model_templates", "custom_header", "mc_total_points",
                      "fill_answer_space", "show_points"]
SEPARATOR_NAMES = {"semi": ";", "comma": ",", "tab": "\t", "pipe": "|"}


def _defaults(overrides: Optional[dict], **defaults) -> dict:
    """``defaults`` updated with the caller's ``overrides`` (keyed by argument dest)."""
    unknown = set(overrides or {}) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown argument defaults: {', '.join(sorted(unknown))}")
    return {**defaults, **(overrides or {})}


def add_generate_pdf_arguments(parser: argparse.ArgumentParser, defaults: Optional[dict] = None) -> None:
    """Add the flags of exam PDF generation (title, models, layout, test scans...).

    ``defaults`` overrides the default of any of them by dest, e.g. ``{"num_models": 1}``.
    """
    d = _defaults(defaults, exam_title="Final Exam", exam_course=None, exam_date=None, lang="en", num_models=4,
                  columns=1, font_size="11pt", total_students=0, extra_model_templates=0, keep_html=False,
                  generate_fakes=0, generate_references=False, custom_header=None, mc_total_points=None,
                  fill_answer_space=False, show_points="auto")
    parser.add_argument("--exam-title", type=str, default=d["exam_title"], help="Title of the exam (default: %(default)s).")
    parser.add_argument("--exam-course", type=str, default=d["exam_course"], help="Course name for the exam.")
    parser.add_argument("--exam-date", type=str, default=d["exam_date"], help="Date of the exam.")
    parser.add_argument("--lang", type=str, default=d["lang"],
                        help="Language code of the printed labels, e.g. 'es', 'en', 'ca' (default: %(default)s).")
    parser.add_argument("--num-models", type=int, default=d["num_models"],
                        help="Number of different exam models to generate (default: %(default)s).")
    parser.add_argument("--columns", type=int, default=d["columns"], choices=[1, 2, 3],
                        help="Number of columns of the question pages (default: %(default)s).")
    parser.add_argument("--font-size", type=str, default=d["font_size"],
                        help="Base font size, e.g. '9pt' or '12px' (default: %(default)s).")
    parser.add_argument("--total-students", type=int, default=d["total_students"],
                        help="Total number of students for a single PDF with all the exams (all_exams.pdf).")
    parser.add_argument("--extra-model-templates", type=int, default=d["extra_model_templates"],
                        help="Number of extra answer sheets per model appended to all_exams.pdf.")
    parser.add_argument("--keep-html", action="store_true", default=d["keep_html"],
                        help="Keep the intermediate HTML files.")
    parser.add_argument("--generate-fakes", type=int, default=d["generate_fakes"],
                        help="Number of simulated scans with fake students and answers, to test the correction.")
    parser.add_argument("--generate-references", action="store_true", default=d["generate_references"],
                        help="Generate a reference scan per model with the correct answers marked.")
    parser.add_argument("--custom-header", type=str, default=d["custom_header"],
                        help="Markdown string or path to a .md file to insert before the questions (e.g. instructions).")
    parser.add_argument("--mc-total-points", type=float, default=d["mc_total_points"],
                        help="Total points of the multiple-choice questions, split evenly (open questions keep their "
                             "own points). Cannot be combined with per-question non-default MC points.")
    parser.add_argument("--fill-answer-space", action="store_true", default=d["fill_answer_space"],
                        help="Grow the open-answer boxes of each page to use the free space left at its bottom.")
    parser.add_argument("--show-points", choices=["auto", "always", "never"], default=d["show_points"],
                        help="Print each question's points, e.g. [1.5 pt], after its number. 'auto' (default) prints "
                             "them when there are open-answer questions or questions of different value.")


def add_correct_arguments(parser: argparse.ArgumentParser, defaults: Optional[dict] = None) -> None:
    """Add the flags of scan correction: inputs, voided questions, class list and mark filling,
    header reader. The output directory, penalty, label language and log level are left to the
    caller, which may already have them.

    ``defaults`` overrides the default of any of them by dest, e.g. ``{"id_reader": "trocr"}``.
    """
    d = _defaults(defaults, input_path=None, exam_dir=None, void_questions=None, void_questions_nicely=None,
                  input_csv=None, id_column=None, mark_column=None, name_column=None, simplify_csv=False,
                  fuzzy_id_match=100, name_match_threshold=70.0, id_reader="llm", id_reader_model="claude-opus-5-5",
                  id_reader_effort="low", api_key_env=None, input_encoding="utf-8", input_sep=",",
                  output_decimal_sep=".", only_analysis=False)
    parser.add_argument("--input-path", type=str, nargs="+", default=d["input_path"],
                        help="One or more scanned PDFs and/or folders of PNG/JPG images or PDFs, read in the order given\n"
                             "(files inside a folder are sorted by name). Use it when a batch was scanned into several PDFs.")
    parser.add_argument("--exam-dir", type=str, required=d["exam_dir"] is None, default=d["exam_dir"],
                        help="Directory with the exam models and solutions (the output of 'generate').")
    parser.add_argument("--void-questions", type=str, default=d["void_questions"],
                        help="Comma-separated list of question numbers to remove from score calculation (e.g., '3,4').")
    parser.add_argument("--void-questions-nicely", type=str, default=d["void_questions_nicely"],
                        help="Comma-separated list of question IDs to void 'nicely'. If correct, it counts. "
                             "If incorrect, it's removed from the total score calculation for that student.")
    parser.add_argument("--input-csv", type=str, default=d["input_csv"],
                        help="Class list (CSV/TSV/XLSX) to match the students to and fill with marks.")
    parser.add_argument("--id-column", type=str, default=d["id_column"], help="Column of --input-csv with the student IDs.")
    parser.add_argument("--mark-column", type=str, default=d["mark_column"], help="Column of --input-csv to fill with marks.")
    parser.add_argument("--name-column", type=str, default=d["name_column"],
                        help="Column of --input-csv with the student names (required to match students, and with --simplify-csv).")
    parser.add_argument("--simplify-csv", action="store_true", default=d["simplify_csv"],
                        help="If set, the output CSV will only contain the id, name, and mark columns.")
    parser.add_argument("--fuzzy-id-match", type=int, default=d["fuzzy_id_match"],
                        help="Fuzzy matching threshold (0-100) for student IDs when filling marks (default: %(default)s).")
    parser.add_argument("--name-match-threshold", type=float, default=d["name_match_threshold"],
                        help="Fuzzy matching threshold (0-100) for matching the names read to --input-csv before analysis "
                             "(default: %(default)s).")
    parser.add_argument("--id-reader", choices=["llm", "trocr"], default=d["id_reader"],
                        help="How to read the model, student-ID and name boxes: a vision LLM (default) or local TrOCR.")
    parser.add_argument("--id-reader-model", type=str, default=d["id_reader_model"],
                        help="Model for --id-reader llm (default: %(default)s). Bare 'claude-...' ids use the Anthropic API; "
                             "others use OpenRouter.")
    parser.add_argument("--id-reader-effort", type=str, default=d["id_reader_effort"],
                        help="Reasoning effort for --id-reader llm (low/medium/high; empty for the provider default; "
                             "default: %(default)s).")
    parser.add_argument("--api-key-env", type=str, default=d["api_key_env"],
                        help="Environment variable holding the API key (default: ANTHROPIC_API_KEY or OPENROUTER_API_KEY).")
    parser.add_argument("--input-encoding", type=str, default=d["input_encoding"],
                        help="Encoding of --input-csv (default: %(default)s).")
    parser.add_argument("--input-sep", type=str, default=d["input_sep"],
                        help="Separator of --input-csv (default: comma ','). Use 'semi' for semicolon ';', "
                             "'tab' for tab, 'pipe' for pipe '|', or any other separator.")
    parser.add_argument("--output-decimal-sep", type=str, default=d["output_decimal_sep"],
                        help="Decimal separator for the output marks (default: dot '.'). Use ',' for comma.")
    parser.add_argument("--only-analysis", action="store_true", default=d["only_analysis"],
                        help="Skip image processing and run analysis on existing correction_results.csv.")


def parse_separator(value: str) -> str:
    """A CSV separator given by name ('semi', 'comma', 'tab', 'pipe') or literally."""
    return SEPARATOR_NAMES.get(value, value)


def run_correction(args: argparse.Namespace, lang: Optional[str] = None) -> bool:
    """Correct scans and analyse the results from the flags of :func:`add_correct_arguments`
    plus ``output_dir``, ``penalty`` and ``lang`` (the language code of the labels on the
    annotated sheets; ``lang`` overrides ``args.lang``). Fills --input-csv with the marks when
    asked. Returns True when correction_results.csv was analysed."""
    from pexams import analysis, correct_exams, utils
    from pexams.grades import fill_marks_in_file

    if args.penalty < 0:
        logging.warning("Penalty cannot be negative (it is subtracted). Converting to positive.")
        args.penalty = abs(args.penalty)
    if not args.only_analysis:
        if not args.input_path:
            logging.error("the following arguments are required: --input-path (unless --only-analysis is used)")
            return False
        missing = [p for p in args.input_path if not os.path.exists(p)]
        if missing:
            logging.error(f"Input path not found: {', '.join(missing)}")
            return False
    if not os.path.isdir(args.exam_dir):
        logging.error(f"Exam directory not found: {args.exam_dir}")
        return False

    solutions_full, solutions_simple, max_score = utils.load_solutions(args.exam_dir)
    if not solutions_simple:
        logging.error(f"Could not load solutions from exam directory {args.exam_dir}.")
        return False

    os.makedirs(args.output_dir, exist_ok=True)
    input_sep = parse_separator(args.input_sep)
    if args.only_analysis:
        from pexams.open_responses import sync_index_with_results

        logging.info("Skipping image correction (--only-analysis). Using existing results.")
        sync_index_with_results(args.output_dir)
    elif not correct_exams.correct_exams(
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
        id_reader=args.id_reader,
        id_reader_model=args.id_reader_model,
        id_reader_effort=args.id_reader_effort or None,
        api_key_env=args.api_key_env,
    ):
        logging.error("Exam correction failed.")
        return False

    logging.info("Correction finished. Starting analysis.")
    results_csv = os.path.join(args.output_dir, "correction_results.csv")
    if not os.path.exists(results_csv):
        logging.error(f"Analysis skipped: correction results file not found at {results_csv}")
        return False
    analysis.analyze_results(
        csv_filepath=results_csv,
        max_score=max_score,
        output_dir=args.output_dir,
        void_questions_str=args.void_questions,
        solutions_per_model=solutions_full,
        void_questions_nicely_str=args.void_questions_nicely,
        penalty=args.penalty,
        lang=lang or args.lang,
    )
    if args.input_csv:
        if args.id_column and args.mark_column:
            fill_marks_in_file(
                args.input_csv, args.id_column, args.mark_column, results_csv,
                args.fuzzy_id_match, args.input_encoding, input_sep, args.output_decimal_sep,
                name_col=args.name_column, simplify_csv=args.simplify_csv,
            )
        else:
            logging.warning("--input-csv provided but --id-column or --mark-column missing. Skipping mark filling.")
    return True


# --- Commands -------------------------------------------------------------------------------

def _find_tests_dir():
    """Locate the tests/ directory whether running from project root or installed."""
    for candidate in (os.path.join(os.getcwd(), "tests"),
                      os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests")):
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


def _cmd_test(args):
    _run_pytest(args.output_dir, k_filter="not overflow")


def _cmd_test_overflow(args):
    _run_pytest(args.output_dir, k_filter="overflow")


def _cmd_correct(args):
    run_correction(args)


def _cmd_generate(args, subparser):
    from pexams import generate_exams, utils
    from pexams.io import gift_converter, md_converter, moodle_xml_converter, rexams_converter, wooclap_converter
    from pexams.io.loader import load_and_prepare_questions

    questions = load_and_prepare_questions(args.input_file)
    if questions is None:
        return
    if args.shuffle_answers is None:
        logging.error("--shuffle-answers cannot be None.")
        return
    # Set global seeds for all exporters
    utils.set_seeds(seed_questions=args.shuffle_questions, seed_answers=args.shuffle_answers)

    output_fmt, out_dir = args.to, args.output_dir
    base_dir = os.path.dirname(os.path.abspath(args.input_file))
    if output_fmt != "pexams":
        for name in PDF_ONLY_ARGUMENTS:
            if getattr(args, name) != subparser.get_default(name):
                logging.warning(f"Argument '--{name}' is ignored for format '{output_fmt}'.")

    if output_fmt == "pexams":
        generate_exams.generate_exams(
            questions=questions,
            output_dir=out_dir,
            num_models=args.num_models,
            exam_title=args.exam_title,
            exam_course=args.exam_course,
            exam_date=args.exam_date,
            columns=args.columns,
            lang=args.lang,
            keep_html=args.keep_html or args.log_level == 'DEBUG',
            font_size=args.font_size,
            generate_fakes=args.generate_fakes,
            generate_references=args.generate_references,
            total_students=args.total_students,
            extra_model_templates=args.extra_model_templates,
            custom_header=args.custom_header,
            markdown_asset_base_dir=base_dir,
            mc_total_points=args.mc_total_points,
            fill_answer_space=args.fill_answer_space,
            show_points=args.show_points,
        )
        return

    # The pexams format shuffles per model; the other formats are shuffled once here.
    utils.shuffle_questions_list(questions)
    for q in questions:
        utils.shuffle_options_for_question(q)
    if output_fmt == "rexams":
        rexams_converter.prepare_for_rexams(questions, out_dir, base_dir=base_dir)
    elif output_fmt == "wooclap":
        wooclap_converter.convert_to_wooclap(questions, os.path.join(out_dir, "wooclap_export.csv"))
    elif output_fmt == "gift":
        gift_converter.convert_to_gift(questions, os.path.join(out_dir, "questions.gift"), base_dir=base_dir)
    elif output_fmt == "md":
        md_converter.save_questions_to_md(questions, os.path.join(out_dir, "questions.md"))
    elif output_fmt == "moodle":
        moodle_xml_converter.convert_to_moodle_xml(questions, os.path.join(out_dir, "questions.xml"), base_dir=base_dir)


def _cmd_correct_online(args):
    from pexams import analysis, utils
    from pexams.io.loader import load_and_prepare_questions
    from pexams.io.online_results import parse_moodle_results, parse_wooclap_results

    questions = load_and_prepare_questions(args.input_file)
    if questions is None:
        return
    solutions_full, _, max_score = utils.create_solutions_from_questions(questions)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.platform == "wooclap":
        results_df = parse_wooclap_results(results_path=args.results, questions=questions,
                                           fuzzy_threshold=args.fuzzy_threshold / 100.0,
                                           encoding=args.encoding, sep=args.sep)
    else:
        question_order = None
        if args.question_order:
            question_order = [int(x.strip()) - 1 for x in args.question_order.split(",")]
        results_df = parse_moodle_results(results_path=args.results, questions=questions,
                                          question_order=question_order, encoding=args.encoding, sep=args.sep)

    # Save correction_results.csv in the standard pexams format, then the standard analysis + PDF report.
    correction_csv = os.path.join(args.output_dir, "correction_results.csv")
    results_df.to_csv(correction_csv, index=False)
    print(f"Correction results saved to: {correction_csv}")
    analysis.analyze_results(csv_filepath=correction_csv, max_score=max_score, output_dir=args.output_dir,
                             solutions_per_model=solutions_full, penalty=args.penalty)


def _cmd_moodle_feedback_zip(args):
    from pexams.feedback_zip import create_moodle_feedback_csv, create_moodle_feedback_zip

    moodle_sep = ";" if args.moodle_sep == "semi" else args.moodle_sep
    common = dict(
        moodle_csv=args.moodle_csv, correction_dir=args.correction_dir, id_column=args.id_column,
        participant_column=args.participant_column, name_column=args.name_column, mark_column=args.mark_column,
        images_dir=args.images_dir, marks_csv=args.marks_csv, moodle_encoding=args.moodle_encoding,
        moodle_sep=moodle_sep, max_mark=args.max_mark, image_id_column=args.image_id_column, overwrite=args.overwrite,
    )
    outputs = []
    if args.feedback_mode in ("zip", "both"):
        if not args.output_zip:
            logging.error("--output-zip is required when --feedback-mode is zip or both.")
            return
        try:
            result = create_moodle_feedback_zip(output_zip=args.output_zip,
                                                feedback_file_format=args.feedback_file_format, **common)
        except Exception as e:
            logging.error("Failed to create Moodle feedback zip: %s", e)
            return
        outputs.append(("zip", result.output_zip, result.manifest_csv, result))

    if args.feedback_mode in ("csv", "both"):
        output_csv = args.output_csv or os.path.join(args.correction_dir, "moodle_feedback_base64.csv")
        try:
            result = create_moodle_feedback_csv(output_csv=output_csv, feedback_column=args.feedback_column,
                                                grade_column=args.grade_column or None,
                                                csv_decimal_sep=args.csv_decimal_sep, **common)
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


# --- Parser ---------------------------------------------------------------------------------

def _add_log_level(parser, help="Set the logging level."):
    parser.add_argument("--log-level", type=str, default="INFO", choices=LOG_LEVELS, help=help)


def _add_correct_parser(subparsers):
    parser = subparsers.add_parser("correct", formatter_class=argparse.RawTextHelpFormatter,
                                   help="Correct scanned exam answer sheets from a PDF file or a folder of images.")
    add_correct_arguments(parser)
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the correction results CSV and any debug images.")
    parser.add_argument("--penalty", type=float, default=0.0,
                        help="Penalty per wrong answer as a fraction of the question's points (e.g. 0.33333). Default is 0.0.")
    parser.add_argument("--lang", type=str, default="en",
                        help="Language code of the labels drawn on the annotated answer sheets (e.g. 'ANULADA' for voided questions with 'es').")
    _add_log_level(parser)
    parser.set_defaults(func=_cmd_correct)


def _add_generate_parser(subparsers):
    parser = subparsers.add_parser("generate", formatter_class=argparse.RawTextHelpFormatter,
                                   help="Generate exams or export questions to other formats.")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to the input file containing questions (Markdown .md or JSON).")
    parser.add_argument("--to", type=str, default="pexams", choices=OUTPUT_FORMATS,
                        help="Output format. Default is 'pexams' (PDF generation).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the output.")
    parser.add_argument("--shuffle-questions", type=int, default=42,
                        help="Seed for shuffling questions (default: 42). If not provided, questions are not shuffled (input order preserved).")
    parser.add_argument("--shuffle-answers", type=int, default=42,
                        help="Seed for shuffling answers (options) (default: 42). Must not be None, answers must be shuffled.")
    add_generate_pdf_arguments(parser)
    _add_log_level(parser)
    parser.set_defaults(func=lambda args: _cmd_generate(args, parser))


def _add_correct_online_parser(subparsers):
    parser = subparsers.add_parser(
        "correct-online", formatter_class=argparse.RawTextHelpFormatter,
        help="Ingest results from online platforms (Wooclap, Moodle) and generate a statistics report.")
    platforms = parser.add_subparsers(dest="platform", required=True, help="The platform from which results were exported.")

    # Shared arguments for all online-correction sub-subparsers
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input-file", required=True,
                        help="Path to the questions file (.md or .json) used when exporting to this platform.")
    common.add_argument("--results", required=True, help="Path to the results file exported from the platform (CSV or XLSX).")
    common.add_argument("--output-dir", required=True,
                        help="Directory where correction_results.csv and stats_report.pdf will be saved.")
    common.add_argument("--penalty", type=float, default=0.0,
                        help="Penalty per wrong answer as a fraction of the question's points (default: 0.0).")
    common.add_argument("--encoding", default="auto", help="Encoding of the results file, e.g. 'utf-8', 'latin1'. Default: auto-detect.")
    common.add_argument("--sep", default="auto", help="CSV separator, e.g. ',' or ';'. Default: auto-detect.")
    common.add_argument("--log-level", default="INFO", choices=LOG_LEVELS, help="Logging verbosity.")

    wooclap = platforms.add_parser("wooclap", parents=[common],
                                   help="Ingest Wooclap results (CSV/XLSX exported from 'Export to Excel').")
    wooclap.add_argument("--fuzzy-threshold", type=int, default=80,
                         help="Minimum similarity score (0–100) for fuzzy matching of question texts in column headers. Default: 80.")
    moodle = platforms.add_parser("moodle", parents=[common],
                                  help="Ingest Moodle quiz results (CSV/XLSX downloaded from Results > Responses).")
    moodle.add_argument("--question-order", default=None,
                        help="Comma-separated list of 1-based question indices that maps 'Resposta 1' to questions[index-1], etc. "
                             "Default: sequential order (1,2,3,...).")
    parser.set_defaults(func=_cmd_correct_online)


def _add_feedback_zip_parser(subparsers):
    parser = subparsers.add_parser(
        "moodle-feedback-zip", formatter_class=argparse.RawTextHelpFormatter,
        help="Create a Moodle assignment feedback-files zip from corrected pexams scans and final marks.")
    add = parser.add_argument
    add("--moodle-csv", required=True, help="CSV/XLSX exported from the Moodle assignment grading table.")
    add("--correction-dir", required=True, help="Directory containing final_marks.csv and scanned_pages/ from 'pexams correct'.")
    add("--output-zip", help="Path of the Moodle feedback zip to create. Required for --feedback-mode zip/both.")
    add("--output-csv", help="Path of the experimental Moodle feedback CSV to create. Default: <correction-dir>/moodle_feedback_base64.csv.")
    add("--feedback-mode", default="zip", choices=["zip", "csv", "both"],
        help="Output mode: feedback-file zip, experimental base64 feedback CSV, or both. Default: zip.")
    add("--id-column", default="Número ID", help="Student ID column in the Moodle export (default: 'Número ID').")
    add("--participant-column", default="Identificador", help="Moodle participant identifier column (default: 'Identificador').")
    add("--name-column", default="Nom complet", help="Student name column in the Moodle export (default: 'Nom complet').")
    add("--mark-column", default="mark", help="Mark column in final_marks.csv (default: 'mark').")
    add("--feedback-column", default="Comentaris de retroacció.",
        help="Feedback comments column for CSV mode (default: 'Comentaris de retroacció.').")
    add("--grade-column", default="Qualificació", help="Grade column to fill in CSV mode. Pass an empty string to skip grade filling.")
    add("--image-id-column", default="student_id",
        help="Student ID column in final_marks.csv and scanned_pages filenames (default: 'student_id').")
    add("--images-dir", default=None, help="Directory with annotated PNG/JPG files. Default: <correction-dir>/scanned_pages.")
    add("--marks-csv", default=None, help="Marks CSV to use. Default: <correction-dir>/final_marks.csv.")
    add("--moodle-encoding", default="utf-8", help="Encoding of the Moodle CSV (default: utf-8).")
    add("--moodle-sep", default=",", help="Separator for the Moodle CSV (default: comma). Use 'semi' for ';'.")
    add("--max-mark", default="10", help="Text shown after the mark in the generated feedback header (default: 10).")
    add("--feedback-file-format", default="pdf", choices=["pdf", "png"], help="File format to put inside the Moodle feedback ZIP. Default: pdf.")
    add("--csv-decimal-sep", default=",", help="Decimal separator for marks written to the CSV grade column (default: comma).")
    add("--overwrite", action="store_true", help="Replace the output zip if it already exists.")
    _add_log_level(parser)
    parser.set_defaults(func=_cmd_moodle_feedback_zip)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pexams: Generate and correct exams using Python, Playwright, and OpenCV.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_correct_parser(subparsers)

    test_parser = subparsers.add_parser("test", formatter_class=argparse.RawTextHelpFormatter,
                                        help="Run a full generate/correct cycle using the bundled sample files.")
    test_parser.add_argument("--output-dir", type=str, default="./pexams_test_output", help="Directory to save the test output.")
    test_parser.set_defaults(func=_cmd_test)
    overflow_parser = subparsers.add_parser("test-overflow", formatter_class=argparse.RawTextHelpFormatter,
                                            help="Run overflow tests for text and question limits.")
    overflow_parser.add_argument("--output-dir", type=str, default="./pexams_test_overflow",
                                 help="Directory to save the overflow test output.")
    overflow_parser.set_defaults(func=_cmd_test_overflow)

    _add_generate_parser(subparsers)
    _add_correct_online_parser(subparsers)
    _add_feedback_zip_parser(subparsers)
    return parser


def main():
    """Main CLI entry point for the pexams library."""
    args = build_parser().parse_args()

    from pexams.env import load_project_dotenv
    load_project_dotenv()
    log_level = getattr(logging, getattr(args, "log_level", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    args.func(args)


if __name__ == "__main__":
    main()
