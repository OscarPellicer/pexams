import argparse
import logging
import json
import os
from pathlib import Path
import glob
import re
from typing import Optional, List

from pexams import correct_exams
from pexams import generate_exams
from pexams import analysis
from pexams.schemas import PexamExam, PexamQuestion
from pydantic import ValidationError
import pexams

def _load_and_prepare_questions(questions_json: str) -> Optional[List[PexamQuestion]]:
    """Loads questions from a JSON file, resolving bundled assets and image paths."""
    questions_path = Path(questions_json)

    # Check if the file exists at the given path. If not, try to find it in the package assets.
    if not questions_path.exists():
        try:
            package_dir = Path(pexams.__file__).parent
            asset_path = package_dir / "assets" / questions_json
            if asset_path.exists():
                questions_path = asset_path
            else:
                raise FileNotFoundError
        except (FileNotFoundError, AttributeError):
            logging.error(f"Questions JSON file not found at '{questions_json}' or as a built-in asset.")
            return None

    try:
        exam = PexamExam.model_validate_json(questions_path.read_text(encoding="utf-8"))
        questions = exam.questions
    except ValidationError as e:
        logging.error(f"Failed to validate questions JSON file: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse questions JSON file: {e}")
        return None
        
    # Resolve paths for images, making them absolute before passing them to the generator.
    json_dir = questions_path.parent
    for q in questions:
        if q.image_source and not Path(q.image_source).is_absolute():
            # First, try to resolve the path relative to the JSON file's directory.
            image_path_rel_json = (json_dir / q.image_source).resolve()
            
            # If that path doesn't exist, try resolving relative to the current working directory.
            image_path_rel_cwd = Path(q.image_source).resolve()

            if image_path_rel_json.exists():
                q.image_source = str(image_path_rel_json)
            elif image_path_rel_cwd.exists():
                q.image_source = str(image_path_rel_cwd)
            else:
                logging.warning(
                    f"Could not find image for question {q.id} at '{q.image_source}'. "
                    f"Checked relative to JSON file and current directory."
                )
    return questions


def main():
    """Main CLI entry point for the pexams library."""
    
    parser = argparse.ArgumentParser(
        description="Pexams: Generate and correct exams using Python, Playwright, and OpenCV."
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Correction Command ---
    correct_parser = subparsers.add_parser(
        "correct",
        help="Correct scanned exam answer sheets from a PDF file or a folder of images.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    correct_parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the single PDF file or a folder containing scanned answer sheets as PNG/JPG images."
    )
    correct_parser.add_argument(
        "--exam-dir",
        type=str,
        required=True,
        help="Path to the directory containing exam models and solutions (e.g., the output from 'generate')."
    )
    correct_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the correction results CSV and any debug images."
    )
    correct_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level."
    )
    correct_parser.add_argument(
        "--void-questions",
        type=str,
        default=None,
        help="Comma-separated list of question numbers to remove from score calculation (e.g., '3,4')."
    )
    correct_parser.add_argument(
        "--void-questions-nicely",
        type=str,
        default=None,
        help="Comma-separated list of question IDs to void 'nicely'. If correct, it counts. If incorrect, it's removed from the total score calculation for that student."
    )

    # --- Test Command ---
    test_parser = subparsers.add_parser(
        "test",
        help="Run a full generate/correct cycle using the bundled sample files."
    )
    test_parser.add_argument(
        "--output-dir",
        type=str,
        default="./pexams_test_output",
        help="Directory to save the test output."
    )

    # --- Generation Command ---
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate exam PDFs from a JSON file of questions."
    )
    generate_parser.add_argument(
        "--questions-json",
        type=str,
        required=True,
        help="Path to the JSON file containing the exam questions."
    )
    generate_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the generated exam PDFs."
    )
    generate_parser.add_argument("--num-models", type=int, default=4, help="Number of different exam models to generate.")
    generate_parser.add_argument("--exam-title", type=str, default="Final Exam", help="Title of the exam.")
    generate_parser.add_argument("--exam-course", type=str, default=None, help="Course name for the exam.")
    generate_parser.add_argument("--exam-date", type=str, default=None, help="Date of the exam.")
    generate_parser.add_argument("--columns", type=int, default=1, choices=[1, 2, 3], help="Number of columns for the questions.")
    generate_parser.add_argument("--font-size", type=str, default="11pt", help="Base font size for the exam (e.g., '10pt', '12px').")
    generate_parser.add_argument("--id-length", type=int, default=10, help="Number of boxes for the student ID.")
    generate_parser.add_argument("--lang", type=str, default="en", help="Language for the answer sheet.")
    generate_parser.add_argument("--keep-html", action="store_true", help="Keep the intermediate HTML files.")
    generate_parser.add_argument("--generate-fakes", type=int, default=0, help="Generate a number of simulated scans with fake answers for testing the correction process. Default is 0.")
    generate_parser.add_argument("--generate-references", action="store_true", help="Generate a reference scan with correct answers for each model.")
    generate_parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper() if hasattr(args, 'log_level') else 'INFO', logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.command == "test":
        output_dir = args.output_dir
        exam_output_dir = os.path.join(output_dir, "exam_output")
        correction_output_dir = os.path.join(output_dir, "correction_results")

        # --- Generation Step ---
        logging.info("--- Running Generation Step ---")
        
        questions = _load_and_prepare_questions("sample_test.json")
        if questions is None:
            return

        generate_exams.generate_exams(
            questions=questions,
            output_dir=exam_output_dir,
            num_models=2,
            generate_fakes=4,
            columns=2,
            exam_title="CI Test Exam",
            exam_course="Test Course",
            exam_date="2025-01-01",
            id_length=8,
            lang="es",
            generate_references=True,
            font_size="10pt"
        )
        
        # --- Correction Step ---
        logging.info("--- Running Correction Step ---")
        simulated_scans_path = os.path.join(exam_output_dir, "simulated_scans")
        
        solutions_per_model = {}
        solutions_per_model_for_correction = {}
        max_score = 0
        try:
            solution_files = glob.glob(os.path.join(exam_output_dir, "exam_model_*_questions.json"))
            for sol_file in solution_files:
                model_id_match = re.search(r"exam_model_(\w+)_questions.json", os.path.basename(sol_file))
                if model_id_match:
                    model_id = model_id_match.group(1)
                    exam = PexamExam.model_validate_json(Path(sol_file).read_text(encoding="utf-8"))
                    solutions_per_model[model_id] = {q.id: q.model_dump() for q in exam.questions}
                    solutions_for_correction = {q.id: q.correct_answer_index for q in exam.questions if q.correct_answer_index is not None}
                    solutions_per_model_for_correction[model_id] = solutions_for_correction
                    if len(solutions_for_correction) > max_score:
                        max_score = len(solutions_for_correction)
        except Exception as e:
            logging.error(f"Failed to load solutions for the test run: {e}")
            return

        correction_success = correct_exams.correct_exams(
            input_path=simulated_scans_path,
            solutions_per_model=solutions_per_model_for_correction,
            output_dir=correction_output_dir
        )
        
        if correction_success:
            logging.info("--- Running Analysis Step ---")
            results_csv = os.path.join(correction_output_dir, "correction_results.csv")
            if os.path.exists(results_csv):
                analysis.analyze_results(
                    csv_filepath=results_csv,
                    max_score=max_score,
                    output_dir=correction_output_dir,
                    solutions_per_model=solutions_per_model,
                    void_questions_str="1",
                    void_questions_nicely_str="2"
                )
        logging.info("--- Test command finished successfully! ---")

    elif args.command == "correct":
        if not os.path.exists(args.input_path):
            logging.error(f"Input path not found: {args.input_path}")
            return
        if not os.path.isdir(args.exam_dir):
            logging.error(f"Exam directory not found: {args.exam_dir}")
            return
            
        # Load all solutions from exam_dir
        solutions_per_model = {}
        solutions_per_model_for_correction = {}
        max_score = 0
        try:
            solution_files = glob.glob(os.path.join(args.exam_dir, "exam_model_*_questions.json"))
            if not solution_files:
                logging.error(f"No 'exam_model_..._questions.json' files found in {args.exam_dir}")
                return

            for sol_file in solution_files:
                model_id_match = re.search(r"exam_model_(\w+)_questions.json", os.path.basename(sol_file))
                if model_id_match:
                    model_id = model_id_match.group(1)
                    exam = PexamExam.model_validate_json(Path(sol_file).read_text(encoding="utf-8"))
                    
                    # Store full question data for analysis
                    solutions_per_model[model_id] = {q.id: q.model_dump() for q in exam.questions}
                    
                    # Store only indices for the correction module
                    solutions_for_correction = {q.id: q.correct_answer_index for q in exam.questions if q.correct_answer_index is not None}
                    solutions_per_model_for_correction[model_id] = solutions_for_correction

                    if len(solutions_for_correction) > max_score:
                        max_score = len(solutions_for_correction)
                        
            logging.info(f"Loaded solutions for models: {list(solutions_per_model.keys())}")
        except Exception as e:
            logging.error(f"Failed to load or parse solutions from {args.exam_dir}: {e}", exc_info=True)
            return

        os.makedirs(args.output_dir, exist_ok=True)
        
        correction_success = correct_exams.correct_exams(
            input_path=args.input_path,
            solutions_per_model=solutions_per_model_for_correction,
            output_dir=args.output_dir
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
                    solutions_per_model=solutions_per_model,
                    void_questions_nicely_str=args.void_questions_nicely
                )
            else:
                logging.error(f"Analysis skipped: correction results file not found at {results_csv}")
    
    elif args.command == "generate":
        questions = _load_and_prepare_questions(args.questions_json)
        if questions is None:
            return
        
        keep_html = args.keep_html or (hasattr(args, 'log_level') and args.log_level == 'DEBUG')

        generate_exams.generate_exams(
            questions=questions,
            output_dir=args.output_dir,
            num_models=args.num_models,
            exam_title=args.exam_title,
            exam_course=args.exam_course,
            exam_date=args.exam_date,
            columns=args.columns,
            id_length=args.id_length,
            lang=args.lang,
            keep_html=keep_html,
            font_size=args.font_size,
            generate_fakes=args.generate_fakes,
            generate_references=args.generate_references
        )

if __name__ == "__main__":
    main()
