import argparse
import logging
import json
import os
from pathlib import Path
import glob
import re

from pexams import correct_exams
from pexams import generate_exams
from pexams import analysis
from pexams.schemas import PexamExam, PexamQuestion
from pydantic import ValidationError

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
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper() if hasattr(args, 'log_level') else 'INFO', logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.command == "correct":
        if not os.path.exists(args.input_path):
            logging.error(f"Input path not found: {args.input_path}")
            return
        if not os.path.isdir(args.exam_dir):
            logging.error(f"Exam directory not found: {args.exam_dir}")
            return
            
        # Load all solutions from exam_dir
        solutions_per_model = {}
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
                    solutions = {q.id: q.correct_answer_index for q in exam.questions if q.correct_answer_index is not None}
                    solutions_per_model[model_id] = solutions
                    if len(solutions) > max_score:
                        max_score = len(solutions)
            logging.info(f"Loaded solutions for models: {list(solutions_per_model.keys())}")
        except Exception as e:
            logging.error(f"Failed to load or parse solutions from {args.exam_dir}: {e}", exc_info=True)
            return

        os.makedirs(args.output_dir, exist_ok=True)
        
        correction_success = correct_exams.correct_exams(
            input_path=args.input_path,
            solutions_per_model=solutions_per_model,
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
                    void_questions_str=args.void_questions
                )
            else:
                logging.error(f"Analysis skipped: correction results file not found at {results_csv}")
    
    elif args.command == "generate":
        if not os.path.exists(args.questions_json):
            logging.error(f"Questions JSON file not found: {args.questions_json}")
            return
        
        try:
            exam = PexamExam.model_validate_json(Path(args.questions_json).read_text(encoding="utf-8"))
            questions = exam.questions
        except ValidationError as e:
            logging.error(f"Failed to validate questions JSON file: {e}")
            return
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse questions JSON file: {e}")
            return

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
            keep_html=args.keep_html,
            font_size=args.font_size,
            generate_fakes=args.generate_fakes,
            generate_references=args.generate_references
        )

if __name__ == "__main__":
    main()
