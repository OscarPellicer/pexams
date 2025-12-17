import argparse
import logging
import json
import os
from pathlib import Path
import glob
import re
from typing import Optional, List, Dict
import sys
import shutil

# Attempt to import pandas for data handling
try:
    import pandas as pd
except ImportError:
    pd = None

from pexams import correct_exams
from pexams import generate_exams
from pexams import analysis
from pexams import utils
from pexams.io import md_converter, rexams_converter, wooclap_converter, gift_converter
from pexams.schemas import PexamExam, PexamQuestion
from pydantic import ValidationError
import pexams

def _load_and_prepare_questions(questions_path_str: str) -> Optional[List[PexamQuestion]]:
    """
    Loads questions from a file (JSON or MD), resolving bundled assets and image paths.
    """
    questions_path = Path(questions_path_str)

    # Check if the file exists at the given path. If not, try to find it in the package assets.
    if not questions_path.exists():
        try:
            package_dir = Path(pexams.__file__).parent
            asset_path = package_dir / "assets" / questions_path_str
            if asset_path.exists():
                questions_path = asset_path
            else:
                raise FileNotFoundError
        except (FileNotFoundError, AttributeError):
            logging.error(f"Questions file not found at '{questions_path_str}' or as a built-in asset.")
            return None

    questions = None
    
    # Determine format by extension
    ext = questions_path.suffix.lower()
    if ext == '.md':
        logging.info(f"Loading questions from Markdown file: {questions_path}")
        questions = md_converter.load_questions_from_md(str(questions_path))
    elif ext == '.json':
        logging.info(f"Loading questions from JSON file: {questions_path}")
        # logging.warning("JSON input format is deprecated. Please use Markdown (.md) format.")
        try:
            exam = PexamExam.model_validate_json(questions_path.read_text(encoding="utf-8"))
            questions = exam.questions
        except ValidationError as e:
            logging.error(f"Failed to validate questions JSON file: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse questions JSON file: {e}")
            return None
    else:
        # Try Markdown first as default
        logging.info(f"No known extension '{ext}'. Trying to load as Markdown...")
        questions = md_converter.load_questions_from_md(str(questions_path))
        if not questions:
             pass

    if not questions:
        logging.error("No questions loaded.")
        return None
        
    # Resolve paths for images, making them absolute before passing them to the generator.
    file_dir = questions_path.parent
    for q in questions:
        if q.image_source and not Path(q.image_source).is_absolute():
            # First, try to resolve the path relative to the input file's directory.
            image_path_rel_file = (file_dir / q.image_source).resolve()
            
            # If that path doesn't exist, try resolving relative to the current working directory.
            image_path_rel_cwd = Path(q.image_source).resolve()
            
            # Also check relative to package assets if loading from sample
            try:
                package_dir = Path(pexams.__file__).parent
                image_path_rel_assets = (package_dir / "assets" / q.image_source).resolve()
            except:
                image_path_rel_assets = Path("nonexistent")

            if image_path_rel_file.exists():
                q.image_source = str(image_path_rel_file)
            elif image_path_rel_cwd.exists():
                q.image_source = str(image_path_rel_cwd)
            elif image_path_rel_assets.exists():
                 q.image_source = str(image_path_rel_assets)
            else:
                logging.warning(
                    f"Could not find image for question {q.id} at '{q.image_source}'. "
                    f"Checked relative to input file, current directory, and assets."
                )
    return questions

def _fill_marks_in_file(input_file: str, id_col: str, mark_col: str, correction_results_csv: str, fuzzy_threshold: int = 100):
    """Fills marks into the input file (CSV/XLSX) based on correction results."""
    if not pd:
        logging.error("Pandas is required to fill marks in input files. Please install pandas.")
        return

    if not os.path.exists(correction_results_csv):
        logging.error(f"Correction results file not found: {correction_results_csv}")
        return

    try:
        # Load input file
        file_ext = os.path.splitext(input_file)[1].lower()
        if file_ext == '.csv':
            df_input = pd.read_csv(input_file)
        elif file_ext in ['.xlsx', '.xls']:
            df_input = pd.read_excel(input_file)
        elif file_ext == '.tsv':
            df_input = pd.read_csv(input_file, sep='\t')
        else:
            logging.error(f"Unsupported input file format: {file_ext}")
            return
            
        # Load correction results (final_marks.csv or correction_results.csv)
        # We prefer final_marks.csv which has the scaled mark
        final_marks_path = os.path.join(os.path.dirname(correction_results_csv), "final_marks.csv")
        if os.path.exists(final_marks_path):
             df_marks = pd.read_csv(final_marks_path)
             mark_source_col = 'mark'
        else:
             logging.warning("final_marks.csv not found, using raw scores from correction_results.csv")
             df_marks = pd.read_csv(correction_results_csv)
             mark_source_col = 'score' # Or whatever column holds the value we want
        
        # Ensure ID columns are string
        if id_col not in df_input.columns:
            logging.error(f"ID column '{id_col}' not found in input file.")
            return
            
        df_input[id_col] = df_input[id_col].astype(str).str.strip()
        df_marks['student_id'] = df_marks['student_id'].astype(str).str.strip()
        
        # Create a mapping from OCR ID to Mark
        ocr_id_to_mark = dict(zip(df_marks['student_id'], df_marks[mark_source_col]))
        ocr_ids = list(ocr_id_to_mark.keys())
        
        # Prepare the mark column in input df
        if mark_col not in df_input.columns:
            df_input[mark_col] = None
            
        matched_count = 0
        
        for idx, row in df_input.iterrows():
            target_id = row[id_col]
            if pd.isna(target_id) or not target_id:
                continue
                
            # Exact match
            if target_id in ocr_id_to_mark:
                df_input.at[idx, mark_col] = ocr_id_to_mark[target_id]
                matched_count += 1
            elif fuzzy_threshold < 100:
                # Fuzzy match against OCR IDs
                best_match_ocr_id = utils.fuzzy_match_id(target_id, ocr_ids, threshold=fuzzy_threshold)
                if best_match_ocr_id:
                    df_input.at[idx, mark_col] = ocr_id_to_mark[best_match_ocr_id]
                    logging.info(f"Fuzzy matched '{target_id}' with OCR ID '{best_match_ocr_id}' (Mark: {ocr_id_to_mark[best_match_ocr_id]})")
                    matched_count += 1
                    
        # Save back to file
        if file_ext == '.csv':
            df_input.to_csv(input_file, index=False)
        elif file_ext in ['.xlsx', '.xls']:
            df_input.to_excel(input_file, index=False)
        elif file_ext == '.tsv':
            df_input.to_csv(input_file, sep='\t', index=False)
            
        logging.info(f"Updated {input_file} with marks. Matched {matched_count}/{len(df_input)} students.")

    except Exception as e:
        logging.error(f"Failed to fill marks in input file: {e}", exc_info=True)


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
    correct_parser.add_argument(
        "--input-csv",
        type=str,
        help="Path to an input CSV/TSV/XLSX file to fill with marks."
    )
    correct_parser.add_argument(
        "--id-column",
        type=str,
        help="Column name in input-csv containing student IDs."
    )
    correct_parser.add_argument(
        "--mark-column",
        type=str,
        help="Column name in input-csv to fill with marks."
    )
    correct_parser.add_argument(
        "--fuzzy-id-match",
        type=int,
        default=100,
        help="Fuzzy matching threshold (0-100) for student IDs."
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

    # --- Generation/Convert Command ---
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate exams or export questions to other formats."
    )
    generate_parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input file containing questions (Markdown .md or JSON)."
    )
    generate_parser.add_argument(
        "--to",
        type=str,
        default="pexams",
        choices=["pexams", "rexams", "wooclap", "gift", "md"],
        help="Output format. Default is 'pexams' (PDF generation)."
    )
    
    generate_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the output."
    )
    generate_parser.add_argument("--exam-title", type=str, default="Final Exam", help="Title of the exam.")
    generate_parser.add_argument("--exam-course", type=str, default=None, help="Course name for the exam.")
    generate_parser.add_argument("--exam-date", type=str, default=None, help="Date of the exam.")
    generate_parser.add_argument("--lang", type=str, default="en", help="Language for the answer sheet / output.")
    generate_parser.add_argument("--num-models", type=int, default=4, help="Number of different exam models to generate (pexams only).")
    generate_parser.add_argument("--columns", type=int, default=1, choices=[1, 2, 3], help="Number of columns (pexams only).")
    generate_parser.add_argument("--font-size", type=str, default="11pt", help="Base font size (pexams only).")
    generate_parser.add_argument("--total-students", type=int, default=0, help="Total number of students for mass PDF generation (pexams only).")
    generate_parser.add_argument("--extra-model-templates", type=int, default=0, help="Number of extra template sheets to generate per model (pexams only).")
    generate_parser.add_argument("--keep-html", action="store_true", help="Keep intermediate HTML files (pexams only).")
    generate_parser.add_argument("--generate-fakes", type=int, default=0, help="Generate simulated scans (pexams only).")
    generate_parser.add_argument("--generate-references", action="store_true", help="Generate reference scan (pexams only).")
    generate_parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper() if hasattr(args, 'log_level') else 'INFO', logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.command == "test":
        try:
            import cv2
            import numpy as np
        except ImportError:
             logging.error("OpenCV/Numpy required for test.")
             return

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info("--- 1. Loading Sample MD Questions ---")
        
        # Load sample_test.md directly (it should now exist in assets)
        # We try to load "sample_test.md" which _load_and_prepare_questions looks for in assets
        questions = _load_and_prepare_questions("sample_test.md")
        if not questions:
            logging.error("Failed to load sample_test.md from assets. Please ensure it exists.")
            return

        # --- 2. Test Exports ---
        logging.info("--- Testing Exports ---")
        for fmt in ["rexams", "wooclap", "gift", "md"]:
            out_export = os.path.join(output_dir, f"export_{fmt}")
            os.makedirs(out_export, exist_ok=True)
            if fmt == "rexams": rexams_converter.prepare_for_rexams(questions, out_export)
            elif fmt == "wooclap": wooclap_converter.convert_to_wooclap(questions, os.path.join(out_export, "w.csv"))
            elif fmt == "gift": gift_converter.convert_to_gift(questions, os.path.join(out_export, "g.gift"))
            elif fmt == "md": md_converter.save_questions_to_md(questions, os.path.join(out_export, "q.md"))
            logging.info(f"Exported to {fmt}")

        # --- 3. Generate Exams & Fakes ---
        logging.info("--- Generating Exams and Fakes ---")
        exam_output_dir = os.path.join(output_dir, "exam_output")
        generate_exams.generate_exams(
            questions=questions,
            output_dir=exam_output_dir,
            num_models=2,
            generate_fakes=4,
            columns=2,
            exam_title="CI Test Exam",
            exam_course="Test Course",
            exam_date="2025-01-01",
            lang="es",
            generate_references=True,
            font_size="10pt",
            total_students=11, # Generate a pdf with 11 exams of alternate models
            extra_model_templates=1, # Generate 1 extra template sheet per model
        )
        
        # --- 4. Correct ---
        logging.info("--- Running Correction ---")
        correction_output_dir = os.path.join(output_dir, "correction_results")
        simulated_scans_path = os.path.join(exam_output_dir, "simulated_scans")
        
        solutions_full, solutions_simple, max_score = utils.load_solutions(exam_output_dir)
        if not solutions_simple:
            logging.error("Failed to load solutions for test.")
            return

        correction_success = correct_exams.correct_exams(
            input_path=simulated_scans_path,
            solutions_per_model=solutions_simple,
            output_dir=correction_output_dir,
            questions_dir=exam_output_dir
        )
        
        if correction_success:
            # --- 5. Analysis ---
            logging.info("--- Running Analysis ---")
            results_csv = os.path.join(correction_output_dir, "correction_results.csv")
            if os.path.exists(results_csv):
                analysis.analyze_results(
                    csv_filepath=results_csv,
                    max_score=max_score,
                    output_dir=correction_output_dir,
                    solutions_per_model=solutions_full,
                    void_questions_str="1",
                    void_questions_nicely_str="2"
                )
                
                # --- 6. Test Fuzzy Match / Mark Filling ---
                logging.info("--- Testing Fuzzy Match & Mark Filling ---")
                df = pd.read_csv(results_csv)
                detected_ids = df['student_id'].tolist()
                
                # Filter out unknown/unreadable if any
                valid_ids = [str(x) for x in detected_ids if 'unknown' not in str(x).lower()]
                
                if valid_ids:
                    target_id = valid_ids[0]
                    # Create a fuzzy version (change last char)
                    if len(target_id) > 0:
                        original_char = target_id[-1]
                        new_char = 'A' if original_char != 'A' else 'B'
                        fuzzy_id = target_id[:-1] + new_char
                        
                        input_csv_path = os.path.join(output_dir, "students_input.csv")
                        with open(input_csv_path, "w", encoding="utf-8") as f:
                            f.write(f"student_id,name,mark\n")
                            f.write(f"{fuzzy_id},Test Student,0\n") # 0 mark initially
                            
                        logging.info(f"Created input CSV with ID '{fuzzy_id}' (Target OCR ID: '{target_id}')")
                        
                        # Run fill marks with high fuzzy tolerance
                        _fill_marks_in_file(input_csv_path, "student_id", "mark", results_csv, fuzzy_threshold=80)
                        
                        # Verify
                        df_in = pd.read_csv(input_csv_path)
                        mark = df_in.iloc[0]['mark']
                        logging.info(f"Mark after filling: {mark}")
                        if mark > 0:
                             logging.info("Fuzzy match verification SUCCESSFUL (Mark > 0).")
                        else:
                             logging.warning("Fuzzy match verification inconclusive (Mark is 0 or failed).")
                else:
                    logging.warning("No valid student IDs found to test fuzzy matching.")

        logging.info("--- Test command finished successfully! ---")

    elif args.command == "correct":
        if not os.path.exists(args.input_path):
            logging.error(f"Input path not found: {args.input_path}")
            return
        if not os.path.isdir(args.exam_dir):
            logging.error(f"Exam directory not found: {args.exam_dir}")
            return
            
        solutions_full, solutions_simple, max_score = utils.load_solutions(args.exam_dir)
        if not solutions_simple:
            return

        os.makedirs(args.output_dir, exist_ok=True)
        
        correction_success = correct_exams.correct_exams(
            input_path=args.input_path,
            solutions_per_model=solutions_simple,
            output_dir=args.output_dir,
            questions_dir=args.exam_dir
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
                    void_questions_nicely_str=args.void_questions_nicely
                )
                
                # Input CSV Filling
                if args.input_csv:
                    if args.id_column and args.mark_column:
                         _fill_marks_in_file(args.input_csv, args.id_column, args.mark_column, results_csv, args.fuzzy_id_match)
                    else:
                        logging.warning("--input-csv provided but --id-column or --mark-column missing. Skipping mark filling.")
            else:
                logging.error(f"Analysis skipped: correction results file not found at {results_csv}")
    
    elif args.command == "generate":
        questions = _load_and_prepare_questions(args.input_file)
        if questions is None:
            return
            
        output_fmt = args.to
        out_dir = args.output_dir
        
        # Helper to warn about ignored arguments
        def check_arg(name, used_formats):
            if output_fmt not in used_formats and getattr(args, name) != parser.get_default(name):
                logging.warning(f"Argument '--{name}' is ignored for format '{output_fmt}'.")

        # Arguments specific to pexams
        pexams_args = ["num_models", "columns", "font_size", "total_students", "keep_html", "generate_fakes", "generate_references", "extra_model_templates"]
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
                extra_model_templates=args.extra_model_templates
            )
        elif output_fmt == "rexams":
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
        else:
            logging.error(f"Unknown output format: {output_fmt}")

if __name__ == "__main__":
    main()
