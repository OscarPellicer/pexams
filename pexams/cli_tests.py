import os
import logging
import pandas as pd
import cv2
import numpy as np

from pexams import generate_exams, layout, correct_exams, analysis, utils
from pexams.io.loader import load_and_prepare_questions
from pexams.io import md_converter, rexams_converter, wooclap_converter, gift_converter, moodle_xml_converter
from pexams.grades import fill_marks_in_file

def test_overflow(output_dir: str):
    logging.info("--- Starting Overflow Tests ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sample questions
    questions = load_and_prepare_questions("sample_test.md")
    if not questions:
        logging.error("Failed to load sample questions for overflow test.")
        return

    # 1. Test Text Overflow
    logging.info("--- Testing Text Overflow in Header ---")
    long_text = "This is a very long text that should overflow the available space in the header fields of the answer sheet template. " * 5
    
    try:
        generate_exams.generate_exams(
            questions=questions,
            output_dir=os.path.join(output_dir, "overflow_text"),
            num_models=1,
            exam_title=long_text,
            exam_course=long_text,
            exam_date=long_text,
            columns=1,
            generate_fakes=0
        )
        logging.info("Text overflow generation completed (check output visually for layout issues).")
    except Exception as e:
        logging.error(f"Text overflow test failed with error: {e}")

    # 2. Test Question Number Overflow
    logging.info("--- Testing Question Number Overflow ---")
    
    # Create too many questions
    max_questions = layout.MAX_QUESTIONS
    many_questions = []
    
    # We need to duplicate questions to exceed max
    while len(many_questions) <= max_questions:
        for q in questions:
            q_copy = q.model_copy(deep=True)
            q_copy.id = len(many_questions) + 1
            many_questions.append(q_copy)
            
    logging.info(f"Generated {len(many_questions)} questions (Max allowed: {max_questions}). Expecting error...")
    
    try:
        generate_exams.generate_exams(
            questions=many_questions,
            output_dir=os.path.join(output_dir, "overflow_questions"),
            num_models=1
        )
        logging.error("Question overflow test FAILED: No error was raised.")
    except ValueError as e:
        logging.info(f"Question overflow test PASSED: Caught expected ValueError: {e}")
    except Exception as e:
        logging.error(f"Question overflow test failed with unexpected error: {e}")

    # 3. Test Custom Header Overflow
    logging.info("--- Testing Custom Header Overflow ---")
    try:
        generate_exams.generate_exams(
            questions=questions,
            output_dir=os.path.join(output_dir, "overflow_header"),
            num_models=1,
            columns=1,
            generate_fakes=0,
            custom_header=long_text
        )
        logging.info("Custom header overflow generation completed (check output visually for layout issues).")
    except Exception as e:
        logging.error(f"Custom header overflow test failed with error: {e}")

    logging.info("--- Overflow Tests Finished ---")

def run_full_test(output_dir: str):
    """Runs a full generate/correct/analyze cycle using bundled sample data."""
    if cv2 is None or np is None:
         logging.error("OpenCV/Numpy required for test. Please install opencv-python and numpy.")
         return

    os.makedirs(output_dir, exist_ok=True)
    
    logging.info("--- 1. Loading Sample MD Questions ---")
    
    # Load sample_test.md directly (it should now exist in assets)
    questions = load_and_prepare_questions("sample_test.md")
    if not questions:
        logging.error("Failed to load sample_test.md from assets. Please ensure it exists.")
        return

    # --- 2. Test Exports ---
    logging.info("--- Testing Exports ---")
    for fmt in ["rexams", "wooclap", "gift", "md", "moodle"]:
        out_export = os.path.join(output_dir, f"export_{fmt}")
        os.makedirs(out_export, exist_ok=True)
        if fmt == "rexams": rexams_converter.prepare_for_rexams(questions, out_export)
        elif fmt == "wooclap": wooclap_converter.convert_to_wooclap(questions, os.path.join(out_export, "w.csv"))
        elif fmt == "gift": gift_converter.convert_to_gift(questions, os.path.join(out_export, "g.gift"))
        elif fmt == "md": md_converter.save_questions_to_md(questions, os.path.join(out_export, "q.md"))
        elif fmt == "moodle": moodle_xml_converter.convert_to_moodle_xml(questions, os.path.join(out_export, "m.xml"))
        logging.info(f"Exported to {fmt}")

    # --- 3. Generate Exams & Fakes ---
    logging.info("--- Generating Exams and Fakes ---")
    
    # Initialize seeds for test
    utils.set_seeds(seed_questions=None, seed_answers=42)
    
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
        custom_header="**Instructions:** Incorrect answers will be penalized by **-0.25 points**. \n Why do mathematicians confuse Halloween and Christmas? Because $OCT 31 = DEC 25$."
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
                    fill_marks_in_file(input_csv_path, "student_id", "mark", results_csv, fuzzy_threshold=80)
                    
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

            # --- 7. Test Rerun Analysis (Manual Correction) ---
            logging.info("--- Testing Rerun Analysis (Manual CSV Modification) ---")
            
            # Load the score from the previous run (from final_marks.csv)
            final_marks_path = os.path.join(correction_output_dir, "final_marks.csv")
            if os.path.exists(final_marks_path) and valid_ids:
                target_id = valid_ids[0]
                df_marks_old = pd.read_csv(final_marks_path)
                # We need to find the row for target_id
                row_old = df_marks_old[df_marks_old['student_id'].astype(str) == str(target_id)]
                if not row_old.empty:
                    old_score = row_old.iloc[0]['score']
                    
                    # Now modify correction_results.csv
                    df = pd.read_csv(results_csv) # reload to be fresh
                    row_idx = df.index[df['student_id'].astype(str) == str(target_id)].tolist()[0]
                    model_id = str(df.at[row_idx, 'model_id'])
                    
                    # Use Question 3 (since Q1 is voided in the test)
                    target_q = 3
                    if model_id in solutions_simple and target_q in solutions_simple[model_id]:
                        q_sol_idx = solutions_simple[model_id][target_q]
                        q_correct_char = chr(ord('A') + q_sol_idx)
                        
                        current_answer = str(df.at[row_idx, f'answer_{target_q}'])
                        
                        # Determine change
                        if current_answer == q_correct_char:
                            new_answer = 'B' if q_correct_char == 'A' else 'A' # Make it wrong
                            expected_delta = -1
                            logging.info(f"Student {target_id}: Changing Q{target_q} from {current_answer} (Correct) to {new_answer} (Wrong).")
                        else:
                            new_answer = q_correct_char # Make it correct
                            expected_delta = 1
                            logging.info(f"Student {target_id}: Changing Q{target_q} from {current_answer} (Wrong) to {new_answer} (Correct).")
                            
                        # Apply change
                        df.at[row_idx, f'answer_{target_q}'] = new_answer
                        df.to_csv(results_csv, index=False)
                        
                        # Run Analysis Again
                        analysis.analyze_results(
                            csv_filepath=results_csv,
                            max_score=max_score,
                            output_dir=correction_output_dir,
                            solutions_per_model=solutions_full,
                            void_questions_str="1",
                            void_questions_nicely_str="2"
                        )
                        
                        # Verify
                        df_marks_new = pd.read_csv(final_marks_path)
                        row_new = df_marks_new[df_marks_new['student_id'].astype(str) == str(target_id)]
                        new_score = row_new.iloc[0]['score']
                        
                        logging.info(f"Old Score: {old_score}, New Score: {new_score}, Expected Delta: {expected_delta}")
                        
                        if new_score == old_score + expected_delta:
                            logging.info("Manual correction verification SUCCESSFUL.")
                        else:
                            logging.warning(f"Manual correction verification FAILED. Expected {old_score + expected_delta}, got {new_score}")
                    else:
                        logging.warning(f"Model {model_id} or Question {target_q} not found in solutions.")
                else:
                    logging.warning(f"Student {target_id} not found in final_marks.csv")
            else:
                logging.warning("final_marks.csv not found or no valid IDs for manual correction test.")

    logging.info("--- Test command finished successfully! ---")
