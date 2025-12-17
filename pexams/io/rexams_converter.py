import logging
import os
from typing import List, Optional

from pexams.schemas import PexamQuestion


def escape_latex(text: str) -> str:
    return text


def prepare_for_rexams(questions: List[PexamQuestion], output_dir: str, max_image_width: Optional[int] = None, max_image_height: Optional[int] = None):
    """Prepares question files (.Rmd) for R/exams using new schema."""
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Preparing {len(questions)} R/exams files in directory: {output_dir}")

    # --- Sanitize quotes before writing ---
    quotes_found_warning = False
    for q in questions:
        if '"' in q.text:
            q.text = q.text.replace('"', "'")
            quotes_found_warning = True
        
        for option in q.options:
            if '"' in option.text:
                option.text = option.text.replace('"', "'")
                quotes_found_warning = True

    if quotes_found_warning:
        logging.warning("Replaced double quotes with single quotes in questions for R/exams preparation.")
    # --- End sanitization ---

    for i, q in enumerate(questions):
        # Create a unique filename
        q_filename = f"question_{q.id}.Rmd"
        q_filepath = os.path.join(output_dir, q_filename)

        try:
            # --- Build the file content in memory ---
            rmd_content = []
            
            escaped_q_text = escape_latex(q.text)
            rmd_content.append("Question")
            rmd_content.append("========")
            rmd_content.append(f"{escaped_q_text}\n")

            if q.image_source:
                # Use absolute path for R/exams, ensuring forward slashes for R compatibility
                image_path = q.image_source.replace("\\", "/")
                
                r_opts = ""
                if max_image_width:
                    r_opts += f", out.width = '{max_image_width}px'"
                if max_image_height:
                    r_opts += f", out.height = '{max_image_height}px'"
                
                rmd_content.append(f"```{'{r}'}{r_opts} include_graphics('{image_path}')\n```\n")

            # Get options list (first one is correct in our schema if shuffled, but here we construct the list)
            # PexamQuestion options are just a list. We need to identify which is correct.
            # R/exams expects a list and a solution bitstring
            
            # Use the order in q.options
            options_list = q.options
            solution_bitstring = "".join(["1" if o.is_correct else "0" for o in options_list])

            rmd_content.append("Questionlist")
            rmd_content.append("------------")
            for option in options_list:
                escaped_option = escape_latex(option.text)
                rmd_content.append(f"* {escaped_option}")

            rmd_content.append("\nSolution") # Add newline before Solution
            rmd_content.append("========")
            if q.explanation:
                escaped_explanation = escape_latex(q.explanation)
                rmd_content.append(f"{escaped_explanation}\n")
            else:
                # Find correct answer text
                correct_text = next((o.text for o in options_list if o.is_correct), "Unknown")
                escaped_correct_answer = escape_latex(correct_text)
                rmd_content.append(f"The correct answer is: {escaped_correct_answer}\n")
            
            rmd_content.append("") # Add blank line before Meta-information

            rmd_content.append("Meta-information")
            rmd_content.append("================")
            rmd_content.append(f"exname: Question {q.id}")
            rmd_content.append(f"extype: mchoice")
            rmd_content.append(f"exsolution: {solution_bitstring}")
            rmd_content.append(f"exshuffle: TRUE")
            
            # Difficulty and other metadata not currently in PexamQuestion schema
            
            final_rmd_str = "\n".join(rmd_content)
            
            # --- Temporary Debugging ---
            logging.debug(f"--- RMD CONTENT FOR {q_filename} ---\n{final_rmd_str}\n------------------------------------")
            
            with open(q_filepath, 'w', encoding='utf-8') as f:
                f.write(final_rmd_str)

        except Exception as e:
            logging.error(f"Failed to create R/exams file {q_filepath}: {e}", exc_info=True)

    logging.info(f"Finished preparing R/exams files.")
