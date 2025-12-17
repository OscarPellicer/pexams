import logging
import os
import random
import re
from typing import List

from pexams.schemas import PexamQuestion

try:
    import pandas as pd
except ImportError:
    pd = None
    logging.debug("Pandas not found, Wooclap Excel export will be basic text.")


def _format_text_for_wooclap(text: str) -> str:
    """
    Applies specific formatting rules for Wooclap compatibility,
    intelligently handling code blocks.
    """
    if not text:
        return ""
        
    code_snippets = []
    
    # Temporarily replace code blocks with placeholders
    def protect_code(match):
        # Wooclap uses \texttt for code, so we'll store the raw content
        code_snippets.append(match.group(1))
        return f"__CODE__{len(code_snippets)-1}__"

    # Protect multiline and inline code blocks (though Wooclap is single line)
    text = re.sub(r'```(.*?)```', protect_code, text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', protect_code, text)

    # Convert $$ to $ for Wooclap math
    text = text.replace('$$', '$')
    
    # Remove markdown bold/italics from non-code text
    text = re.sub(r'(?<!\w)\*\*(.*?)\*\*(?!\w)', r'\1', text)
    text = re.sub(r'(?<!\w)\*(.*?)\*(?!\w)', r'\1', text)

    # Restore code blocks, wrapping them in \texttt{}
    for i, code in enumerate(code_snippets):
        # Escape characters that are special in LaTeX
        escaped_code = code.replace('\\', '\\textbackslash{}')
        escaped_code = escaped_code.replace('{', '\\{').replace('}', '\\}')
        escaped_code = escaped_code.replace('_', '\\_').replace('^', '\\^')
        escaped_code = escaped_code.replace('&', '\\&').replace('%', '\\%').replace('#', '\\#')
        text = text.replace(f"__CODE__{i}__", f"$\\texttt{{{escaped_code}}}$")
            
    return text


def convert_to_wooclap(questions: List[PexamQuestion], output_file: str):
    """
    Converts questions to CSV format suitable for Wooclap import.
    - Skips questions with images.
    - Applies LaTeX-based formatting for math and code.
    """
    # Ensure the output file has a .csv extension
    if not output_file.lower().endswith('.csv'):
        base, _ = os.path.splitext(output_file)
        output_file = base + '.csv'
        logging.warning(f"Output filename did not end with .csv, changing to: {output_file}")


    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    logging.info(f"Attempting to convert {len(questions)} questions to Wooclap CSV: {output_file}")

    if not pd:
         logging.warning("Pandas library not found. Cannot create CSV file for Wooclap. Creating placeholder text file.")
         placeholder_content = "Wooclap CSV export requires the 'pandas' library.\nInstall it with: pip install pandas\n\n"
         for q in questions:
              # Basic text representation for placeholder
              safe_text = q.text.replace('`', '"')
              correct_opts = [o.text for o in q.options if o.is_correct]
              incorrect_opts = [o.text for o in q.options if not o.is_correct]
              
              safe_correct = correct_opts[0].replace('`', '"') if correct_opts else "NO CORRECT ANSWER"
              safe_distractors = [d.replace('`', '"') for d in incorrect_opts]
              placeholder_content += f"Q{q.id}: {safe_text}\nCorrect: {safe_correct}\nIncorrect: {'; '.join(safe_distractors)}\n\n"
         # Create a .txt placeholder if pandas is missing
         txt_placeholder_file = os.path.splitext(output_file)[0] + ".txt"
         with open(txt_placeholder_file, 'w', encoding='utf-8') as f:
              f.write(placeholder_content)
         logging.info(f"Created placeholder text file: {txt_placeholder_file}")
         return

    # Proceed with Pandas export
    data = []
    max_choices = 0 # Keep track of the maximum number of choices for column definition
    questions_with_images_skipped = 0

    for q in questions:
        if q.image_source:
            logging.warning(f"Skipping question {q.id} because Wooclap does not support images.")
            questions_with_images_skipped += 1
            continue

        # Apply Wooclap specific formatting
        title = _format_text_for_wooclap(q.text)
        
        # Base row structure
        row = {'Type': 'MCQ', 'Title': title}
        
        # Helper to get options
        correct_opt = next((o.text for o in q.options if o.is_correct), None)
        distractors = [o.text for o in q.options if not o.is_correct]
        
        if not correct_opt:
             logging.warning(f"Question {q.id} has no correct answer, skipping.")
             continue
        
        options = [correct_opt] + distractors
        current_num_choices = len(options)
        if current_num_choices > max_choices:
            max_choices = current_num_choices

        random.shuffle(options) # Shuffle options for Wooclap display

        correct_index_str = None
        for i, option in enumerate(options):
             # Replace backticks in option text
             safe_option = _format_text_for_wooclap(option)
             col_name = f'Choice {i+1}'
             row[col_name] = safe_option
             if option == correct_opt:
                 correct_index_str = str(i+1) # Wooclap often uses 1-based index

        # Assert that exactly one correct answer index was found (for MCQ)
        assert correct_index_str is not None, f"Correct answer not found for question ID {q.id}"
        row['Correct'] = correct_index_str

        data.append(row)

    if questions_with_images_skipped > 0:
        logging.warning(f"Total questions skipped due to images: {questions_with_images_skipped}")

    if data:
        # Define the exact column order dynamically based on max_choices
        choice_cols = [f'Choice {i+1}' for i in range(max_choices)]
        column_order = ['Type', 'Title', 'Correct'] + choice_cols

        # Create DataFrame, Pandas handles missing columns (e.g., if one question has fewer choices)
        df = pd.DataFrame(data)
        # Reindex to ensure the exact column order and include all necessary choice columns
        df = df.reindex(columns=column_order)

        try:
            # Write to CSV file without pandas index, using specified order
            df.to_csv(output_file, index=False, encoding='utf-8', columns=column_order)
            logging.info(f"Successfully converted {len(questions)} questions to Wooclap CSV: {output_file}")
        except Exception as e:
             logging.error(f"Failed to write Wooclap CSV file {output_file}: {e}", exc_info=True)
    else:
        logging.info("No questions to convert to Wooclap format.")
