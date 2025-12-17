import base64
import logging
import mimetypes
import os
import random
import re
from typing import List, Optional

from pexams.schemas import PexamQuestion


def escape_gift(text: str) -> str:
    """Escapes special characters for GIFT format."""
    # Process in a single pass to avoid issues with chained replacements
    escaped_text = ""
    for char in text:
        if char in ['~', '=', '#', '{', '}', ':', '\\']:
            escaped_text += '\\' + char
        else:
            escaped_text += char
    return escaped_text


def convert_to_gift(questions: List[PexamQuestion], output_file: str, max_image_width: Optional[int] = None, max_image_height: Optional[int] = None):
    """Converts questions to GIFT format using PexamQuestion."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    logging.info(f"Converting {len(questions)} questions to GIFT: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, q in enumerate(questions):
            
            # Use question ID in title for uniqueness
            question_name = f"Q{i+1}_{q.id}"
            f.write(f"::{escape_gift(question_name)}::")

            # --- Prepare Question Text ---
            question_text = q.text
            format_specifier = "[markdown]"  # Always use Markdown as requested

            if q.image_source:
                # Correctly resolve the image path relative to the project root
                # Assuming image_source is absolute or relative to CWD
                image_path = q.image_source
                if not os.path.isabs(image_path):
                     image_path = os.path.abspath(image_path)
                     
                try:
                    with open(image_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        mime_type, _ = mimetypes.guess_type(image_path)
                        if mime_type:
                            style = ""
                            if max_image_width:
                                style += f"width:{max_image_width}px; "
                            if max_image_height:
                                style += f"max-height:{max_image_height}px;"
                            
                            img_tag = f"<img src='data:{mime_type};base64,{encoded_string}' alt='Image for question' style='{style}' />"
                            question_text += f"<p>{img_tag}</p>"
                            # No need for special logic, as Markdown handles HTML
                        else:
                            logging.warning(f"Could not determine MIME type for {image_path}. Skipping image.")
                except FileNotFoundError:
                    logging.warning(f"Image file not found: {image_path}. Skipping image embedding.")
                except Exception as e:
                    logging.error(f"Error processing image {image_path}: {e}", exc_info=True)

            # Convert single '$' to '$$' for MathJax, but not if already '$$'
            def replace_latex_delimiters(text):
                # This regex finds single '$' not preceded or followed by another '$'
                return re.sub(r'(?<!\$)\$(?!\$)', '$$', text)

            question_text = replace_latex_delimiters(question_text)
            
            # Prepend format tag if needed
            question_text_with_format = f"{format_specifier}{question_text}"

            f.write(f"{escape_gift(question_text_with_format)} {{\n")

            # --- Prepare and Write Options ---
            options_list = list(q.options)
            random.shuffle(options_list)  # Shuffle order in GIFT file

            for option in options_list:
                processed_option = replace_latex_delimiters(option.text)
                option_escaped = escape_gift(processed_option)
                prefix = "=" if option.is_correct else "~"
                # For now, no per-answer feedback is in the schema
                f.write(f"\t{prefix}{option_escaped}\n")

            # --- Add General Feedback (Explanation) ---
            if q.explanation:
                processed_explanation = replace_latex_delimiters(q.explanation)
                explanation_escaped = escape_gift(processed_explanation)
                f.write(f"\t####{explanation_escaped}\n")  # General feedback marker

            f.write("}\n\n")

    print(f"Successfully converted questions to GIFT: {output_file}")
