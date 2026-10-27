import logging
import os
import random
import re
from typing import List, Optional

from pexams.markdown_render import (image_data_uri, inline_image_styles, limit_image_size, question_image_limits,
                                     render_markdown, replace_image_sources)
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


def _gift_html(text: str, base_dir: Optional[str], nl2br: bool = False, paragraph: bool = True) -> str:
    """Markdown rendered like the PDF, as one-line HTML: images embedded as data URIs and their
    layout as inline styles. GIFT ends a question at a blank line, so newlines become spaces
    (``&#10;`` inside ``<pre>``, where they matter)."""
    html_text = inline_image_styles(render_markdown(text or "", nl2br=nl2br))
    html_text = replace_image_sources(html_text, base_dir, image_data_uri)
    if not paragraph:
        html_text = re.sub(r"^<p>(.*)</p>$", r"\1", html_text, flags=re.DOTALL)
    parts = re.split(r"(<pre\b.*?</pre>)", html_text, flags=re.DOTALL)
    return "".join(p.replace("\n", "&#10;") if p.startswith("<pre") else p.replace("\n", " ") for p in parts)


def convert_to_gift(questions: List[PexamQuestion], output_file: str, max_image_width: Optional[int] = None,
                    max_image_height: Optional[int] = None, base_dir: Optional[str] = None):
    """Converts questions to GIFT (HTML format). Images in the text (paths relative to
    ``base_dir``, the folder of the questions file) are embedded."""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    logging.info(f"Converting {len(questions)} questions to GIFT: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, q in enumerate(questions):
            if q.is_open_answer:
                logging.warning("Skipping open-answer question %s: GIFT export currently supports multiple choice only.", q.id)
                continue

            # Use question ID in title for uniqueness
            question_name = f"Q{i+1}_{q.id}"
            f.write(f"::{escape_gift(question_name)}::")

            question_text = _gift_html(q.text, base_dir, nl2br=True)
            question_text = limit_image_size(question_text, *question_image_limits(q, max_image_width, max_image_height))
            f.write(f"[html]{escape_gift(question_text)} {{\n")

            options_list = list(q.options)
            random.shuffle(options_list)  # Shuffle order in GIFT file
            for option in options_list:
                prefix = "=" if option.is_correct else "~"
                f.write(f"\t{prefix}{escape_gift(_gift_html(option.text, base_dir, paragraph=False))}\n")

            if q.explanation:  # general feedback
                f.write(f"\t####{escape_gift(_gift_html(q.explanation, base_dir))}\n")

            f.write("}\n\n")

    logging.info(f"Successfully converted questions to GIFT: {output_file}")
