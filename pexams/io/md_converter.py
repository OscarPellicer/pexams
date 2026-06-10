import re
import os
import logging
from typing import List, Optional, Tuple

from pexams.schemas import PexamAnswerArea, PexamQuestion, PexamOption


def _parse_question_header(header_line: str) -> Tuple[str, dict]:
    """Parse headers such as '## q1 {type=open points=4 lines=10}'."""
    match = re.match(r'^##\s+(.+)', header_line)
    if not match:
        return "", {}

    raw_header = match.group(1).strip()
    attrs = {}
    attr_match = re.search(r'\{([^}]*)\}\s*$', raw_header)
    if attr_match:
        raw_header = raw_header[:attr_match.start()].strip()
        for token in attr_match.group(1).split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            attrs[key.strip().lower()] = value.strip().strip('"\'')

    return raw_header, attrs


def _normalize_question_type(raw_type: Optional[str]) -> str:
    if not raw_type:
        return "multiple_choice"
    normalized = raw_type.strip().lower().replace("-", "_")
    if normalized in {"open", "open_answer", "short_answer", "free_text", "essay"}:
        return "open_answer"
    return "multiple_choice"


def _parse_float_attr(attrs: dict, key: str, default: Optional[float] = None) -> Optional[float]:
    if key not in attrs:
        return default
    try:
        return float(attrs[key])
    except (TypeError, ValueError):
        logging.warning("Invalid numeric value for '%s': %s", key, attrs[key])
        return default


def _parse_int_attr(attrs: dict, key: str, default: int) -> int:
    if key not in attrs:
        return default
    try:
        return int(attrs[key])
    except (TypeError, ValueError):
        logging.warning("Invalid integer value for '%s': %s", key, attrs[key])
        return default


def _parse_bool_attr(attrs: dict, key: str, default: bool = False) -> bool:
    if key not in attrs:
        return default
    value = str(attrs[key]).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    logging.warning("Invalid boolean value for '%s': %s", key, attrs[key])
    return default


def _split_open_answer_sections(lines: List[str]) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    sections = {
        "question": [],
        "expected_answer": [],
        "rubric": [],
        "answer_area": [],
    }
    current = "question"
    section_markers = {
        "**expected answer:**": "expected_answer",
        "**expected_answer:**": "expected_answer",
        "**rubric:**": "rubric",
        "**answer area:**": "answer_area",
        "**answer_area:**": "answer_area",
    }

    for line in lines:
        marker = line.strip().lower()
        if marker in section_markers:
            current = section_markers[marker]
            continue
        sections[current].append(line)

    question_text = "\n".join(sections["question"]).strip()
    expected_answer = "\n".join(sections["expected_answer"]).strip() or None
    rubric = "\n".join(sections["rubric"]).strip() or None
    answer_area = "\n".join(sections["answer_area"]).strip() or None
    return question_text, expected_answer, rubric, answer_area

def load_questions_from_md(path: str) -> List[PexamQuestion]:
    """
    Parses a Markdown file containing questions into a list of PexamQuestion objects.
    
    Format specification:
    ## question_id
    > ![Image for question](image.png)
    Question text...
    * Correct answer
    * Wrong answer 1
    * Wrong answer 2
    
    **Explanation:**
    Explanation text...
    """
    if not os.path.exists(path):
        logging.error(f"Questions markdown file not found: {path}")
        return []
        
    questions: List[PexamQuestion] = []
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split the content by the new question header '## ' that appears at the start of a line
    question_blocks = re.split(r'(?=^## )', content, flags=re.MULTILINE)

    md_dir = os.path.dirname(os.path.abspath(path))

    for block in question_blocks:
        block = block.strip()
        if not block.startswith("##"):
            continue

        lines = block.split('\n')
        
        header_line = lines[0]
        question_id_str, attrs = _parse_question_header(header_line)
        if not question_id_str:
            continue

        content_lines = lines[1:]
        
        # Check for and extract a quoted image line
        image_path = None
        if content_lines and content_lines[0].strip().startswith('>'):
            img_match = re.search(r'!\[.*\]\((.*)\)', content_lines[0])
            if img_match:
                raw_image_path = img_match.group(1).strip()
                # Resolve image path relative to MD file
                # If it's just a filename, it's in the same dir.
                # If it's a relative path, it's relative to MD dir.
                
                # Check if it exists
                abs_image_path = os.path.normpath(os.path.join(md_dir, raw_image_path))
                if os.path.exists(abs_image_path):
                    image_path = abs_image_path
                else:
                    logging.warning(f"Image not found at {abs_image_path} for question {question_id_str}")
                    # Keep the raw path if not found, maybe it works later or is a URL?
                    image_path = raw_image_path

                # Remove the image line from the content
                content_lines = content_lines[1:]

        question_type = _normalize_question_type(attrs.get("type"))
        points = _parse_float_attr(attrs, "points", 1.0) or 1.0
        answer_lines = _parse_int_attr(attrs, "lines", 8)
        height_mm = _parse_float_attr(attrs, "height_mm")
        show_lines = _parse_bool_attr(attrs, "show_lines", False)
        font_size = attrs.get("font_size")

        if question_type == "open_answer":
            question_text, expected_answer, rubric, answer_area_text = _split_open_answer_sections(content_lines)
            if not question_text:
                logging.warning(f"Open-answer question ID '{question_id_str}' has no question text. Skipping.")
                continue

            if answer_area_text:
                line_match = re.search(r'lines\s*[:=]\s*(\d+)', answer_area_text, flags=re.IGNORECASE)
                if line_match:
                    answer_lines = int(line_match.group(1))
                height_match = re.search(r'height_mm\s*[:=]\s*([0-9.]+)', answer_area_text, flags=re.IGNORECASE)
                if height_match:
                    height_mm = float(height_match.group(1))
                show_lines_match = re.search(r'show_lines\s*[:=]\s*(true|false|yes|no|1|0|on|off)', answer_area_text, flags=re.IGNORECASE)
                if show_lines_match:
                    show_lines = _parse_bool_attr({"show_lines": show_lines_match.group(1)}, "show_lines", show_lines)

            questions.append(PexamQuestion(
                id=question_id_str,
                question_type="open_answer",
                text=question_text,
                points=points,
                options=[],
                image_source=image_path,
                expected_answer=expected_answer,
                rubric=rubric,
                answer_area=PexamAnswerArea(lines=answer_lines, height_mm=height_mm, show_lines=show_lines),
                font_size=font_size,
            ))
            continue

        first_answer_idx = -1
        for i, line in enumerate(content_lines):
            stripped = line.strip()
            if stripped.startswith('* ') or stripped.startswith('- ') or re.match(r'^\d+\.\s', stripped):
                first_answer_idx = i
                break
        
        if first_answer_idx == -1:
            logging.warning(f"Could not find any answers for question ID '{question_id_str}'. Skipping.")
            continue
            
        question_text = "\n".join(content_lines[:first_answer_idx]).strip()
        
        answer_and_exp_lines = content_lines[first_answer_idx:]
        
        answers_text = []
        explanation_lines = []
        is_parsing_exp = False
        
        for line in answer_and_exp_lines:
            stripped = line.strip()
            if stripped.lower().startswith("**explanation:**"):
                is_parsing_exp = True
                continue
            
            if is_parsing_exp:
                explanation_lines.append(line)
                continue
            
            if stripped.startswith(('* ', '- ')):
                answers_text.append(re.sub(r'^[\*\-]\s*', '', stripped))
            elif re.match(r'^\d+\.\s', stripped):
                 answers_text.append(re.sub(r'^\d+\.\s*', '', stripped))

        if not answers_text:
            continue

        options = []
        # First answer is correct
        options.append(PexamOption(text=answers_text[0], is_correct=True))
        # Rest are distractors
        for dist in answers_text[1:]:
             options.append(PexamOption(text=dist, is_correct=False))

        explanation = "\n".join(explanation_lines).strip() or None
        
        questions.append(PexamQuestion(
            id=question_id_str,
            question_type="multiple_choice",
            text=question_text,
            points=points,
            options=options,
            image_source=image_path,
            explanation=explanation,
            font_size=font_size,
        ))
        
    return questions

def save_questions_to_md(questions: List[PexamQuestion], output_file: str):
    """
    Saves a list of PexamQuestion objects to a Markdown file following the specification.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for q in questions:
            # Header
            attrs = []
            if q.question_type != "multiple_choice":
                attrs.append(f"type={q.question_type}")
            if q.points != 1.0:
                attrs.append(f"points={q.points:g}")
            if q.font_size:
                attrs.append(f"font_size={q.font_size}")
            if q.is_open_answer:
                attrs.append(f"lines={q.answer_area.lines}")
                if q.answer_area.show_lines:
                    attrs.append("show_lines=true")
                if q.answer_area.height_mm is not None:
                    attrs.append(f"height_mm={q.answer_area.height_mm:g}")
            attr_text = f" {{{' '.join(attrs)}}}" if attrs else ""
            f.write(f"## {q.id}{attr_text}\n")
            
            # Image (in blockquote)
            if q.image_source:
                # Try to make path relative to output file if possible
                image_path = q.image_source
                try:
                    rel_path = os.path.relpath(q.image_source, os.path.dirname(os.path.abspath(output_file)))
                    if not rel_path.startswith(".."):
                         image_path = rel_path
                except ValueError:
                    pass # Keep absolute if on different drive
                
                f.write(f"> ![Image for question]({image_path})\n")
            
            # Question text
            f.write(f"{q.text}\n")

            if q.is_open_answer:
                if q.expected_answer:
                    f.write(f"\n**Expected answer:**\n{q.expected_answer}\n")
                if q.rubric:
                    f.write(f"\n**Rubric:**\n{q.rubric}\n")
                f.write("\n")
                continue
            
            # Options (First one is correct)
            # Find correct option
            correct_opt = next((o for o in q.options if o.is_correct), None)
            other_opts = [o for o in q.options if not o.is_correct]
            
            if correct_opt:
                f.write(f" * {correct_opt.text}\n")
            
            for opt in other_opts:
                f.write(f" * {opt.text}\n")
                
            # Explanation
            if q.explanation:
                f.write(f"\n**Explanation:**\n{q.explanation}\n")
            
            f.write("\n")
