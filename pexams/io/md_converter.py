import re
import os
import logging
from dataclasses import dataclass, field
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


_FENCE_RE = re.compile(r'^\s{0,3}(`{3,}|~{3,})')
_OPTION_RE = re.compile(r'^(?:[*-]|\d+\.)\s')
_LEADING_IMAGE_RE = re.compile(r'^>\s*!\[([^\]]*)\]\(([^)]*)\)')
_IMAGE_SIZE_RE = re.compile(r'^(w|h):(\d+(?:\.\d+)?)(px|mm|cm|%)?$', re.IGNORECASE)
IMAGE_ALIGNMENTS = ("left", "right", "center")


def _code_fence_mask(lines: List[str]) -> List[bool]:
    """True for every line that belongs to a fenced code block (``` or ~~~), delimiters included."""
    mask = []
    fence = None
    for line in lines:
        match = _FENCE_RE.match(line)
        if fence is None:
            if match:
                fence = match.group(1)
            mask.append(fence is not None)
        else:
            mask.append(True)
            if match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence) \
                    and not line.strip().strip(fence[0]):
                fence = None
    return mask


def _strip_html_comments(content: str) -> str:
    """Remove <!-- ... --> comments, except inside fenced code blocks."""
    lines = content.split('\n')
    mask = _code_fence_mask(lines)
    chunks, current, in_code = [], [], False
    for line, is_code in zip(lines, mask):
        if current and is_code != in_code:
            chunks.append((in_code, '\n'.join(current)))
            current = []
        current.append(line)
        in_code = is_code
    chunks.append((in_code, '\n'.join(current)))
    return '\n'.join(text if is_code else re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL) for is_code, text in chunks)


def _split_question_blocks(content: str) -> List[List[str]]:
    """Split a questions file into the lines of each '## id' block, ignoring '## ' inside code blocks."""
    lines = _strip_html_comments(content).split('\n')
    blocks: List[List[str]] = []
    for line, is_code in zip(lines, _code_fence_mask(lines)):
        if not is_code and line.startswith('## '):
            blocks.append([line])
        elif blocks:
            blocks[-1].append(line)
    return blocks


def parse_image_alt(alt: str) -> Tuple[dict, str]:
    """Split an image's alt text into its layout spec and the remaining alt/caption text.

    Grammar: ``[spec|]text`` or ``spec``, where spec is space-separated tokens ``w:<size>``,
    ``h:<size>`` (size in px, mm, cm or %; a plain number is px) and one of ``left``, ``right``,
    ``center``. ``'w:60mm right|Chest X-ray'`` gives ``({'width': '60mm', 'align': 'right'},
    'Chest X-ray')``. Alt text that is not a valid spec (e.g. ``'left ventricle'``) is all text.
    """
    alt = (alt or "").strip()
    spec_text, sep, caption = alt.partition("|")
    layout = {}
    for token in spec_text.split():
        size = _IMAGE_SIZE_RE.match(token)
        if size:
            layout["width" if size.group(1).lower() == "w" else "height"] = size.group(2) + (size.group(3) or "px").lower()
        elif token.lower() in IMAGE_ALIGNMENTS:
            layout["align"] = token.lower()
        else:
            return {}, alt
    return layout, caption.strip() if sep else ""


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

    for line, is_code in zip(lines, _code_fence_mask(lines)):
        marker = line.strip().lower()
        if not is_code and marker in section_markers:
            current = section_markers[marker]
            continue
        sections[current].append(line)

    question_text = "\n".join(sections["question"]).strip()
    expected_answer = "\n".join(sections["expected_answer"]).strip() or None
    rubric = "\n".join(sections["rubric"]).strip() or None
    answer_area = "\n".join(sections["answer_area"]).strip() or None
    return question_text, expected_answer, rubric, answer_area


def _parse_answer_area(attrs: dict, answer_area_text: Optional[str]) -> PexamAnswerArea:
    """Answer box size from the header attributes, overridden by an '**Answer area:**' section."""
    answer_lines = _parse_float_attr(attrs, "lines", 8) or 8
    height_mm = _parse_float_attr(attrs, "height_mm")
    show_lines = _parse_bool_attr(attrs, "show_lines", False)
    if answer_area_text:
        line_match = re.search(r'lines\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', answer_area_text, flags=re.IGNORECASE)
        if line_match:
            answer_lines = float(line_match.group(1))
        height_match = re.search(r'height_mm\s*[:=]\s*([0-9.]+)', answer_area_text, flags=re.IGNORECASE)
        if height_match:
            height_mm = float(height_match.group(1))
        show_lines_match = re.search(r'show_lines\s*[:=]\s*(true|false|yes|no|1|0|on|off)', answer_area_text, flags=re.IGNORECASE)
        if show_lines_match:
            show_lines = _parse_bool_attr({"show_lines": show_lines_match.group(1)}, "show_lines", show_lines)
    return PexamAnswerArea(lines=answer_lines, height_mm=height_mm, show_lines=show_lines)


@dataclass
class MarkdownQuestion:
    """One '## id' block of a questions file, before any path resolution or validation."""
    id: str
    attrs: dict
    question_type: str
    text: str
    options: List[str] = field(default_factory=list)  # multiple choice, correct one first
    explanation: Optional[str] = None
    expected_answer: Optional[str] = None
    rubric: Optional[str] = None
    answer_area: PexamAnswerArea = field(default_factory=PexamAnswerArea)


def _take_leading_image(content_lines: List[str]) -> Tuple[Optional[str], List[str]]:
    """Take the legacy image line ``> ![..](img.png)`` at the top of a question.

    Images are ordinary Markdown images in the text. A plain legacy line returns the image as
    Markdown, to be appended below the text (where it was always printed). One with a layout
    spec (``> ![w:40mm right](img.png)``) is moved into the first line of the text.
    """
    while content_lines and not content_lines[0].strip():
        content_lines = content_lines[1:]
    match = _LEADING_IMAGE_RE.match(content_lines[0].strip()) if content_lines else None
    if not match:
        return None, content_lines
    alt, path = match.group(1), match.group(2).strip()
    image_md = f"![{alt}]({path})"
    rest = content_lines[1:]
    if not parse_image_alt(alt)[0]:
        return image_md, rest
    while rest and not rest[0].strip():
        rest = rest[1:]
    if rest and not _FENCE_RE.match(rest[0]) and not rest[0].lstrip().startswith("$$"):
        return None, [f"{image_md} {rest[0]}"] + rest[1:]
    return None, [image_md, ""] + rest


def _with_image_below(text: str, image_md: Optional[str]) -> str:
    return f"{text}\n\n{image_md}" if image_md else text


def parse_questions_markdown(content: str) -> List[MarkdownQuestion]:
    """Parse the text of a questions file into its question blocks.

    Shared by ``load_questions_from_md`` and ptestgen, so both read the format the same way.
    Lines inside fenced code blocks are never taken as headers, options or section markers.
    """
    questions: List[MarkdownQuestion] = []
    for lines in _split_question_blocks(content):
        question_id, attrs = _parse_question_header(lines[0])
        if not question_id:
            continue
        image_below, content_lines = _take_leading_image(lines[1:])
        question_type = _normalize_question_type(attrs.get("type"))

        if question_type == "open_answer":
            question_text, expected_answer, rubric, answer_area_text = _split_open_answer_sections(content_lines)
            if not question_text:
                logging.warning(f"Open-answer question ID '{question_id}' has no question text. Skipping.")
                continue
            questions.append(MarkdownQuestion(
                id=question_id, attrs=attrs, question_type=question_type,
                text=_with_image_below(question_text, image_below),
                expected_answer=expected_answer, rubric=rubric,
                answer_area=_parse_answer_area(attrs, answer_area_text),
            ))
            continue

        code_mask = _code_fence_mask(content_lines)
        first_answer_idx = next(
            (i for i, line in enumerate(content_lines) if not code_mask[i] and _OPTION_RE.match(line.strip())), -1
        )
        if first_answer_idx == -1:
            logging.warning(f"Could not find any answers for question ID '{question_id}'. Skipping.")
            continue

        options, explanation_lines = [], []
        is_parsing_exp = False
        for line in content_lines[first_answer_idx:]:
            stripped = line.strip()
            if is_parsing_exp:
                explanation_lines.append(line)
            elif stripped.lower().startswith("**explanation:**"):
                is_parsing_exp = True
            elif _OPTION_RE.match(stripped):
                options.append(_OPTION_RE.sub('', stripped, count=1).strip())
        if not options:
            continue

        questions.append(MarkdownQuestion(
            id=question_id, attrs=attrs, question_type=question_type,
            text=_with_image_below("\n".join(content_lines[:first_answer_idx]).strip(), image_below), options=options,
            explanation="\n".join(explanation_lines).strip() or None,
        ))
    return questions


def load_questions_from_md(path: str) -> List[PexamQuestion]:
    """
    Parses a Markdown file containing questions into a list of PexamQuestion objects.
    
    Format specification (see the README for open-answer questions, code, math and images):
    ## question_id
    Question text, with images anywhere: ![w:60mm right|Caption](image.png)
    * Correct answer
    * Wrong answer 1
    * Wrong answer 2
    
    **Explanation:**
    Explanation text...
    """
    if not os.path.exists(path):
        logging.error(f"Questions markdown file not found: {path}")
        return []

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    questions: List[PexamQuestion] = []
    for parsed in parse_questions_markdown(content):
        common = dict(
            id=parsed.id,
            text=parsed.text,
            points=_parse_float_attr(parsed.attrs, "points", 1.0) or 1.0,
            font_size=parsed.attrs.get("font_size"),
        )
        if parsed.question_type == "open_answer":
            questions.append(PexamQuestion(
                question_type="open_answer",
                options=[],
                expected_answer=parsed.expected_answer,
                rubric=parsed.rubric,
                answer_area=parsed.answer_area,
                **common,
            ))
        else:
            # First answer is correct, the rest are distractors.
            options = [PexamOption(text=text, is_correct=i == 0) for i, text in enumerate(parsed.options)]
            questions.append(PexamQuestion(
                question_type="multiple_choice",
                options=options,
                explanation=parsed.explanation,
                **common,
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
                attrs.append(f"lines={q.answer_area.lines:g}")
                if q.answer_area.show_lines:
                    attrs.append("show_lines=true")
                if q.answer_area.height_mm is not None:
                    attrs.append(f"height_mm={q.answer_area.height_mm:g}")
            attr_text = f" {{{' '.join(attrs)}}}" if attrs else ""
            f.write(f"## {q.id}{attr_text}\n")
            
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
