"""Exam PDF generation: HTML for the answer sheet and question pages, rendered and paginated
with Playwright, plus the per-model question files and open-answer box positions."""
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Union

from pypdf import PdfReader, PdfWriter

from pexams import layout, utils
from pexams.markdown_render import absolute_markdown_images, limit_image_size as _limit_image_size, render_markdown as _render_markdown, resolve_relative_media_uris as _resolve_relative_media_uris
from pexams.pdf_render import MATHJAX_ARITHMATEX_CONFIG, PlaywrightError, head_assets, html_to_pdf
from pexams.print_layout import (
    PRINT_CONTENT_WIDTH_MM, PX_PER_MM_CSS, extract_open_answer_area_metadata,
    keep_questions_within_pages, wrap_pages_with_header_footer,
)
from pexams.schemas import PexamExam, PexamQuestion
from pexams.translations import LANG_STRINGS

EXAM_FONTS_URL = ("https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,300..800;1,300..800"
                  "&family=Roboto:ital,wght@0,300;0,400;0,500;0,700&display=swap")
COLUMN_CLASSES = {1: "", 2: "two-columns", 3: "three-columns"}


def _abs_style(x, y, width=None, height=None) -> str:
    """Inline style placing an element at (x, y) mm, optionally sized in mm."""
    style = f"position: absolute; left: {x}mm; top: {y}mm;"
    if width is not None:
        style += f" width: {width}mm;"
    if height is not None:
        style += f" height: {height}mm;"
    return style


def _box_style(coords) -> str:
    """Inline style for a layout rectangle (``top_left``/``bottom_right`` in mm)."""
    x, y = coords.top_left
    return _abs_style(x, y, coords.bottom_right[0] - x, coords.bottom_right[1] - y)


def _answer_sheet_header_elements(layout_data, exam_model: int, exam_title: str, exam_course: Optional[str],
                                  exam_date: Optional[str], selected_lang: dict) -> List[str]:
    """Title, model box and course/date line."""
    elements = [
        f'<h1 class="exam-title" style="{_abs_style(*layout_data.exam_title, layout_data.exam_title_width)}">{exam_title}</h1>',
        f'<div class="model-id-box" style="{_box_style(layout_data.model_id_box)}"></div>',
        # The model number is positioned separately, over the box
        f'<div class="model-id-text" style="{_box_style(layout_data.model_id_box)}">{exam_model}</div>',
    ]
    exam_info_parts = []
    if exam_course:
        exam_info_parts.append(f"<span>{selected_lang['course']}: {exam_course}</span>")
    if exam_date:
        exam_info_parts.append(f"<span>{selected_lang['date']}: {exam_date}</span>")
    exam_info_html = "\n".join(exam_info_parts)
    style_info = _abs_style(*layout_data.exam_info, layout_data.exam_info_width)
    elements.append(f'<div class="exam-info" style="{style_info}">{exam_info_html}</div>')
    return elements


def _student_box_elements(layout_data, selected_lang: dict) -> List[str]:
    """Labelled name, ID and signature boxes."""
    elements = []
    for label_class, label_pos, label_key, box_class, box in (
        ("student-name-label", layout_data.student_name_label, "name", "student-name-box", layout_data.student_name_box),
        ("student-id-label", layout_data.student_id_label, "id", "id-box", layout_data.student_id_box),
        ("student-signature-label", layout_data.student_signature_label, "signature",
         "student-signature-box", layout_data.student_signature_box),
    ):
        elements.append(f'<div class="{label_class}" style="{_abs_style(*label_pos)}"><b>{selected_lang[label_key]}</b></div>')
        elements.append(f'<div class="{box_class}" style="{_box_style(box)}"></div>')
    return elements


def _instructions_element(layout_data, selected_lang: dict) -> str:
    """Marking instructions with a correct and a crossed-out example box."""
    example_correct_html = '<div class="example-box correct"></div>'
    example_incorrect_html = '<div class="example-box incorrect"><div class="incorrect-line"></div></div>'

    instructions_html = f"""
    <div class="instructions-box">
        <h4>{selected_lang.get('instructions_title', 'Instructions')}</h4>
        <ul>
            <li>{selected_lang.get('instructions_answers', '')}</li>
            <li class="instruction-example-container">
                <div class="instruction-example">
                    <span>{selected_lang.get('instructions_example_correct', '')}</span>
                    {example_correct_html}
                </div>
                <div class="instruction-example">
                    <span>{selected_lang.get('instructions_example_incorrect', '')}</span>
                    {example_incorrect_html}
                </div>
            </li>
            <li>{selected_lang.get('instructions_corrections', '')}</li>
        </ul>
    </div>
    """
    return f'<div style="{_abs_style(*layout_data.instructions)}">{instructions_html}</div>'


def _bubble_grid_elements(layout_data) -> List[str]:
    """Option letters above each column group, question numbers and the bubbles."""
    elements = []
    for labels in layout_data.header_labels.values():
        for i, (x, y) in labels.items():
            style = _abs_style(x, y, layout.HEADER_OPTION_LABEL_WIDTH, layout.HEADER_ROW_HEIGHT)
            elements.append(f'<div class="header-option" style="{style}">{chr(ord("A") + i)}</div>')

    for q_id, (x, y) in layout_data.question_numbers.items():
        style = _abs_style(x, y, layout.QUESTION_NUMBER_WIDTH, layout.ANSWER_ROW_HEIGHT)
        elements.append(f'<div class="question-number" style="{style}">{q_id}</div>')

    for options in layout_data.answer_boxes.values():
        for coords in options.values():
            style = _abs_style(*coords.top_left, layout.OPTION_BOX_WIDTH, layout.OPTION_BOX_HEIGHT)
            elements.append(f'<div class="option-box" style="{style}"></div>')
    return elements


def _generate_answer_sheet_html(
    questions: List[PexamQuestion],
    exam_model: int,
    exam_title: str,
    exam_course: Optional[str],
    exam_date: Optional[str],
    lang: str = "en",
    header_only: bool = False,
) -> str:
    """Generates the pure HTML for the answer sheet with absolutely positioned elements.

    With ``header_only`` only the title, model box and student fields are returned (no
    instructions or answer grid), at the same coordinates as on the answer sheet, so they
    can be placed at the top of the first question page and still be read by the corrector.
    """
    selected_lang = LANG_STRINGS.get(lang, LANG_STRINGS["en"])
    layout_data = layout.get_answer_sheet_layout(questions)

    html_elements = _answer_sheet_header_elements(layout_data, exam_model, exam_title, exam_course, exam_date, selected_lang)
    html_elements += _student_box_elements(layout_data, selected_lang)

    if header_only:
        all_elements_html = "\n".join(html_elements)
        return f'<div class="first-page-header">\n{all_elements_html}\n</div>'

    html_elements.append(_instructions_element(layout_data, selected_lang))
    html_elements += _bubble_grid_elements(layout_data)
    all_elements_html = "\n".join(html_elements)

    return f"""
<div class="page-container answer-sheet-page">
    <div class="fiducial top-left"></div>
    <div class="fiducial top-right"></div>
    <div class="fiducial bottom-left"></div>
    <div class="fiducial bottom-right"></div>
    {all_elements_html}
</div>
"""


SHOW_POINTS_CHOICES = ("auto", "always", "never")

# Languages written with a decimal point; the rest of LANG_STRINGS use a decimal comma.
_DECIMAL_POINT_LANGS = {"en"}


def _format_points(points: float, lang: str = "en") -> str:
    """Format a question's points for printing, e.g. ``[1.5 pt]`` or ``[1,5 pt]``."""
    value = f"{points:.2f}".rstrip("0").rstrip(".")
    if lang not in _DECIMAL_POINT_LANGS:
        value = value.replace(".", ",")
    return f"[{value} pt]"


def _should_show_points(questions: List[PexamQuestion], show_points: str = "auto") -> bool:
    """Resolve ``show_points``: ``auto`` prints them only when they tell the student something,
    that is, when there are open-answer questions or questions of different value."""
    if show_points not in SHOW_POINTS_CHOICES:
        raise ValueError(f"show_points must be one of {SHOW_POINTS_CHOICES}, got {show_points!r}.")
    if show_points != "auto":
        return show_points == "always"
    distinct_points = {round(float(q.points), 6) for q in questions}
    return any(q.is_open_answer for q in questions) or len(distinct_points) > 1


def _generate_questions_markdown(
    questions: List[PexamQuestion],
    show_points: bool = False,
    lang: str = "en",
) -> str:
    """Generates the Markdown for the question pages.

    show_points: Print each question's points (e.g. ``[1.5 pt]``) right after its number.
    """
    md_parts = []
    for q in questions:
        question_classes = ["question-wrapper"]
        if q.is_open_answer:
            question_classes.append("open-answer-question")
        question_style = f' style="font-size: {q.font_size};"' if q.font_size else ""
        md_parts.append(f'\n<div class="{" ".join(question_classes)}" data-question-id="{q.id}"{question_style}>\n')

        # Convert question text to HTML. Single newlines become <br> (nl2br), blank lines separate paragraphs.
        question_text_html = _render_markdown(q.text, nl2br=True)
        number_html = f"<b>{q.id}.</b> "
        if show_points:
            number_html += f'<span class="question-points">{_format_points(q.points, lang)}</span> '
        if question_text_html.startswith("<p>") and question_text_html.endswith("</p>") and question_text_html.count("<p>") == 1:
            question_text_html = number_html + question_text_html[3:-4]
        elif question_text_html.startswith("<p>"):
            question_text_html = question_text_html.replace("<p>", "<p>" + number_html, 1)
        else:
            question_text_html = number_html + question_text_html
        
        question_text_html = _limit_image_size(question_text_html, q.max_image_width, q.max_image_height)
        md_parts.append(f'<div class="question-text">{question_text_html}</div>\n')
        
        if q.is_open_answer:
            height_style = f"min-height: {max(1, q.answer_area.lines) * 7 + 6}mm;"
            if q.answer_area.height_mm is not None:
                height_style = f"height: {q.answer_area.height_mm}mm;"
            # Fractional lines only enlarge the box; guide lines are drawn for whole lines.
            line_count = max(1, int(q.answer_area.lines))
            fixed_height_attr = ' data-fixed-height="true"' if q.answer_area.height_mm is not None else ""
            md_parts.append(
                f'<div class="open-answer-box" data-question-id="{q.id}" '
                f'data-lines="{q.answer_area.lines:g}"{fixed_height_attr} style="{height_style}">'
            )
            if q.answer_area.show_lines:
                for _ in range(line_count):
                    md_parts.append('<div class="open-answer-line"></div>')
            md_parts.append("</div>")
        else:
            md_parts.append('<div class="options-block">')
            for i, option in enumerate(q.options):
                option_label = chr(ord('A') + i)
                # Convert option text to HTML, ensuring it's a single paragraph block
                option_text_html = _render_markdown(option.text.replace('\n', ' <br> '))
                if option_text_html.startswith("<p>"):
                    option_text_html = option_text_html[3:-4]

                md_parts.append(f'<div class="option-item"><span class="option-label"><b>{option_label})</b></span><span class="option-text">{option_text_html}</span></div>')
            md_parts.append("</div>") # Close options-block
            
        md_parts.append('</div>\n')

    return "\n".join(md_parts)


def _create_mass_exam_pdf(model_pdfs: List[str], total_students: int, output_dir: str, extra_model_templates: int = 0):
    writer = PdfWriter()
    num_models = len(model_pdfs)
    
    logging.info(f"Generating single PDF for {total_students} students...")
    
    model_readers = []
    for pdf_path in model_pdfs:
        model_readers.append(PdfReader(pdf_path))
        
    for i in range(total_students):
        model_idx = i % num_models
        reader = model_readers[model_idx]
        
        # Add all pages from the model
        for page in reader.pages:
            writer.add_page(page)
            
        # If odd number of pages, add a blank page for double-sided printing
        if len(reader.pages) % 2 != 0:
             writer.add_blank_page()
             
    if extra_model_templates > 0:
        logging.info(f"Appending {extra_model_templates} extra template sheets per model...")
        for i, pdf_path in enumerate(model_pdfs):
            reader = model_readers[i]
            # Assumes the answer sheet is the first page
            first_page = reader.pages[0]
            
            for _ in range(extra_model_templates):
                writer.add_page(first_page)
                # Add blank page to ensure each template is on a separate sheet if printed duplex
                writer.add_blank_page()

    output_path = os.path.join(output_dir, "all_exams.pdf")
    writer.write(output_path)
    logging.info(f"Saved mass exam PDF to {output_path}")


def _load_questions(questions: Union[List[PexamQuestion], str]) -> Optional[List[PexamQuestion]]:
    """The questions themselves, or those of a questions JSON file (None if it cannot be read)."""
    if not isinstance(questions, str):
        return questions
    if not os.path.exists(questions):
        logging.error(f"Questions JSON file not found at: {questions}")
        return None
    logging.info(f"Loading questions from: {questions}")
    try:
        return PexamExam.model_validate_json(Path(questions).read_text(encoding="utf-8")).questions
    except Exception as e:
        logging.error(f"Failed to parse questions JSON file: {e}")
        return None


def _check_and_scale_mc_points(questions: List[PexamQuestion], mc_total_points: Optional[float]) -> None:
    """Check the number of multiple-choice questions and, with ``mc_total_points``, share those
    points equally among them (questions with their own points are not allowed then)."""
    mc_questions = [q for q in questions if q.is_multiple_choice]
    if len(mc_questions) > layout.MAX_QUESTIONS:
        logging.error(f"Too many multiple-choice questions ({len(mc_questions)}). Max allowed is {layout.MAX_QUESTIONS}.")
        raise ValueError(f"Too many multiple-choice questions ({len(mc_questions)}). Max allowed is {layout.MAX_QUESTIONS}.")
    if mc_total_points is None:
        return
    if mc_total_points <= 0:
        raise ValueError("--mc-total-points must be greater than 0.")
    explicit_points = [q for q in mc_questions if q.points != 1.0]
    if explicit_points:
        explicit_ids = ", ".join(str(q.original_id or q.id) for q in explicit_points)
        raise ValueError(
            "--mc-total-points cannot be combined with per-question MC points. "
            f"Questions with explicit non-default points: {explicit_ids}"
        )
    if mc_questions:
        points_per_question = mc_total_points / len(mc_questions)
        for q in mc_questions:
            q.points = points_per_question


def _fakes_per_model(generate_fakes: int, num_models: int) -> List[int]:
    """Number of simulated scans of each model, spread as evenly as possible."""
    if generate_fakes <= 0:
        return []
    if num_models <= 0:
        logging.warning("generate_fakes is greater than 0 but num_models is 0. No fakes will be generated.")
        return []
    base, remainder = divmod(generate_fakes, num_models)
    return [base + 1 if i < remainder else base for i in range(num_models)]


def _custom_header_html(custom_header: Optional[Union[str, Path]], markdown_asset_base_dir: Optional[str]) -> str:
    """The custom header (Markdown text or a .md file) as HTML, with its relative media resolved."""
    if not custom_header:
        return ""
    is_file = str(custom_header).strip().lower().endswith(".md") and os.path.exists(str(custom_header))
    header_content = custom_header
    if is_file:
        logging.info(f"Loading custom header from file: {custom_header}")
        with open(custom_header, 'r', encoding='utf-8') as f:
            header_content = f.read()
    header_html = f'<div class="custom-header">{_render_markdown(str(header_content))}</div>'
    resolve_base = os.path.dirname(os.path.abspath(str(custom_header))) if is_file else markdown_asset_base_dir
    return _resolve_relative_media_uris(header_html, resolve_base) if resolve_base else header_html


def _shuffle_model_options(model_questions: List[PexamQuestion]) -> None:
    """Shuffle the options of each multiple-choice question (in place) and keep its correct one marked."""
    for q in model_questions:
        if q.is_multiple_choice and q.options and q.correct_answer_index is not None:
            original_correct_option = q.options[q.correct_answer_index]
            utils.shuffle_options_for_question(q)
            try:
                new_correct_index = q.options.index(original_correct_option)
                for opt_idx, option in enumerate(q.options):
                    option.is_correct = (opt_idx == new_correct_index)
            except ValueError:
                logging.error(f"Could not find the original correct answer for question {q.id} after shuffling. This should not happen.")


def _exam_html(model_questions: List[PexamQuestion], model_number: int, exam_title: str, exam_course: Optional[str],
               exam_date: Optional[str], lang: str, css_content: str, font_size: str, column_class: str,
               custom_header_html: str, print_points: bool, markdown_asset_base_dir: Optional[str]) -> str:
    """The complete HTML document of one exam model."""
    model_mc_questions = [q for q in model_questions if q.is_multiple_choice]
    # Without multiple-choice questions there is nothing to mark on an answer sheet: the
    # student header goes at the top of the first question page instead, saving the answer
    # sheet and the blank page after it. It keeps the answer-sheet coordinates, so the
    # corrector reads the model, ID and name from the first page as usual.
    header_on_first_page = not model_mc_questions
    answer_sheet_html = _generate_answer_sheet_html(
        model_mc_questions, model_number, exam_title=exam_title, exam_course=exam_course, exam_date=exam_date,
        lang=lang, header_only=header_on_first_page,
    )
    if header_on_first_page:
        first_page_header_html = answer_sheet_html
        # Push the questions (all columns) below the student boxes; the container adds its 11mm top padding.
        header_bottom_mm = layout.STUDENT_NAME_BOX_TL[1] + layout.STUDENT_NAME_BOX_HEIGHT
        spacer_html = f'<div class="first-page-header-spacer" style="height: {header_bottom_mm + 5 - 11}mm;"></div>'
        pages_before_questions_html = ""
    else:
        first_page_header_html = ""
        spacer_html = ""
        pages_before_questions_html = f"""{answer_sheet_html}
    <div class="page-container" style="page-break-after: always;"></div>"""
    # Already HTML: a second Markdown pass could only alter code blocks and math.
    questions_html = _generate_questions_markdown(model_questions, show_points=print_points, lang=lang)
    if markdown_asset_base_dir:
        questions_html = _resolve_relative_media_uris(questions_html, markdown_asset_base_dir)

    head = head_assets(fonts_url=EXAM_FONTS_URL, mathjax_config=MATHJAX_ARITHMATEX_CONFIG)
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{exam_title} - Model {model_number}</title>
    {head}
    <style>
        {css_content}
    </style>
    <style>
        body {{ font-size: {font_size}; }}
    </style>
</head>
<body>
    {pages_before_questions_html}
    <div class="page-container questions-page">
        <div class="fiducial top-left"></div>
        <div class="fiducial top-right"></div>
        <div class="fiducial bottom-left"></div>
        <div class="fiducial bottom-right"></div>
        {first_page_header_html}
        {spacer_html}
        <div class="questions-container {column_class}">
        {custom_header_html}
        {questions_html}
        </div>
    </div>
</body>
</html>
"""


def _render_exam_pdf(html_path: str, pdf_path: str, model_questions: List[PexamQuestion], model_number: int,
                     exam_title: str, exam_date: Optional[str], fill_answer_space: bool) -> List[dict]:
    """Print one exam model to PDF and return the measured positions of its open-answer boxes."""
    def lay_out(page):
        # Lay the page out exactly as it will be printed (A4 content width, print media),
        # keep questions and answer boxes from crossing page breaks, and only then
        # measure the open-answer areas so their coordinates match the printed pages.
        page.emulate_media(media="print")
        page.set_viewport_size({"width": round(PRINT_CONTENT_WIDTH_MM * PX_PER_MM_CSS), "height": 1200})
        n_pages = keep_questions_within_pages(page, fill_answer_space=fill_answer_space)
        logging.info(f"Questions laid out on {n_pages} page(s) with per-page fiducial markers.")
        areas = extract_open_answer_area_metadata(page, model_questions, model_number)
        # Every printed page is now its own container: draw header and footer inside
        # each A4 sheet so they use the exam font. Chromium's header/footer templates
        # cannot load web or embedded fonts and fall back to Times.
        header_text = f"{exam_title} - {exam_date}" if exam_date else exam_title
        wrap_pages_with_header_footer(page, header_text, f"Model {model_number}")
        return areas

    return html_to_pdf(html_path, pdf_path, before_pdf=lay_out, prefer_css_page_size=True,
                       margin={'top': '0', 'bottom': '0', 'left': '0', 'right': '0'})


def _generate_test_scans(pdf_path: str, model_questions: List[PexamQuestion], output_dir: str, model_number: int,
                         generate_references: bool, fakes_per_model: List[int]) -> None:
    from pexams.simulated_scans import generate_reference_scan, generate_simulated_scan

    if generate_references:
        logging.info(f"Generating reference scan for model {model_number}")
        generate_reference_scan(pdf_path, model_questions, output_dir, model_number)
    if not fakes_per_model:
        return
    if model_number > len(fakes_per_model):
        logging.warning(f"Skipping fake generation for model {model_number} due to index out of range. "
                        "This can happen if num_models is inconsistent.")
        return
    num_fakes = fakes_per_model[model_number - 1]
    if num_fakes > 0:
        logging.info(f"Test mode enabled: Generating {num_fakes} simulated scan(s) for model {model_number}")
        for fake_idx in range(1, num_fakes + 1):
            generate_simulated_scan(pdf_path, model_questions, output_dir, f"{model_number}_{fake_idx}")


def generate_exams(
    questions: Union[List[PexamQuestion], str],
    output_dir: str,
    num_models: int = 4,
    exam_title: str = "Final Exam",
    exam_course: Optional[str] = None,
    exam_date: Optional[str] = None,
    columns: int = 1,
    lang: str = "en",
    keep_html: bool = False,
    font_size: str = "10pt",
    generate_fakes: int = 0,
    generate_references: bool = False,
    total_students: int = 0,
    extra_model_templates: int = 0,
    custom_header: Optional[Union[str, Path]] = None,
    markdown_asset_base_dir: Optional[str] = None,
    mc_total_points: Optional[float] = None,
    fill_answer_space: bool = False,
    show_points: str = "auto",
):
    """
    Generates exam PDFs from a list of questions using Playwright.
    The questions can be provided as a list of PexamQuestion objects or a path to a JSON file.

    markdown_asset_base_dir: Directory used to resolve relative paths in markdown images
    (e.g. ``![](fig.png)``) and similar ``src`` attributes to file:// URIs for PDF rendering.
    Typically the directory containing the source ``.md`` file.

    fill_answer_space: Grow the open-answer boxes of each page to use the free space left
    at its bottom (boxes with an explicit ``height_mm`` keep their size).

    show_points: Print each question's points (e.g. ``[1.5 pt]``) right after its number:
    ``always``, ``never``, or ``auto`` (default), which prints them only when the exam has
    open-answer questions or questions of different value. The printed value is the
    question's own ``points`` (after ``mc_total_points``), not rescaled to any total.
    """
    logging.info("Starting pexams PDF generation.")
    questions_list = _load_questions(questions)
    if questions_list is None:
        return
    _check_and_scale_mc_points(questions_list, mc_total_points)
    print_points = _should_show_points(questions_list, show_points)

    logging.info(f"Loaded {len(questions_list)} questions.")
    logging.info(f"Exams will be output to: {output_dir}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")
    fakes_per_model = _fakes_per_model(generate_fakes, num_models)

    css_path = os.path.join(os.path.dirname(__file__), "pexams.css")
    if not os.path.exists(css_path):
        logging.error(f"CSS theme not found at {css_path}. Cannot generate exams.")
        return
    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()
    custom_header_html = _custom_header_html(custom_header, markdown_asset_base_dir)

    # We work on a copy to avoid mutating the original list if it's reused
    questions_shuffled = list(questions_list)
    utils.shuffle_questions_list(questions_shuffled)

    generated_pdfs = []
    open_answer_areas = []
    for i in range(1, num_models + 1):
        # Each model shuffles the options of its own copy of the (already shuffled) questions.
        model_questions = deepcopy(questions_shuffled)
        _shuffle_model_options(model_questions)
        for q in model_questions:  # the questions JSON is read from other folders (grading, reports)
            q.text = absolute_markdown_images(q.text, markdown_asset_base_dir)
            q.explanation = absolute_markdown_images(q.explanation, markdown_asset_base_dir)
            for option in q.options:
                option.text = absolute_markdown_images(option.text, markdown_asset_base_dir)

        questions_json_path = os.path.join(output_dir, f"exam_model_{i}_questions.json")
        with open(questions_json_path, "w", encoding="utf-8") as f:
            f.write(PexamExam(questions=model_questions).model_dump_json(indent=4))
        logging.info(f"Saved questions for model {i} to: {questions_json_path}")

        html_filepath = os.path.join(output_dir, f"exam_model_{i}.html")
        pdf_filepath = os.path.join(output_dir, f"exam_model_{i}.pdf")
        with open(html_filepath, "w", encoding="utf-8") as f:
            f.write(_exam_html(model_questions, i, exam_title, exam_course, exam_date, lang, css_content, font_size,
                               COLUMN_CLASSES.get(columns, ""), custom_header_html, print_points,
                               markdown_asset_base_dir))
        logging.info(f"Generated HTML for exam model {i}: {html_filepath}")

        try:
            open_answer_areas.extend(_render_exam_pdf(html_filepath, pdf_filepath, model_questions, i, exam_title,
                                                      exam_date, fill_answer_space))
            logging.info(f"Successfully generated PDF for model {i}: {pdf_filepath}")
            generated_pdfs.append(pdf_filepath)
            _generate_test_scans(pdf_filepath, model_questions, output_dir, i, generate_references, fakes_per_model)
        except PlaywrightError as e:
            logging.error(f"Playwright failed to generate PDF for model {i}: {e}")
            logging.error("Have you installed the browser binaries? Try running 'playwright install'")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred while generating PDF for model {i}: {e}")
            break
        finally:
            if not keep_html and os.path.exists(html_filepath):
                os.remove(html_filepath)
                logging.info(f"Removed temporary HTML file: {html_filepath}")

    if open_answer_areas:
        open_areas_path = os.path.join(output_dir, "open_answer_areas.json")
        with open(open_areas_path, "w", encoding="utf-8") as f:
            json.dump(open_answer_areas, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved open-answer area metadata to: {open_areas_path}")

    if total_students > 0 and generated_pdfs:
        _create_mass_exam_pdf(generated_pdfs, total_students, output_dir, extra_model_templates)
