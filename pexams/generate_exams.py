import logging
from typing import List, Optional, Union
import random
import os
import markdown
from pathlib import Path
from playwright.sync_api import sync_playwright, Error as PlaywrightError
from faker import Faker
import cv2
import numpy as np
from copy import deepcopy

from pexams.schemas import PexamQuestion, PexamExam
from pexams import layout
from pexams.translations import LANG_STRINGS

def _generate_answer_sheet_html(
    questions: List[PexamQuestion],
    exam_model: int,
    exam_title: str,
    exam_course: Optional[str],
    exam_date: Optional[str],
    lang: str = "en",
    id_length: int = 10
) -> str:
    """Generates the pure HTML for the answer sheet with absolutely positioned elements."""

    selected_lang = LANG_STRINGS.get(lang, LANG_STRINGS["en"])
    
    layout_data = layout.get_answer_sheet_layout(len(questions), id_length)
    html_elements = []

    # --- Header Elements ---
    style_title = f"position: absolute; left: {layout_data.exam_title[0]}mm; top: {layout_data.exam_title[1]}mm;"
    html_elements.append(f'<h1 class="exam-title" style="{style_title}">{exam_title}</h1>')

    # --- Model ID Box ---
    mib_coords = layout_data.model_id_box
    x_mib, y_mib = mib_coords.top_left
    w_mib = mib_coords.bottom_right[0] - x_mib
    h_mib = mib_coords.bottom_right[1] - y_mib
    style_model_box = f"position: absolute; left: {x_mib}mm; top: {y_mib}mm; width: {w_mib}mm; height: {h_mib}mm;"
    html_elements.append(f'<div class="model-id-box" style="{style_model_box}"></div>')
    # Position the text separately
    style_model_text = f"position: absolute; left: {x_mib}mm; top: {y_mib}mm; width: {w_mib}mm; height: {h_mib}mm;"
    html_elements.append(f'<div class="model-id-text" style="{style_model_text}">{exam_model}</div>')
    
    exam_info_parts = []
    if exam_course: exam_info_parts.append(f"<span>{selected_lang['course']}: {exam_course}</span>")
    if exam_date: exam_info_parts.append(f"<span>{selected_lang['date']}: {exam_date}</span>")
    exam_info_html = "\n".join(exam_info_parts)
    style_info = f"position: absolute; left: {layout_data.exam_info[0]}mm; top: {layout_data.exam_info[1]}mm;"
    html_elements.append(f'<div class="exam-info" style="{style_info}">{exam_info_html}</div>')

    # --- Student Info ---
    style_name_label = f"position: absolute; left: {layout_data.student_name_label[0]}mm; top: {layout_data.student_name_label[1]}mm;"
    html_elements.append(f'<div class="student-name-label" style="{style_name_label}"><b>{selected_lang["name"]}</b></div>')
    
    snb_coords = layout_data.student_name_box
    x_snb, y_snb = snb_coords.top_left
    w_snb = snb_coords.bottom_right[0] - x_snb
    h_snb = snb_coords.bottom_right[1] - y_snb
    style_name_box = f"position: absolute; left: {x_snb}mm; top: {y_snb}mm; width: {w_snb}mm; height: {h_snb}mm;"
    html_elements.append(f'<div class="student-name-box" style="{style_name_box}"></div>')

    style_id_label = f"position: absolute; left: {layout_data.student_id_label[0]}mm; top: {layout_data.student_id_label[1]}mm;"
    html_elements.append(f'<div class="student-id-label" style="{style_id_label}"><b>{selected_lang["id"]}</b></div>')
    
    for id_box_coords in layout_data.student_id_boxes:
        x, y = id_box_coords.top_left
        w = id_box_coords.bottom_right[0] - x
        h = id_box_coords.bottom_right[1] - y
        style = f"position: absolute; left: {x}mm; top: {y}mm; width: {w}mm; height: {h}mm;"
        html_elements.append(f'<div class="id-box" style="{style}"></div>')

    # --- Signature ---
    style_sig_label = f"position: absolute; left: {layout_data.student_signature_label[0]}mm; top: {layout_data.student_signature_label[1]}mm;"
    html_elements.append(f'<div class="student-signature-label" style="{style_sig_label}"><b>{selected_lang["signature"]}</b></div>')

    ssb_coords = layout_data.student_signature_box
    x_ssb, y_ssb = ssb_coords.top_left
    w_ssb = ssb_coords.bottom_right[0] - x_ssb
    h_ssb = ssb_coords.bottom_right[1] - y_ssb
    style_sig_box = f"position: absolute; left: {x_ssb}mm; top: {y_ssb}mm; width: {w_ssb}mm; height: {h_ssb}mm;"
    html_elements.append(f'<div class="student-signature-box" style="{style_sig_box}"></div>')

    # --- Instructions ---
    example_correct_html = '<div class="example-box correct"></div>'
    example_incorrect_html = '<div class="example-box incorrect"><div class="incorrect-line"></div></div>'

    instructions_html = f"""
    <div class="instructions-box">
        <h4>{selected_lang.get('instructions_title', 'Instructions')}</h4>
        <ul>
            <li>{selected_lang.get('instructions_id', '')}</li>
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
    style_instructions = f"position: absolute; left: {layout_data.instructions[0]}mm; top: {layout_data.instructions[1]}mm;"
    html_elements.append(f'<div style="{style_instructions}">{instructions_html}</div>')

    # --- Answer Grid Elements ---
    for group_index, labels in layout_data.header_labels.items():
        for i, (x, y) in labels.items():
            label = chr(ord("A") + i)
            style = (f"position: absolute; left: {x}mm; top: {y}mm; "
                     f"width: {layout.HEADER_OPTION_LABEL_WIDTH}mm; height: {layout.HEADER_ROW_HEIGHT}mm;")
            html_elements.append(f'<div class="header-option" style="{style}">{label}</div>')

    for q_id, (x, y) in layout_data.question_numbers.items():
        style = (f"position: absolute; left: {x}mm; top: {y}mm; "
                 f"width: {layout.QUESTION_NUMBER_WIDTH}mm; height: {layout.ANSWER_ROW_HEIGHT}mm;")
        html_elements.append(f'<div class="question-number" style="{style}">{q_id}</div>')

    for q_id, options in layout_data.answer_boxes.items():
        for opt_idx, coords in options.items():
            x, y = coords.top_left
            style = (f"position: absolute; left: {x}mm; top: {y}mm; "
                     f"width: {layout.OPTION_BOX_WIDTH}mm; height: {layout.OPTION_BOX_HEIGHT}mm;")
            html_elements.append(f'<div class="option-box" style="{style}"></div>')
            
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


def _generate_questions_markdown(
    questions: List[PexamQuestion]
) -> str:
    """Generates the Markdown for the question pages."""
    md_parts = []
    # Configure markdown extensions
    extensions = [
        'pymdownx.arithmatex',
        'pymdownx.inlinehilite',
        'fenced_code',
        'codehilite'
    ]
    extension_configs = {
        'pymdownx.arithmatex': {'generic': True}
    }
    for q in questions:
        md_parts.append('\n<div class="question-wrapper">\n')

        # Convert question text to HTML, ensuring it's treated as a single paragraph block
        question_text_html = markdown.markdown(q.text.replace('\n', ' <br> '), extensions=extensions, extension_configs=extension_configs).strip()
        # Remove paragraph tags that markdown lib might add
        if question_text_html.startswith("<p>"):
            question_text_html = question_text_html[3:-4]
        
        md_parts.append(f'<div class="question-text"><b>{q.id}.</b> {question_text_html}</div>\n')
        
        if q.image_source:
            # Convert the path to a file URI to be safe for HTML rendering
            src_path = Path(q.image_source)
            if src_path.exists():
                src = src_path.as_uri()
                md_parts.append(f'<img src="{src}" alt="Image for question {q.id}">\n')

        md_parts.append('<div class="options-block">')
        for i, option in enumerate(q.options):
            option_label = chr(ord('A') + i)
            # Convert option text to HTML, ensuring it's a single paragraph block
            option_text_html = markdown.markdown(option.text.replace('\n', ' <br> '), extensions=extensions, extension_configs=extension_configs).strip()
            if option_text_html.startswith("<p>"):
                option_text_html = option_text_html[3:-4]

            md_parts.append(f'<div class="option-item"><span class="option-label"><b>{option_label})</b></span><span class="option-text">{option_text_html}</span></div>')
        md_parts.append("</div>") # Close options-block
            
        md_parts.append('</div>\n')

    return "\n".join(md_parts)


def generate_exams(
    questions: Union[List[PexamQuestion], str], 
    output_dir: str, 
    num_models: int = 4, 
    exam_title: str = "Final Exam",
    exam_course: Optional[str] = None,
    exam_date: Optional[str] = None,
    columns: int = 1,
    id_length: int = 10,
    lang: str = "en",
    keep_html: bool = False,
    font_size: str = "11pt",
    generate_fakes: int = 0,
    generate_references: bool = False
):
    """
    Generates exam PDFs from a list of questions using Playwright.
    The questions can be provided as a list of PexamQuestion objects or a path to a JSON file.
    """
    logging.info(f"Starting pexams PDF generation.")
    
    if isinstance(questions, str):
        if not os.path.exists(questions):
            logging.error(f"Questions JSON file not found at: {questions}")
            return
        logging.info(f"Loading questions from: {questions}")
        try:
            loaded_exam = PexamExam.model_validate_json(Path(questions).read_text(encoding="utf-8"))
        except Exception as e:
            logging.error(f"Failed to parse questions JSON file: {e}")
            return
        questions_list = loaded_exam.questions
    else:
        questions_list = questions

    logging.info(f"Loaded {len(questions_list)} questions.")
    logging.info(f"Exams will be output to: {output_dir}")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")
        
    fakes_per_model = []
    if generate_fakes > 0:
        if num_models > 0:
            base_fakes = generate_fakes // num_models
            rem_fakes = generate_fakes % num_models
            fakes_per_model = [base_fakes + 1 if i < rem_fakes else base_fakes for i in range(num_models)]
        else:
            logging.warning("generate_fakes is greater than 0 but num_models is 0. No fakes will be generated.")

    css_path = os.path.join(os.path.dirname(__file__), "pexams.css")
    if not os.path.exists(css_path):
        logging.error(f"CSS theme not found at {css_path}. Cannot generate exams.")
        return
        
    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()

    column_classes = {1: "", 2: "two-columns", 3: "three-columns"}
    column_class = column_classes.get(columns, "")

    # --- Shuffle questions once to have a consistent order for all models ---
    questions_shuffled = list(questions_list)
    random.shuffle(questions_shuffled)
    # Re-number questions for the consistent order
    for q_idx, q in enumerate(questions_shuffled, 1):
        q.id = q_idx

    for i in range(1, num_models + 1):
        # Deepcopy to avoid modifying the base shuffled list
        model_questions = deepcopy(questions_shuffled)
        
        # --- Shuffle options for each question to create unique models ---
        for q in model_questions:
            if q.options and q.correct_answer_index is not None:
                # Store the original correct option before shuffling
                original_correct_option = q.options[q.correct_answer_index]
                
                # Shuffle the options
                random.shuffle(q.options)
                
                # Find the new index of the correct option and update the is_correct flag
                try:
                    new_correct_index = q.options.index(original_correct_option)
                    for opt_idx, option in enumerate(q.options):
                        option.is_correct = (opt_idx == new_correct_index)
                except ValueError:
                    logging.error(f"Could not find the original correct answer for question {q.id} after shuffling. This should not happen.")
                    # Handle error case if necessary, though it's unlikely.

        # Save the questions for this model to a JSON file
        model_exam = PexamExam(questions=model_questions)
        questions_json_path = os.path.join(output_dir, f"exam_model_{i}_questions.json")
        with open(questions_json_path, "w", encoding="utf-8") as f:
            f.write(model_exam.model_dump_json(indent=4))
        logging.info(f"Saved questions for model {i} to: {questions_json_path}")

        answer_sheet_html = _generate_answer_sheet_html(
            model_questions, 
            i, 
            exam_title=exam_title,
            exam_course=exam_course,
            exam_date=exam_date,
            lang=lang, 
            id_length=id_length
        )
        questions_md = _generate_questions_markdown(model_questions)
        questions_html = markdown.markdown(questions_md)
        
        final_html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{exam_title} - Model {i}</title>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,300..800;1,300..800&family=Roboto:ital,wght@0,300;0,400;0,500;0,700&display=swap">
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        {css_content}
    </style>
    <style>
        body {{ font-size: {font_size}; }}
    </style>
</head>
<body>
    {answer_sheet_html}
    <div class="page-container" style="page-break-after: always;"></div>
    <div class="page-container questions-container {column_class}">
        {questions_html}
    </div>
</body>
</html>
"""
        html_filepath = os.path.join(output_dir, f"exam_model_{i}.html")
        pdf_filepath = os.path.join(output_dir, f"exam_model_{i}.pdf")

        with open(html_filepath, "w", encoding="utf-8") as f:
            f.write(final_html_content)
        logging.info(f"Generated HTML for exam model {i}: {html_filepath}")
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                # Use file:// protocol to load the local HTML file
                page.goto(f"file:///{os.path.abspath(html_filepath)}")
                
                # Wait for MathJax to finish rendering.
                # typesetPromise() will not resolve until all math is rendered.
                page.evaluate("MathJax.typesetPromise()")
                
                header_text = f"{exam_title} - {exam_date}" if exam_date else exam_title

                page.pdf(
                    path=pdf_filepath,
                    format='A4',
                    print_background=True,
                    margin={'top': '15mm', 'bottom': '15mm', 'left': '15mm', 'right': '15mm'},
                    display_header_footer=True,
                    header_template=f'<div style="font-family: Open Sans, sans-serif; font-size: 9px; color: #888; width: 100%; text-align: center;">{header_text}</div>',
                    footer_template=f'<div style="font-family: Open Sans, sans-serif; font-size: 9px; color: #888; width: 100%; text-align: center;">Model {i} - Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>'
                )
                browser.close()
            logging.info(f"Successfully generated PDF for model {i}: {pdf_filepath}")

            if generate_references:
                logging.info(f"Generating reference scan for model {i}")
                _generate_reference_scan(pdf_filepath, model_questions, output_dir, i, id_length)

            if generate_fakes > 0 and fakes_per_model:
                if i <= len(fakes_per_model):
                    num_fakes_for_this_model = fakes_per_model[i-1]
                    if num_fakes_for_this_model > 0:
                        logging.info(f"Test mode enabled: Generating {num_fakes_for_this_model} simulated scan(s) for model {i}")
                        for fake_idx in range(1, num_fakes_for_this_model + 1):
                            _generate_simulated_scan(pdf_filepath, model_questions, output_dir, f"{i}_{fake_idx}", id_length)
                else:
                    logging.warning(f"Skipping fake generation for model {i} due to index out of range. This can happen if num_models is inconsistent.")

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

def _find_fiducial_markers_for_sim(image):
    # This is a simplified copy from correct_exams.py for test generation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 97, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_height, page_width = image.shape[:2]
    
    corner_regions = {
        "tl": (0, page_width * 0.2, 0, page_height * 0.2),
        "tr": (page_width * 0.8, page_width, 0, page_height * 0.2),
        "bl": (0, page_width * 0.2, page_height * 0.8, page_height),
        "br": (page_width * 0.8, page_width, page_height * 0.8, page_height),
    }
    
    corner_candidates = {"tl": [], "tr": [], "bl": [], "br": []}
    ref_dim = 8 * (300 / 25.4)
    min_area, max_area = (ref_dim ** 2) * 0.2, (ref_dim ** 2) * 1.5

    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area): continue
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h) if h > 0 else 0
        if not (0.4 < aspect_ratio < 2.5): continue
        hull = cv2.convexHull(c)
        if hull.shape[0] < 3: continue
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area) if hull_area > 0 else 0
        if solidity > 0.6: continue
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        for name, (x_min, x_max, y_min, y_max) in corner_regions.items():
            if (x_min < cx < x_max) and (y_min < cy < y_max):
                corner_candidates[name].append((cx, cy))
                break
    
    final_corners = {}
    page_corners = {"tl": (0, 0), "tr": (page_width, 0), "bl": (0, page_height), "br": (page_width, page_height)}
    
    for name, candidates in corner_candidates.items():
        if not candidates: return None
        px, py = page_corners[name]
        final_corners[name] = min(candidates, key=lambda p: np.sqrt((p[0] - px)**2 + (p[1] - py)**2))

    tl, tr, bl, br = final_corners["tl"], final_corners["tr"], final_corners["bl"], final_corners["br"]
    if not all([tl, tr, bl, br]): return None
    return np.array([tl, tr, br, bl], dtype="float32")


def _generate_reference_scan(original_pdf_path: str, questions: List[PexamQuestion], output_dir: str, model_num: int, id_length: int):
    """
    Takes the first page of a PDF, converts it to an image, and adds the correct answers
    to generate a reference scan.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        logging.error("pdf2image is required for reference generation. Please install it.")
        return

    ref_output_dir = os.path.join(output_dir, "reference_scans")
    os.makedirs(ref_output_dir, exist_ok=True)

    try:
        images = convert_from_path(original_pdf_path, first_page=1, last_page=1, dpi=300)
        if not images: return
        img_cv = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"Failed to convert PDF to image for reference scan: {e}")
        return

    marker_corners = _find_fiducial_markers_for_sim(img_cv)
    if marker_corners is None:
        logging.error("Could not find fiducial markers in the generated PDF for reference scan. Skipping.")
        return

    PX_PER_MM = 10.0
    
    from pexams.correct_exams import _apply_perspective_transform
    warped_sheet = _apply_perspective_transform(img_cv, marker_corners, PX_PER_MM)

    layout_data = layout.get_answer_sheet_layout(len(questions), id_length)
    
    for q in questions:
        q_id = q.id
        correct_idx = q.correct_answer_index
        if q_id in layout_data.answer_boxes and correct_idx is not None and correct_idx in layout_data.answer_boxes[q_id]:
            chosen_option = layout_data.answer_boxes[q_id][correct_idx]
            tl_x = int(chosen_option.top_left[0] * PX_PER_MM)
            tl_y = int(chosen_option.top_left[1] * PX_PER_MM)
            br_x = int(chosen_option.bottom_right[0] * PX_PER_MM)
            br_y = int(chosen_option.bottom_right[1] * PX_PER_MM)
            
            cv2.rectangle(warped_sheet, (tl_x + 2, tl_y + 2), (br_x - 2, br_y - 2), (10, 10, 10), -1)

    output_path = os.path.join(ref_output_dir, f"reference_scan_model_{model_num}.png")
    cv2.imwrite(output_path, warped_sheet)
    logging.info(f"Saved reference scan to {output_path}")


def _generate_simulated_scan(original_pdf_path: str, questions: List[PexamQuestion], output_dir: str, model_num: Union[int, str], id_length: int):
    """
    Takes the first page of a PDF, converts it to an image, finds the fiducial markers,
    applies a perspective transform to get a perfect top-down view, and then adds
    fake answers, name, ID, and signature to simulate a realistic, aligned scan.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        logging.error("pdf2image is required for test mode. Please install it.")
        return

    fake = Faker()
    scan_output_dir = os.path.join(output_dir, "simulated_scans")
    os.makedirs(scan_output_dir, exist_ok=True)

    try:
        images = convert_from_path(original_pdf_path, first_page=1, last_page=1, dpi=300)
        if not images: return
        img_cv = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"Failed to convert PDF to image for simulation: {e}")
        return

    # Find fiducial markers in the scanned image
    marker_corners = _find_fiducial_markers_for_sim(img_cv)
    if marker_corners is None:
        logging.error("Could not find fiducial markers in the generated PDF for simulation. Skipping.")
        return

    # Use the same high-resolution scale as the correction process
    PX_PER_MM = 10.0
    
    # Apply perspective transform to get a perfect, aligned image
    from pexams.correct_exams import _apply_perspective_transform
    warped_sheet = _apply_perspective_transform(img_cv, marker_corners, PX_PER_MM)

    # --- Draw Fake Data onto the warped sheet ---
    layout_data = layout.get_answer_sheet_layout(len(questions), id_length)
    
    # Fake Name
    name_box = layout_data.student_name_box
    name_pos = (int(name_box.top_left[0] * PX_PER_MM) + 10, int(name_box.center[1] * PX_PER_MM) + 15)
    cv2.putText(warped_sheet, fake.name(), name_pos, cv2.FONT_HERSHEY_SIMPLEX, 2, (20, 20, 20), 4)

    # Fake ID
    fake_id = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=len(layout_data.student_id_boxes)))
    for i, box in enumerate(layout_data.student_id_boxes):
        text = fake_id[i]
        font_scale = 1.8
        font_thickness = 4
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        center_x_px = int(box.center[0] * PX_PER_MM)
        center_y_px = int(box.center[1] * PX_PER_MM)
        id_pos = (center_x_px - text_width // 2, center_y_px + text_height // 2)
        cv2.putText(warped_sheet, text, id_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), font_thickness)

    # Fake Signature (scribble)
    sig_box = layout_data.student_signature_box
    center_y = int(sig_box.center[1] * PX_PER_MM)
    start_x = int(sig_box.top_left[0] * PX_PER_MM) + 15
    end_x = int(sig_box.bottom_right[0] * PX_PER_MM) - 15
    for x in range(start_x, end_x, 10):
        y_offset = random.randint(-20, 20)
        cv2.line(warped_sheet, (x, center_y + y_offset), (x+10, center_y - y_offset), (30, 30, 30), 4)

    # Fake Answers
    for q_id, options in layout_data.answer_boxes.items():
        if random.random() > 0.1: # 10% chance to leave blank
            chosen_option = random.choice(list(options.values()))
            tl_x = int(chosen_option.top_left[0] * PX_PER_MM)
            tl_y = int(chosen_option.top_left[1] * PX_PER_MM)
            br_x = int(chosen_option.bottom_right[0] * PX_PER_MM)
            br_y = int(chosen_option.bottom_right[1] * PX_PER_MM)
            
            points = np.array([
                [tl_x + random.randint(1, 4), tl_y + random.randint(1, 4)],
                [br_x - random.randint(1, 4), tl_y + random.randint(1, 4)],
                [br_x - random.randint(1, 4), br_y - random.randint(1, 4)],
                [tl_x + random.randint(1, 4), br_y - random.randint(1, 4)]
            ])
            cv2.fillPoly(warped_sheet, [points], (10, 10, 10))

    # Save the final simulated scan
    output_path = os.path.join(scan_output_dir, f"simulated_scan_model_{model_num}.png")
    cv2.imwrite(output_path, warped_sheet)
    logging.info(f"Saved simulated scan to {output_path}")
