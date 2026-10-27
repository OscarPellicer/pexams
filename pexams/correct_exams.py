"""Correction of scanned multiple-choice answer sheets: header reading, mark detection and
scoring, student identification, annotated sheets and ``correction_results.csv``."""
import csv
import json
import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import cv2

from pexams import layout
from pexams.header_reading import HeaderReader
from pexams.open_responses import extract_open_answer_responses
from pexams.scan_reading import PX_PER_MM, apply_perspective_transform, find_fiducial_markers, load_scan_pages
from pexams.schemas import PexamExam, PexamOption, PexamQuestion
from pexams.student_matching import match_scanned_students, normalize_student_id

RAW_SHEETS_DIR = ".raw"  # inside scanned_pages: the aligned answer sheets without annotations

# Fill of an answer box measured inside a margin, so the printed border does not count. In a
# first partial exam (17 sheets, 1700 boxes): unmarked boxes <= 0.10 inside (0.29 on the whole
# box, for a border redrawn after correction fluid), marked boxes >= 0.53, light pencil included.
BOX_INSET_MM = 0.5
FILL_THRESHOLD = 0.3

RESULT_COLUMNS = ["page", "student_id", "student_name", "ocr_student_id", "ocr_student_name", "model_id",
                  "score", "total_questions", "total_points", "multiple_marks"]

VOID_GREY = (130, 130, 130)


def draw_void_marks(image, layout_data, voided: Sequence, label: str, px_per_mm: float = PX_PER_MM) -> None:
    """Draw, in place, a grey line labelled ``label`` across the answer row of each voided question."""
    to_px = lambda mm: int(round(mm * px_per_mm))  # noqa: E731
    for q_num in voided:
        options = layout_data.answer_boxes.get(q_num) or layout_data.answer_boxes.get(str(q_num))
        if not options:
            continue
        x0 = min(b.top_left[0] for b in options.values())
        x1 = max(b.bottom_right[0] for b in options.values())
        y_mid = (min(b.top_left[1] for b in options.values()) + max(b.bottom_right[1] for b in options.values())) / 2
        cv2.line(image, (to_px(x0 - 1), to_px(y_mid)), (to_px(x1 + 1), to_px(y_mid)), VOID_GREY, 3)
        cv2.putText(image, label, (to_px(x1 + 1.5), to_px(y_mid + 1.2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    VOID_GREY, 1, cv2.LINE_AA)


def annotate_answer_sheet(warped_sheet, layout_data, model_id: str = "", student_name: str = "", id_text: str = "",
                          answers: Optional[Dict] = None, solutions: Optional[Dict] = None,
                          voided: Sequence = (), void_label: str = "VOID", px_per_mm: float = PX_PER_MM):
    """Draw the correction on an aligned answer sheet: the header boxes with what was read
    (model, ID, name) in blue, the correct option of each question in green, a wrong answer in
    red, and a grey line with ``void_label`` across voided questions. Used by the correction and
    again by the analysis, which redraws from the unannotated sheet after manual fixes or voids."""
    image = warped_sheet.copy()
    green, red, blue = (0, 255, 0), (0, 0, 255), (255, 0, 0)
    thickness = 3
    to_px = lambda mm: int(round(mm * px_per_mm))  # noqa: E731

    def box(coords, color):
        tl = (to_px(coords.top_left[0]), to_px(coords.top_left[1]))
        br = (to_px(coords.bottom_right[0]), to_px(coords.bottom_right[1]))
        cv2.rectangle(image, tl, br, color, thickness)
        return tl, br

    def text_below(text, tl, br, centered, padding):
        if not text:
            return
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        x = tl[0] + (br[0] - tl[0] - text_w) // 2 if centered else tl[0]
        cv2.putText(image, text, (x, br[1] + text_h + padding), cv2.FONT_HERSHEY_SIMPLEX, 1.5, blue, 3)

    text_below(str(model_id or ""), *box(layout_data.model_id_box, blue), centered=True, padding=5)
    text_below(student_name, *box(layout_data.student_name_box, blue), centered=False, padding=10)
    box(layout_data.student_signature_box, blue)
    text_below(id_text, *box(layout_data.student_id_box, blue), centered=True, padding=5)

    answers = answers or {}
    for q_num, correct_idx in (solutions or {}).items():
        options = layout_data.answer_boxes.get(q_num, {})
        if correct_idx in options:
            box(options[correct_idx], green)
        student_idx = answers.get(q_num, -1)
        if student_idx != -1 and student_idx != correct_idx and student_idx in options:
            box(options[student_idx], red)

    draw_void_marks(image, layout_data, voided, void_label, px_per_mm)
    return image


def _publish_sheet(result: dict, scanned_pages_dir: str, name: str) -> None:
    """Copy a page's annotated sheet to scanned_pages/<name>.png and its unannotated sheet to
    scanned_pages/.raw/<name>.png (from which the analysis redraws after fixes or voids)."""
    shutil.copyfile(result["staged_scan_path"], os.path.join(scanned_pages_dir, f"{name}.png"))
    raw = result.get("staged_raw_path")
    if raw and os.path.exists(raw):
        os.makedirs(os.path.join(scanned_pages_dir, RAW_SHEETS_DIR), exist_ok=True)
        shutil.copyfile(raw, os.path.join(scanned_pages_dir, RAW_SHEETS_DIR, f"{name}.png"))


def _analyze_and_score(warped_image, solutions: Dict[int, int], px_per_mm: float, questions: List[PexamQuestion]):
    """Analyzes the warped sheet to find marked answers and scores them."""
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    detected_answers = {}
    multiple_marks = {}
    score = 0

    questions = [q for q in questions if q.is_multiple_choice]
    layout_data = layout.get_answer_sheet_layout(questions)

    for q in questions:
        q_num = q.id
        if q_num not in layout_data.answer_boxes:
            continue

        marked_options = []
        for opt_idx, coords in sorted(layout_data.answer_boxes[q_num].items()):
            tl_x, tl_y = coords.top_left
            br_x, br_y = coords.bottom_right
            # Measure only the inside of the box: its printed border alone (or a border redrawn
            # by hand after correction fluid) fills ~20-30% of the whole box, too close to the
            # threshold, while the inside of an unmarked box stays near 0.
            inset = BOX_INSET_MM
            x_px, y_px = int((tl_x + inset) * px_per_mm), int((tl_y + inset) * px_per_mm)
            x2_px, y2_px = int((br_x - inset) * px_per_mm), int((br_y - inset) * px_per_mm)

            roi = thresh[y_px:y2_px, x_px:x2_px]
            if roi.size == 0:
                continue
            if cv2.countNonZero(roi) > (roi.size * FILL_THRESHOLD):
                marked_options.append(opt_idx)

        # Two or more filled boxes (e.g. one crossed out instead of erased) are not resolved by
        # picking the darkest, which favours the crossed-out one: the question is left blank
        # here and resolved afterwards (_resolve_multiple_marks) or reviewed by the teacher.
        bubbled_option = marked_options[0] if len(marked_options) == 1 else -1
        if len(marked_options) > 1:
            multiple_marks[q_num] = marked_options
        detected_answers[q_num] = bubbled_option

        if q_num in solutions and solutions[q_num] == bubbled_option:
            score += float(q.points)

    total_points = sum(float(q.points) for q in questions if q.id in solutions)
    return {"score": score, "total_questions": len(solutions), "total_points": total_points, "answers": detected_answers,
            "multiple_marks": multiple_marks}


_MULTIPLE_MARKS_PROMPT = (
    "This is one row of a scanned multiple-choice answer sheet: question {question}, with the option "
    "boxes {letters} from left to right. More than one box looks filled. A student cancels a filled box "
    "by crossing it out (an X or lines drawn over it) or with correction fluid. Which option is the "
    "student's final answer? "
    'Return only JSON: {{"answer": "<one letter, or empty if it cannot be decided>", "reason": "<short>"}}'
)


def _resolve_multiple_marks(warped_image, layout_data, q_num, options: List[int], llm, px_per_mm,
                            page_number=None) -> Optional[int]:
    """Ask the vision LLM which of several filled boxes is the final answer; None if unclear."""
    from pexams.llm import parse_json_response

    boxes = layout_data.answer_boxes.get(q_num) or {}
    if not boxes or llm is None:
        return None
    x0 = min(b.top_left[0] for b in boxes.values()) - 8   # include the question number
    x1 = max(b.bottom_right[0] for b in boxes.values()) + 4
    y0 = min(b.top_left[1] for b in boxes.values()) - 3   # crossing strokes often overshoot the box
    y1 = max(b.bottom_right[1] for b in boxes.values()) + 3
    roi = warped_image[max(0, int(y0 * px_per_mm)):int(y1 * px_per_mm), max(0, int(x0 * px_per_mm)):int(x1 * px_per_mm)]
    ok, png = cv2.imencode(".png", roi)
    if roi.size == 0 or not ok:
        return None
    letters = [chr(ord("A") + i) for i in sorted(boxes)]
    try:
        text, _ = llm.complete(
            _MULTIPLE_MARKS_PROMPT.format(question=q_num, letters=", ".join(letters)),
            images=[(png.tobytes(), "image/png")], max_tokens=2000, json_output=True,
            purpose="resolve_multiple_marks", context={"page": page_number, "question": q_num},
        )
        answer = str(parse_json_response(text).get("answer", "") or "").strip().upper()[:1]
    except Exception as e:
        logging.error("LLM could not resolve the double mark of question %s on page %s: %s", q_num, page_number, e)
        return None
    index = ord(answer) - ord("A") if answer else -1
    return index if index in options else None


def _resolve_page_multiple_marks(page_result: dict, warped_sheet, sheet_layout, questions: List[PexamQuestion],
                                 solutions: Dict[int, int], llm, page_number: int) -> None:
    """Rows with several filled boxes: the LLM decides which is the final answer (e.g. the other
    one is crossed out); unresolved ones stay blank. Replaces ``page_result["multiple_marks"]`` by
    a note for review, e.g. "5:B+C->B" (resolved) or "7:A+D->?" (left blank)."""
    notes = []
    for q_num, options in page_result.get("multiple_marks", {}).items():
        resolved = _resolve_multiple_marks(warped_sheet, sheet_layout, q_num, options, llm, PX_PER_MM, page_number)
        marked = "+".join(chr(ord("A") + i) for i in options)
        if resolved is not None:
            page_result["answers"][q_num] = resolved
            if solutions.get(q_num) == resolved:
                question = next((q for q in questions if q.id == q_num), None)
                page_result["score"] += float(question.points) if question else 1.0
            notes.append(f"{q_num}:{marked}->{chr(ord('A') + resolved)}")
        else:
            notes.append(f"{q_num}:{marked}->?")
    page_result["multiple_marks"] = "; ".join(notes)
    if notes:
        logging.warning("Page %s: several boxes marked in a question (%s). Unresolved ones (->?) count as blank; "
                        "review them in correction_results.csv (column multiple_marks).", page_number, "; ".join(notes))


def _exam_pages_per_model(questions_dir: Optional[str]) -> Dict[str, int]:
    """Printed pages of each model's exam (answer sheet included), from ``open_answer_areas.json``.
    Question pages carry fiducial markers too (needed to rectify open answers), so after an
    answer sheet the remaining pages of that exam must not be read as new answer sheets."""
    pages = {}
    areas_path = Path(questions_dir) / "open_answer_areas.json" if questions_dir else None
    if areas_path and areas_path.exists():
        for area in json.loads(areas_path.read_text(encoding="utf-8")):
            model_key = str(area.get("model_id"))
            pages[model_key] = max(pages.get(model_key, 1), int(area.get("page_index", 1)))
    return pages


def _load_model_questions(questions_dir: str, model_id: str) -> Optional[List[PexamQuestion]]:
    """The multiple-choice questions of a model, from ``exam_model_<id>_questions.json``."""
    questions_path = Path(questions_dir) / f"exam_model_{model_id}_questions.json"
    if not questions_path.exists():
        logging.warning(f"Questions file not found for model {model_id} at {questions_path}. Skipping page.")
        return None
    try:
        exam_model = PexamExam.model_validate_json(questions_path.read_text(encoding="utf-8"))
        return [q for q in exam_model.questions if q.is_multiple_choice]
    except Exception as e:
        logging.error(f"Failed to parse questions file for model {model_id}: {e}. Skipping page.")
        return None


def _save_staged_sheets(page_result: dict, warped_sheet, sheet_layout, solutions: Dict[int, int],
                        staged_pages_dir: str, page_number: int) -> None:
    """Save the annotated sheet and the sheet without marks (to redraw it later) under the scan id."""
    scanned_id = page_result["scanned_id"]
    annotated_image = annotate_answer_sheet(
        warped_sheet, sheet_layout, model_id=page_result["model_id"], student_name=page_result["ocr_student_name"],
        id_text=page_result["ocr_student_id"] or scanned_id, answers=page_result["answers"], solutions=solutions)
    png_path = os.path.join(staged_pages_dir, f"{scanned_id}.png")
    page_result["staged_scan_path"] = png_path
    cv2.imwrite(png_path, annotated_image)
    raw_dir = os.path.join(staged_pages_dir, RAW_SHEETS_DIR)
    os.makedirs(raw_dir, exist_ok=True)
    page_result["staged_raw_path"] = os.path.join(raw_dir, f"{scanned_id}.png")
    cv2.imwrite(page_result["staged_raw_path"], warped_sheet)
    logging.info(f"Saved annotated scan for page {page_number} to {png_path}")


def _correct_answer_sheet(frame, page_number: int, source_path: str, reader: HeaderReader,
                          solutions_per_model: Dict[str, Dict[int, int]], questions_dir: str,
                          simulated_scan_manifest: Dict[str, dict], staged_pages_dir: str,
                          debug_dir: str, header_layout) -> Optional[dict]:
    """Read and score one scanned page as an answer sheet. Returns its result (with
    ``model_id``), or None if it is not a readable answer sheet."""
    marker_corners = find_fiducial_markers(frame, debug_dir, page_number)
    if marker_corners is None:
        logging.warning(f"Could not find 4 fiducial markers on page {page_number}. Skipping page.")
        return None
    warped_sheet = apply_perspective_transform(frame, marker_corners, PX_PER_MM)

    model_id, header = reader.read_model_id(warped_sheet, header_layout, page_number)
    if not model_id or model_id not in solutions_per_model:
        logging.warning(f"Could not detect a valid model ID for page {page_number} (detected: '{model_id}'). Skipping page.")
        return None
    logging.info(f"Page {page_number}: Detected Model ID '{model_id}'.")
    solutions = solutions_per_model[model_id]

    questions = _load_model_questions(questions_dir, model_id)
    if questions is None:
        # Still an answer sheet: its exam's other pages are skipped.
        return {"model_id": model_id, "skip": True}
    sheet_layout = layout.get_answer_sheet_layout(questions)
    ocr_student_id, student_name = reader.read_student(warped_sheet, sheet_layout, header, page_number)

    scanned_id = f"scan_{uuid.uuid4().hex[:10]}"
    if "?" in ocr_student_id or not ocr_student_id:
        logging.warning("Could not reliably OCR student ID for page %s. Keeping only staged scan id: %s", page_number, scanned_id)

    forced_student_id = ""
    if source_path:
        simulated_entry = simulated_scan_manifest.get(os.path.basename(source_path), {})
        forced_student_id = str(simulated_entry.get("student_id", "")).strip().upper()
        if forced_student_id:
            logging.info("Page %s: using predefined simulated-scan student ID '%s' while preserving OCR result '%s'.",
                         page_number, forced_student_id, ocr_student_id)

    page_result = _analyze_and_score(warped_sheet, solutions, PX_PER_MM, questions)
    _resolve_page_multiple_marks(page_result, warped_sheet, sheet_layout, questions, solutions, reader.llm, page_number)
    page_result.update({
        "page": page_number,
        "scanned_id": scanned_id,
        "student_id": forced_student_id or scanned_id,
        "student_name": "",
        "ocr_student_id": ocr_student_id,
        "ocr_student_name": student_name,
        "model_id": model_id,
    })
    _save_staged_sheets(page_result, warped_sheet, sheet_layout, solutions, staged_pages_dir, page_number)
    return page_result


def _reset_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _identify_with_roster(all_results: List[dict], output_dir: str, scanned_pages_dir: str, roster_csv: str,
                          roster_id_column: Optional[str], roster_name_column: Optional[str],
                          name_match_threshold: float, roster_encoding: str, roster_sep: str) -> bool:
    """Match each exam to the class list (``student_matches.csv``) and publish its sheets under
    the roster ID. False if the matching cannot be done or some exam is unmatched."""
    if not roster_id_column or not roster_name_column:
        logging.error("--input-csv student matching requires both --id-column and --name-column.")
        return False
    matches_csv = os.path.join(output_dir, "student_matches.csv")
    try:
        matches = match_scanned_students(
            all_results, roster_csv=roster_csv, id_column=roster_id_column, name_column=roster_name_column,
            output_csv=matches_csv, fuzzy_threshold=name_match_threshold, encoding=roster_encoding, sep=roster_sep,
        )
    except Exception as e:
        logging.error("Failed to match scanned students against roster: %s", e)
        return False

    matches_by_scan = {match.scanned_id: match for match in matches}
    if any(not match.matched for match in matches):
        logging.error("Fix %s before running analysis or mark export.", matches_csv)
        return False

    _reset_dir(scanned_pages_dir)
    for result in all_results:
        match = matches_by_scan.get(str(result.get("scanned_id", "")))
        if not match:
            continue
        result["student_id"] = match.roster_student_id
        result["student_name"] = match.roster_student_name
        source_scan = result.get("staged_scan_path")
        if source_scan and os.path.exists(source_scan):
            safe_student_id = re.sub(r"[^A-Za-z0-9_-]+", "_", match.roster_student_id).strip("_") or match.scanned_id
            _publish_sheet(result, scanned_pages_dir, safe_student_id)
    return True


def _identify_without_roster(all_results: List[dict], scanned_pages_dir: str) -> None:
    """Without a roster, identify each exam by the ID read from it (when readable and unique),
    keep the name read from it, and publish its sheets under that ID."""
    read_ids = [normalize_student_id(r.get("ocr_student_id", "")) for r in all_results]
    for result, read_id in zip(all_results, read_ids):
        result["student_name"] = result.get("ocr_student_name", "")
        if len(read_id) >= 3 and "?" not in str(result.get("ocr_student_id", "")) and read_ids.count(read_id) == 1:
            result["student_id"] = read_id
    _reset_dir(scanned_pages_dir)
    for result in all_results:
        source_scan = result.get("staged_scan_path")
        if source_scan and os.path.exists(source_scan):
            _publish_sheet(result, scanned_pages_dir, result['student_id'])


def _write_results_csv(all_results: List[dict], q_nums: List[int], results_csv_path: str) -> None:
    headers = RESULT_COLUMNS + [f"answer_{q}" for q in q_nums]
    with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for result in all_results:
            row = {column: result.get(column, "N/A") for column in RESULT_COLUMNS}
            for column in ("ocr_student_id", "ocr_student_name", "multiple_marks"):
                row[column] = result.get(column, "")
            detected_answers = result.get("answers", {})
            for q_num in q_nums:
                answer = detected_answers.get(q_num, -1)
                row[f"answer_{q_num}"] = chr(ord('A') + answer) if answer != -1 else "NA"
            writer.writerow(row)
    logging.info(f"Correction complete. Results saved to: {results_csv_path}")


def correct_exams(
    input_path: Union[str, Sequence[str]],
    solutions_per_model: Dict[str, Dict[int, int]],
    output_dir: str,
    questions_dir: str,
    roster_csv: Optional[str] = None,
    roster_id_column: Optional[str] = None,
    roster_name_column: Optional[str] = None,
    roster_encoding: str = "utf-8",
    roster_sep: str = ",",
    name_match_threshold: float = 70.0,
    id_reader: str = "llm",
    id_reader_model: str = "claude-opus-5-5",
    id_reader_effort: Optional[str] = "low",
    api_key_env: Optional[str] = None,
) -> bool:
    """Correct scanned exams. ``input_path`` is a PDF or a directory, or a list of them read in
    order as a single batch (see :func:`pexams.scan_reading.load_scan_pages`). ``id_reader="llm"``
    (default) reads the model, student-ID and name boxes with a vision LLM (``id_reader_model``);
    ``"trocr"`` uses the local TrOCR model, which is also the fallback when no API key is available
    or a call fails. Every LLM call is logged to ``<output_dir>/llm_calls.jsonl``."""
    # Several inputs (PDFs and/or directories) are read in the order given, as one batch.
    input_paths = [str(input_path)] if isinstance(input_path, (str, os.PathLike)) else [str(p) for p in input_path]
    logging.info(f"Starting pexams correction for: {', '.join(input_paths)}")
    os.makedirs(output_dir, exist_ok=True)
    scanned_pages_dir = os.path.join(output_dir, "scanned_pages")
    staged_pages_dir = os.path.join(output_dir, "_scanned_pages_unmatched")
    os.makedirs(staged_pages_dir, exist_ok=True)
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    try:
        images_to_process, image_sources, simulated_scan_manifest = load_scan_pages(input_paths)
    except Exception as e:
        logging.error(f"Failed to load the scans: {e}")
        return False
    logging.info("Loaded %d scanned page(s) from %d input(s).", len(images_to_process), len(input_paths))
    if not images_to_process:
        logging.warning("No images found to process.")
        return False

    reader = HeaderReader(id_reader, id_reader_model, id_reader_effort, api_key_env, output_dir,
                          inputs=[*input_paths, roster_csv or ""], px_per_mm=PX_PER_MM, debug_dir=debug_dir)
    if not reader.ready:
        return False

    # The header boxes do not depend on the questions: any layout locates them.
    dummy_question = PexamQuestion(id=1, text='d', options=[PexamOption(text=f'{i}', is_correct=(i == 0)) for i in range(5)])
    header_layout = layout.get_answer_sheet_layout(questions=[dummy_question])
    exam_pages_per_model = _exam_pages_per_model(questions_dir)

    all_results = []
    skip_until_page = 0
    for i, frame in enumerate(images_to_process):
        page_number = i + 1
        if page_number <= skip_until_page:
            logging.info(f"Page {page_number} belongs to the previous exam (response pages). Skipping answer-sheet detection.")
            continue
        logging.info(f"Processing page {page_number}...")
        result = _correct_answer_sheet(
            frame, page_number, image_sources[i] if i < len(image_sources) else "", reader, solutions_per_model,
            questions_dir, simulated_scan_manifest, staged_pages_dir, debug_dir, header_layout)
        if result is None:
            continue
        skip_until_page = page_number + exam_pages_per_model.get(str(result["model_id"]), 1) - 1
        if not result.get("skip"):
            all_results.append(result)

    if roster_csv:
        if not _identify_with_roster(all_results, output_dir, scanned_pages_dir, roster_csv, roster_id_column,
                                     roster_name_column, name_match_threshold, roster_encoding, roster_sep):
            return False
    else:
        _identify_without_roster(all_results, scanned_pages_dir)

    if not all_results:
        logging.warning("No pages were processed successfully.")
        return True
    # The CSV has one answer column per question of the first model (all models have the same questions).
    q_nums = sorted(solutions_per_model[next(iter(solutions_per_model))].keys())
    try:
        _write_results_csv(all_results, q_nums, os.path.join(output_dir, "correction_results.csv"))
    except IOError as e:
        logging.error(f"Failed to write results to CSV: {e}")
        return False

    extract_open_answer_responses(images_to_process=images_to_process, page_results=all_results,
                                  questions_dir=questions_dir, output_dir=output_dir, debug_dir=debug_dir)
    return True
