"""Test images made from a generated exam: simulated scans of answer sheets filled with fake
students and answers, and reference sheets with the correct answers marked."""
import json
import logging
import os
import random
from typing import List, Optional, Union

import cv2
import numpy as np

from pexams import layout
from pexams.scan_reading import PX_PER_MM, apply_perspective_transform, find_fiducial_markers
from pexams.schemas import PexamQuestion

SCAN_DPI = 300


def _warped_answer_sheet(pdf_path: str, purpose: str) -> Optional[np.ndarray]:
    """The first page (answer sheet) of an exam PDF, rendered and warped like a corrected scan."""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        logging.error(f"pdf2image is required for {purpose}. Please install it.")
        return None
    try:
        images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=SCAN_DPI)
        if not images:
            return None
        img_cv = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"Failed to convert PDF to image for {purpose}: {e}")
        return None
    marker_corners = find_fiducial_markers(img_cv)
    if marker_corners is None:
        logging.error(f"Could not find fiducial markers in the generated PDF for {purpose}. Skipping.")
        return None
    return apply_perspective_transform(img_cv, marker_corners, PX_PER_MM)


def _box_px(box):
    """Pixel corners (tl_x, tl_y, br_x, br_y) of a layout box on a warped sheet."""
    return (int(box.top_left[0] * PX_PER_MM), int(box.top_left[1] * PX_PER_MM),
            int(box.bottom_right[0] * PX_PER_MM), int(box.bottom_right[1] * PX_PER_MM))


def generate_reference_scan(original_pdf_path: str, questions: List[PexamQuestion], output_dir: str, model_num: int):
    """Save ``reference_scans/reference_scan_model_<n>.png``: the answer sheet with the correct
    option of every question filled in."""
    ref_output_dir = os.path.join(output_dir, "reference_scans")
    os.makedirs(ref_output_dir, exist_ok=True)
    warped_sheet = _warped_answer_sheet(original_pdf_path, "reference scan")
    if warped_sheet is None:
        return

    questions = [q for q in questions if q.is_multiple_choice]
    layout_data = layout.get_answer_sheet_layout(questions)
    for q in questions:
        correct_idx = q.correct_answer_index
        if q.id in layout_data.answer_boxes and correct_idx is not None and correct_idx in layout_data.answer_boxes[q.id]:
            tl_x, tl_y, br_x, br_y = _box_px(layout_data.answer_boxes[q.id][correct_idx])
            cv2.rectangle(warped_sheet, (tl_x + 2, tl_y + 2), (br_x - 2, br_y - 2), (10, 10, 10), -1)

    output_path = os.path.join(ref_output_dir, f"reference_scan_model_{model_num}.png")
    cv2.imwrite(output_path, warped_sheet)
    logging.info(f"Saved reference scan to {output_path}")


def _draw_fake_student(warped_sheet, layout_data, name: str, student_id: str) -> None:
    """Write a name, an ID and a scribbled signature in the header boxes."""
    name_box = layout_data.student_name_box
    name_pos = (int(name_box.top_left[0] * PX_PER_MM) + 10, int(name_box.center[1] * PX_PER_MM) + 15)
    cv2.putText(warped_sheet, name, name_pos, cv2.FONT_HERSHEY_SIMPLEX, 2, (20, 20, 20), 4)

    id_box = layout_data.student_id_box
    font_scale, font_thickness = 1.8, 4
    (text_width, text_height), _ = cv2.getTextSize(student_id, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    id_pos = (int(id_box.center[0] * PX_PER_MM) - text_width // 2, int(id_box.center[1] * PX_PER_MM) + text_height // 2)
    cv2.putText(warped_sheet, student_id, id_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), font_thickness)

    sig_box = layout_data.student_signature_box
    center_y = int(sig_box.center[1] * PX_PER_MM)
    start_x = int(sig_box.top_left[0] * PX_PER_MM) + 15
    end_x = int(sig_box.bottom_right[0] * PX_PER_MM) - 15
    for x in range(start_x, end_x, 10):
        y_offset = random.randint(-20, 20)
        cv2.line(warped_sheet, (x, center_y + y_offset), (x + 10, center_y - y_offset), (30, 30, 30), 4)


def _draw_fake_answers(warped_sheet, layout_data) -> None:
    """Fill a random option of each question (leaving ~10% blank), with slightly ragged edges."""
    for options in layout_data.answer_boxes.values():
        if random.random() > 0.1:
            tl_x, tl_y, br_x, br_y = _box_px(random.choice(list(options.values())))
            points = np.array([
                [tl_x + random.randint(1, 4), tl_y + random.randint(1, 4)],
                [br_x - random.randint(1, 4), tl_y + random.randint(1, 4)],
                [br_x - random.randint(1, 4), br_y - random.randint(1, 4)],
                [tl_x + random.randint(1, 4), br_y - random.randint(1, 4)],
            ])
            cv2.fillPoly(warped_sheet, [points], (10, 10, 10))


def _add_to_manifest(manifest_path: str, filename: str, student_id: str, model_num) -> None:
    """Record the fake student ID of a simulated scan, which the corrector uses instead of OCR."""
    try:
        manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        manifest[filename] = {"student_id": student_id, "model_num": str(model_num)}
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
    except Exception as e:
        logging.warning(f"Failed to update simulated scan manifest for {filename}: {e}")


def generate_simulated_scan(original_pdf_path: str, questions: List[PexamQuestion], output_dir: str,
                            model_num: Union[int, str]):
    """Save ``simulated_scans/simulated_scan_model_<n>.png``: the answer sheet, aligned as a
    corrected scan, with a fake name, ID, signature and answers."""
    from faker import Faker

    fake = Faker()
    scan_output_dir = os.path.join(output_dir, "simulated_scans")
    os.makedirs(scan_output_dir, exist_ok=True)
    warped_sheet = _warped_answer_sheet(original_pdf_path, "simulation")
    if warped_sheet is None:
        return

    layout_data = layout.get_answer_sheet_layout([q for q in questions if q.is_multiple_choice])
    name = fake.name()
    fake_id = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=8))
    _draw_fake_student(warped_sheet, layout_data, name, fake_id)
    _draw_fake_answers(warped_sheet, layout_data)

    output_filename = f"simulated_scan_model_{model_num}.png"
    output_path = os.path.join(scan_output_dir, output_filename)
    cv2.imwrite(output_path, warped_sheet)
    _add_to_manifest(os.path.join(scan_output_dir, "simulated_scan_manifest.json"), output_filename, fake_id, model_num)
    logging.info(f"Saved simulated scan to {output_path}")
