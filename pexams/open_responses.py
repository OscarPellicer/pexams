"""Crops of the open-answer boxes of scanned exams, with a check that each crop lines up with
its printed box and a measure of the handwriting outside it."""
import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from pdf2image import convert_from_path

from pexams import layout
from pexams.scan_reading import PX_PER_MM, apply_perspective_transform, find_fiducial_markers

PAGE_MARGIN_MM = 15.0          # printed page margin (pexams.css @page): the printable area starts here
CONTEXT_MARGIN_MM = 10.0       # page margin kept around the printable area when looking for overflow
OUTSIDE_BELOW_MM = 30.0        # handwriting up to this far below a box is taken as part of its answer
INK_OUTSIDE_MIN_MM2 = 6.0      # more handwriting than this outside the box flags the answer (a short word)


def border_found(warped_page, x, y, w, h, px_per_mm, tolerance_mm=3.0, min_dark_fraction=0.5) -> bool:
    """True if the answer box's top and bottom borders are where the layout says they are.
    A printed box border is a long dark line across most of the width; handwriting is not."""
    gray = cv2.cvtColor(warped_page, cv2.COLOR_BGR2GRAY) if warped_page.ndim == 3 else warped_page
    x0, x1 = x + int(0.1 * w), x + int(0.9 * w)
    if x1 <= x0:
        return True
    tol = max(1, int(tolerance_mm * px_per_mm))
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    for edge in (y, y + h):
        rows = threshold[max(0, edge - tol):min(gray.shape[0], edge + tol + 1), x0:x1]
        if rows.size == 0 or (rows > 0).mean(axis=1).max() < min_dark_fraction:
            return False
    return True


def render_template_page(pdf_path: str, page_number: int, px_per_mm: float, margin_mm: float) -> Optional[np.ndarray]:
    """Grayscale render of a blank exam page, aligned with ``apply_perspective_transform(...,
    margin_mm)`` of its scan: what is on the scan but not here was written by the student."""
    try:
        pages = convert_from_path(pdf_path, dpi=px_per_mm * 25.4, first_page=page_number,
                                  last_page=page_number, grayscale=True)
    except Exception as e:
        logging.warning("Cannot render %s page %s to find writing outside the boxes: %s", pdf_path, page_number, e)
        return None
    if not pages:
        return None
    page = np.array(pages[0].convert("L"))
    offset = int(round((PAGE_MARGIN_MM - margin_mm) * px_per_mm))
    width = int((layout.PRINTABLE_WIDTH + 2 * margin_mm) * px_per_mm)
    height = int((layout.PRINTABLE_HEIGHT + 2 * margin_mm) * px_per_mm)
    template = np.full((height, width), 255, dtype=np.uint8)
    crop = page[offset:offset + height, offset:offset + width]
    template[:crop.shape[0], :crop.shape[1]] = crop
    return template


def handwriting_mask(warped_page: np.ndarray, template_gray: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Ink on the scan that is not printed on the blank page (tolerating ~1.5 mm of misalignment),
    without the scanner's specks."""
    gray = cv2.cvtColor(warped_page, cv2.COLOR_BGR2GRAY) if warped_page.ndim == 3 else warped_page
    height, width = min(gray.shape[0], template_gray.shape[0]), min(gray.shape[1], template_gray.shape[1])
    scan_ink = (gray[:height, :width] < 190).astype(np.uint8)
    tolerance = max(3, int(1.5 * px_per_mm) | 1)
    printed = cv2.dilate((template_gray[:height, :width] < 200).astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance, tolerance)))
    hand = scan_ink & (1 - printed)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hand, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= max(4, int(0.3 * px_per_mm ** 2))
    keep[0] = False
    return keep[labels]


def ink_outside_box(hand: np.ndarray, box_px: tuple, below_bottom_px: int, margin_px: int, px_per_mm: float) -> float:
    """mm² of handwriting in the strip below the box (down to ``below_bottom_px``) and in the
    side margins next to it."""
    x, y, w, h = box_px
    gap = int(1.5 * px_per_mm)  # skip the printed border and its blur
    right_edge = hand.shape[1]
    regions = [
        hand[y + h + gap:below_bottom_px, 0:right_edge],          # below, full width
        hand[y:y + h, 0:max(0, x - gap)],                          # left margin
        hand[y:y + h, min(right_edge, x + w + gap):right_edge],    # right margin
    ]
    return float(sum(int(r.sum()) for r in regions if r.size)) / (px_per_mm ** 2)


def _safe_name(text: str, fallback: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or fallback


class _ScannedPages:
    """Scanned pages warped with a margin around the printable area (to see writing outside
    it), and their handwriting masks; both computed once per page."""

    def __init__(self, images: List[np.ndarray], questions_dir: str, debug_dir: str):
        self.images, self.questions_dir, self.debug_dir = images, questions_dir, debug_dir
        self._warped, self._hand, self._templates = {}, {}, {}

    def warped(self, index: int) -> Optional[np.ndarray]:
        if index not in self._warped:
            frame = self.images[index]
            corners = find_fiducial_markers(frame, self.debug_dir, index + 1)
            if corners is None:
                logging.warning("Could not find fiducials on open-answer page %s. Skipping crops from that page.",
                                index + 1)
                self._warped[index] = None
            else:
                self._warped[index] = apply_perspective_transform(frame, corners, PX_PER_MM, margin_mm=CONTEXT_MARGIN_MM)
        return self._warped[index]

    def handwriting(self, index: int, model_id: str, page_number: int) -> Optional[np.ndarray]:
        """Handwriting mask of a scanned page (None without the blank exam PDF)."""
        if index not in self._hand:
            key = (model_id, page_number)
            if key not in self._templates:
                pdf_path = Path(self.questions_dir) / f"exam_model_{model_id}.pdf"
                self._templates[key] = (render_template_page(str(pdf_path), page_number, PX_PER_MM, CONTEXT_MARGIN_MM)
                                        if pdf_path.exists() else None)
            template = self._templates[key]
            self._hand[index] = None if template is None else handwriting_mask(self._warped[index], template, PX_PER_MM)
        return self._hand[index]


def _crop_area(pages: _ScannedPages, open_areas: List[dict], area: dict, result: dict, answer_page_index: int,
               target_page_index: int, output_dir: str, responses_dir: str) -> Optional[dict]:
    """Save the crop of one open-answer area of one exam (and, when the student wrote outside
    the box, a crop with its surroundings) and return its row of the index."""
    student_id = str(result.get("student_id", "unknown"))
    model_id = str(result.get("model_id", ""))
    warped_page = pages.warped(target_page_index)
    if warped_page is None:
        return None

    margin_px = int(round(CONTEXT_MARGIN_MM * PX_PER_MM))
    x = max(0, int(float(area["x_mm"]) * PX_PER_MM)) + margin_px
    y = max(0, int(float(area["y_mm"]) * PX_PER_MM)) + margin_px
    w = max(1, int(float(area["width_mm"]) * PX_PER_MM))
    h = max(1, int(float(area["height_mm"]) * PX_PER_MM))
    crop = warped_page[y:y + h, x:x + w]
    layout_ok = border_found(warped_page, x, y, w, h, PX_PER_MM)
    if crop.size == 0:
        logging.warning("Empty open-answer crop for student %s, question %s.", student_id, area.get("question_id"))
        return None

    crop_filename = (f"{_safe_name(student_id, 'unknown')}_model_{model_id}_q_"
                     f"{_safe_name(str(area.get('question_id', 'question')), 'question')}.png")
    crop_path = os.path.join(responses_dir, crop_filename)
    cv2.imwrite(crop_path, crop)

    # Writing outside the box (below it, down to the next box, or in the side margins)
    # is part of the answer too: measure it and save a crop with its surroundings.
    next_box_tops = [float(a["y_mm"]) for a in open_areas
                     if str(a.get("model_id")) == model_id and a.get("page_index") == area.get("page_index")
                     and float(a["y_mm"]) > float(area["y_mm"])]
    below_mm = min([float(area["y_mm"]) + float(area["height_mm"]) + OUTSIDE_BELOW_MM,
                    layout.PRINTABLE_HEIGHT + CONTEXT_MARGIN_MM - 1] + [t - 1 for t in next_box_tops])
    below_px = int(below_mm * PX_PER_MM) + margin_px
    hand = pages.handwriting(target_page_index, model_id, int(area["page_index"]))
    ink_mm2 = None if hand is None else ink_outside_box(hand, (x, y, w, h), below_px, margin_px, PX_PER_MM)
    ink_outside = "unknown" if ink_mm2 is None else ("yes" if ink_mm2 >= INK_OUTSIDE_MIN_MM2 else "no")
    context_path, context_box = "", ""
    if ink_outside == "yes":
        top = max(0, y - int(2 * PX_PER_MM))
        context_dir = os.path.join(responses_dir, "context")
        os.makedirs(context_dir, exist_ok=True)
        context_path = os.path.join(context_dir, crop_filename)
        cv2.imwrite(context_path, warped_page[top:below_px, :])
        context_box = f"{x},{y - top},{w},{h}"

    return {
        "student_id": student_id,
        "student_name": str(result.get("student_name", "")),
        "model_id": model_id,
        "question_id": area.get("question_id"),
        "original_id": area.get("original_id") or "",
        "points": area.get("points") or "",
        "answer_sheet_page": answer_page_index + 1,
        "response_page": target_page_index + 1,
        "x_mm": area.get("x_mm"),
        "y_mm": area.get("y_mm"),
        "width_mm": area.get("width_mm"),
        "height_mm": area.get("height_mm"),
        # Relative to the index CSV (saved in output_dir), so the folder stays portable
        # and does not depend on the working directory used when correcting.
        "crop_path": os.path.relpath(crop_path, output_dir),
        "layout_check": "ok" if layout_ok else "box_not_found",
        # Handwriting outside the box (yes/no; unknown without the blank exam PDF). When
        # yes, context_crop_path shows the box and its surroundings, and context_box is
        # the box within that image (x,y,width,height in pixels).
        "ink_outside_box": ink_outside,
        "ink_outside_mm2": "" if ink_mm2 is None else round(ink_mm2, 1),
        "context_crop_path": os.path.relpath(context_path, output_dir) if context_path else "",
        "context_box": context_box,
    }


def _write_index(index_rows: List[dict], output_dir: str, responses_dir: str) -> None:
    mismatched = [r for r in index_rows if r["layout_check"] != "ok"]
    if mismatched:
        logging.warning(
            "%d of %d answer crops do not line up with the printed box borders (layout_check=box_not_found in "
            "open_responses_index.csv). Is --exam-dir the exact version that was printed? Check a few crops.",
            len(mismatched), len(index_rows),
        )
    index_path = os.path.join(output_dir, "open_responses_index.csv")
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)
    logging.info("Saved %d open-answer response crop(s) to %s", len(index_rows), responses_dir)
    outside = [r for r in index_rows if r["ink_outside_box"] == "yes"]
    if outside:
        logging.warning(
            "%d answer(s) have writing outside their box (ink_outside_box=yes); pevaluate exam-open grades "
            "them from context_crop_path, which includes it.", len(outside))


def sync_index_with_results(output_dir: str) -> int:
    """Copy student_id and student_name from ``correction_results.csv`` (after a manual fix, e.g.
    the names from the class list) into ``open_responses_index.csv``, matched by answer-sheet
    page, so that ``pevaluate exam-open`` uses them. Returns the number of fields changed."""
    results_path = os.path.join(output_dir, "correction_results.csv")
    index_path = os.path.join(output_dir, "open_responses_index.csv")
    if not (os.path.exists(results_path) and os.path.exists(index_path)):
        return 0
    with open(results_path, newline="", encoding="utf-8-sig") as f:
        by_page = {str(r.get("page", "")): r for r in csv.DictReader(f)}
    with open(index_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    changed = 0
    for row in rows:
        result = by_page.get(str(row.get("answer_sheet_page", "")))
        if result is None:
            continue
        for key in ("student_id", "student_name"):
            if result.get(key, "") != row.get(key, ""):
                if key == "student_id":
                    logging.warning("Open answers of page %s: student_id %s -> %s (pevaluate exam-open grades "
                                    "them again: its saved grades are kept by student_id).",
                                    row["answer_sheet_page"], row[key], result[key])
                row[key] = result.get(key, "")
                changed += 1
    if changed:
        with open(index_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        logging.info("Updated %d student id/name field(s) in %s from correction_results.csv.", changed, index_path)
    return changed


def extract_open_answer_responses(images_to_process: List[np.ndarray], page_results: List[dict], questions_dir: str,
                                  output_dir: str, debug_dir: str) -> None:
    """Crop the open-answer areas of every corrected exam (positions from the exam's
    ``open_answer_areas.json``) to ``<output_dir>/open_responses/`` and index them in
    ``<output_dir>/open_responses_index.csv``."""
    areas_path = Path(questions_dir) / "open_answer_areas.json"
    if not areas_path.exists():
        logging.info("No open_answer_areas.json found. Skipping open-answer crop extraction.")
        return
    try:
        open_areas = json.loads(areas_path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.error(f"Failed to read open-answer area metadata at {areas_path}: {e}")
        return
    if not open_areas:
        return

    responses_dir = os.path.join(output_dir, "open_responses")
    os.makedirs(responses_dir, exist_ok=True)
    pages = _ScannedPages(images_to_process, questions_dir, debug_dir)
    index_rows = []
    for result in page_results:
        answer_page_index = int(result.get("page", 0)) - 1
        if answer_page_index < 0:
            continue
        for area in open_areas:
            if str(area.get("model_id")) != str(result.get("model_id", "")):
                continue
            try:
                target_page_index = answer_page_index + int(area["page_index"]) - 1
            except (KeyError, TypeError, ValueError):
                logging.warning("Invalid open-answer area metadata row: %s", area)
                continue
            if target_page_index < 0 or target_page_index >= len(images_to_process):
                logging.warning(
                    "Open-answer page out of range for student %s, model %s, question %s: target page %s.",
                    str(result.get("student_id", "unknown")), str(result.get("model_id", "")),
                    area.get("question_id"), target_page_index + 1,
                )
                continue
            row = _crop_area(pages, open_areas, area, result, answer_page_index, target_page_index,
                             output_dir, responses_dir)
            if row is not None:
                index_rows.append(row)

    if not index_rows:
        logging.warning("No open-answer response crops were extracted.")
        return
    _write_index(index_rows, output_dir, responses_dir)
