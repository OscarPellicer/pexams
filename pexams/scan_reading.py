"""Scanned pages: loading (PDFs and image folders), fiducial detection and perspective warping."""
import glob
import json
import logging
import os
from typing import Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np
from pdf2image import convert_from_path

from pexams import layout

# Resolution of the warped sheets, used for the whole correction.
PX_PER_MM = 10.0

SCAN_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Fiducial crosses are 8 mm boxes: their centres are 4 mm from the printable area's edges
# (the bottom ones sit 5 mm above its bottom edge).
FIDUCIAL_CENTER_OFFSET_MM = 4
FIDUCIAL_BOTTOM_GAP_MM = 5


def load_simulated_scan_manifest(input_path: str) -> Dict[str, dict]:
    """The ``simulated_scan_manifest.json`` of a folder of simulated scans ({} if none)."""
    manifest_path = os.path.join(input_path, "simulated_scan_manifest.json")
    if not os.path.isdir(input_path) or not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception as e:
        logging.warning("Failed to read simulated scan manifest %s: %s", manifest_path, e)
        return {}


def load_scan_pages(input_paths: Union[str, Sequence[str]]) -> Tuple[List["np.ndarray"], List[str], Dict[str, dict]]:
    """Load the scanned pages of one or more inputs, in the order given.

    Each input is a PDF or a directory. A directory contributes its images and PDFs sorted
    by file name, so a batch scanned into several PDFs (e.g. timestamped by the scanner) is
    read in scan order. Pages are simply concatenated: an exam may even be split between
    two consecutive PDFs. Returns the page images, their source (the image path, or
    ``"file.pdf#page=N"``) and the merged simulated-scan manifest.
    """
    if isinstance(input_paths, (str, os.PathLike)):
        input_paths = [input_paths]

    images: List[np.ndarray] = []
    sources: List[str] = []
    manifest: Dict[str, dict] = {}

    def add_pdf(pdf_path):
        logging.info("Converting PDF pages to images: %s", pdf_path)
        for page_number, pil_img in enumerate(convert_from_path(pdf_path), start=1):
            images.append(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
            sources.append(f"{pdf_path}#page={page_number}")

    for input_path in input_paths:
        input_path = str(input_path)
        if os.path.isdir(input_path):
            manifest.update(load_simulated_scan_manifest(input_path))
            # glob returns files in arbitrary order; sort them so the pages of each exam are
            # processed in sequence (response pages are located relative to the first page).
            files = sorted(
                f for f in glob.glob(os.path.join(input_path, "*"))
                if f.lower().endswith(SCAN_IMAGE_EXTENSIONS + (".pdf",))
            )
            logging.info("Input directory %s: %d scan file(s).", input_path, len(files))
            for f in files:
                if f.lower().endswith(".pdf"):
                    add_pdf(f)
                    continue
                img = cv2.imread(f)
                if img is None:
                    logging.warning(f"Could not read image file: {f}")
                    continue
                images.append(img)
                sources.append(f)
        elif os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
            add_pdf(input_path)
        else:
            raise ValueError(f"Input path '{input_path}' is not a PDF file or a directory.")
    return images, sources, manifest


def _fiducial_template_for(image):
    """Create a plus-shaped fiducial template scaled to the scan resolution."""
    page_height, page_width = image.shape[:2]
    size = max(41, int(round(min(page_width, page_height) * 0.049)))
    if size % 2 == 0:
        size += 1
    arm = max(25, int(round(size * 0.80)))
    thickness = max(5, int(round(size * 0.09)))

    template = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    half_arm = arm // 2
    half_thickness = thickness // 2
    cv2.rectangle(template, (center - half_thickness, center - half_arm),
                  (center + half_thickness, center + half_arm), 255, -1)
    cv2.rectangle(template, (center - half_arm, center - half_thickness),
                  (center + half_arm, center + half_thickness), 255, -1)
    return template


def find_fiducial_markers(image, debug_dir=None, page_number=None):
    """Find the four fiducial crosses in the corners of a page by template matching.

    Returns their centres as a float32 array ordered TL, TR, BR, BL, or None if any is missing.
    With DEBUG logging, ``debug_dir`` and ``page_number``, intermediate images are saved.
    """
    if image is None:
        logging.error("Input image is None.")
        return None
    debug = bool(debug_dir and page_number and logging.getLogger().isEnabledFor(logging.DEBUG))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use a moderately permissive fixed threshold. The fiducials are printed,
    # but real scans often lighten the strokes enough that a very dark cutoff
    # misses parts of the cross and makes template matching unstable.
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"page_{page_number}_1_thresh.png"), thresh)

    page_height, page_width = image.shape[:2]
    corner_regions = {
        "tl": (0, int(page_width * 0.25), 0, int(page_height * 0.25)),
        "tr": (int(page_width * 0.75), page_width, 0, int(page_height * 0.25)),
        "bl": (0, int(page_width * 0.25), int(page_height * 0.75), page_height),
        "br": (int(page_width * 0.75), page_width, int(page_height * 0.75), page_height),
    }

    template = _fiducial_template_for(image)
    template_height, template_width = template.shape[:2]
    match_threshold = 0.40
    debug_img_candidates = image.copy() if debug else None

    final_corners = {}
    match_scores = {}
    for name, (x_min, x_max, y_min, y_max) in corner_regions.items():
        roi = thresh[y_min:y_max, x_min:x_max]
        if roi.shape[0] < template_height or roi.shape[1] < template_width:
            final_corners[name] = None
            match_scores[name] = 0
            continue

        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, max_location = cv2.minMaxLoc(result)
        match_scores[name] = max_value
        center = (
            int(x_min + max_location[0] + template_width / 2),
            int(y_min + max_location[1] + template_height / 2),
        )
        final_corners[name] = center if max_value >= match_threshold else None

        if debug:
            color = (0, 128, 255) if max_value >= match_threshold else (0, 0, 255)
            cv2.circle(debug_img_candidates, center, 20, color, 2)
            cv2.putText(debug_img_candidates, f"{name}:{max_value:.2f}", (center[0] + 24, center[1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"page_{page_number}_2_candidates.png"), debug_img_candidates)

    tl, tr, bl, br = final_corners["tl"], final_corners["tr"], final_corners["bl"], final_corners["br"]
    if not all([tl, tr, bl, br]):
        logging.warning(
            "Could not find all 4 corner fiducial markers. Found: TL=%s, TR=%s, BL=%s, BR=%s. Scores: %s",
            bool(tl), bool(tr), bool(bl), bool(br), {k: round(v, 3) for k, v in match_scores.items()},
        )
        return None
    logging.info("Successfully found 4 fiducial markers.")

    if debug:
        debug_img = image.copy()
        cv2.circle(debug_img, tl, 20, (0, 0, 255), 5)  # Red TL
        cv2.circle(debug_img, tr, 20, (0, 255, 0), 5)  # Green TR
        cv2.circle(debug_img, bl, 20, (255, 0, 0), 5)  # Blue BL
        cv2.circle(debug_img, br, 20, (0, 255, 255), 5)  # Yellow BR
        cv2.imwrite(os.path.join(debug_dir, f"page_{page_number}_3_centroids.png"), debug_img)

    return np.array([tl, tr, br, bl], dtype="float32")


def apply_perspective_transform(image, corners, px_per_mm: float, margin_mm: float = 0.0):
    """Warp a scanned page to a top-down view of the 180x267 mm printable area.

    ``margin_mm`` extends the output by that much on every side, so that the page margins
    outside the fiducial marks are kept too: layout coordinate (x, y) mm is then at pixel
    ((x + margin_mm) * px_per_mm, (y + margin_mm) * px_per_mm).
    """
    dst_width_px = int((layout.PRINTABLE_WIDTH + 2 * margin_mm) * px_per_mm)
    dst_height_px = int((layout.PRINTABLE_HEIGHT + 2 * margin_mm) * px_per_mm)

    offset = FIDUCIAL_CENTER_OFFSET_MM
    right = layout.PRINTABLE_WIDTH - offset
    bottom = layout.PRINTABLE_HEIGHT - FIDUCIAL_BOTTOM_GAP_MM - offset
    ideal_mm = [(offset, offset), (right, offset), (right, bottom), (offset, bottom)]  # TL, TR, BR, BL
    dst = np.array([((x + margin_mm) * px_per_mm, (y + margin_mm) * px_per_mm) for x, y in ideal_mm],
                   dtype="float32")

    matrix = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, matrix, (dst_width_px, dst_height_px), borderValue=(255, 255, 255))
