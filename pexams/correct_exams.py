import os
import logging
import csv
import uuid
import glob
from typing import Dict, List
import numpy as np

# Attempt to import OpenCV and other libraries, with graceful failure
try:
    import cv2
    import numpy as np
    from pdf2image import convert_from_path
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    import re
    OPENCV_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import OpenCV and other libraries: {e}")
    OPENCV_AVAILABLE = False

from pexams import layout
from pexams.schemas import PexamExam, PexamQuestion, PexamOption
from pathlib import Path

# Define the standard resolution for the entire correction process for consistency.
PX_PER_MM = 10.0

def _find_fiducial_markers(image, debug_dir=None, page_number=None):
    """
    Detects four fiducial crosses in the corners of an image using contour analysis,
    inspired by the working implementation in rexams' cv_utils.py.
    """
    if image is None:
        logging.error("Input image is None.")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Using a fixed binary threshold, which is more robust for this task.
    # The value 97 is a good starting point from the rexams implementation.
    _, thresh = cv2.threshold(gray, 97, 255, cv2.THRESH_BINARY_INV)
    
    if debug_dir and page_number and logging.getLogger().isEnabledFor(logging.DEBUG):
        cv2.imwrite(os.path.join(debug_dir, f"page_{page_number}_1_thresh.png"), thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    page_height, page_width = image.shape[:2]
    
    # --- Define corner regions ---
    # (x_min, x_max, y_min, y_max) for each corner
    corner_regions = {
        "tl": (0, page_width * 0.2, 0, page_height * 0.2),
        "tr": (page_width * 0.8, page_width, 0, page_height * 0.2),
        "bl": (0, page_width * 0.2, page_height * 0.8, page_height),
        "br": (page_width * 0.8, page_width, page_height * 0.8, page_height),
    }
    
    # Store all valid candidates for each corner
    corner_candidates = {"tl": [], "tr": [], "bl": [], "br": []}

    # Find all contours that have the shape properties of a cross.
    shape_candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h) if h > 0 else 0
        if not (0.4 < aspect_ratio < 2.5):
            continue
            
        hull = cv2.convexHull(c)
        if hull.shape[0] < 3: continue
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area) if hull_area > 0 else 0
        if solidity > 0.6:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        shape_candidates.append({'cx': cx, 'cy': cy, 'area': area})

    # Sort the shape candidates by area, largest first.
    shape_candidates.sort(key=lambda c: c['area'], reverse=True)

    # Assume the fiducials are among the top 10 largest cross-like shapes.
    top_candidates = shape_candidates[:10]

    if debug_dir and page_number and logging.getLogger().isEnabledFor(logging.DEBUG):
        debug_img_candidates = image.copy()
        for cand in top_candidates:
             cv2.circle(debug_img_candidates, (cand['cx'], cand['cy']), 20, (0, 128, 255), 2)

    for cand in top_candidates:
        cx, cy = cand['cx'], cand['cy']
        for name, (x_min, x_max, y_min, y_max) in corner_regions.items():
            if (x_min < cx < x_max) and (y_min < cy < y_max):
                corner_candidates[name].append((cx, cy))
                break
    
    if debug_dir and page_number and logging.getLogger().isEnabledFor(logging.DEBUG):
        cv2.imwrite(os.path.join(debug_dir, f"page_{page_number}_2_candidates.png"), debug_img_candidates)

    # --- Select the best candidate for each corner ---
    # The best candidate is the one closest to the actual page corner.
    final_corners = {}
    page_corners = {
        "tl": (0, 0), "tr": (page_width, 0),
        "bl": (0, page_height), "br": (page_width, page_height)
    }
    
    for name, candidates in corner_candidates.items():
        if not candidates:
            final_corners[name] = None
            continue
        
        page_corner_x, page_corner_y = page_corners[name]
        
        # Calculate distance from the page corner for each candidate
        best_candidate = min(
            candidates,
            key=lambda p: np.sqrt((p[0] - page_corner_x)**2 + (p[1] - page_corner_y)**2)
        )
        final_corners[name] = best_candidate

    tl, tr, bl, br = final_corners["tl"], final_corners["tr"], final_corners["bl"], final_corners["br"]

    if not all([tl, tr, bl, br]):
        logging.warning(f"Could not find all 4 corner fiducial markers. Found: TL={bool(tl)}, TR={bool(tr)}, BL={bool(bl)}, BR={bool(br)}")
        return None
        
    logging.info("Successfully found 4 fiducial markers.")

    if debug_dir and page_number and logging.getLogger().isEnabledFor(logging.DEBUG):
        debug_img = image.copy()
        cv2.circle(debug_img, tl, 20, (0, 0, 255), 5) # Red TL
        cv2.circle(debug_img, tr, 20, (0, 255, 0), 5) # Green TR
        cv2.circle(debug_img, bl, 20, (255, 0, 0), 5) # Blue BL
        cv2.circle(debug_img, br, 20, (0, 255, 255), 5) # Yellow BR
        cv2.imwrite(os.path.join(debug_dir, f"page_{page_number}_3_centroids.png"), debug_img)
        
    return np.array([tl, tr, br, bl], dtype="float32")

def _apply_perspective_transform(image, corners, px_per_mm: float):
    """
    Applies a perspective transform to warp the scanned image into a perfect,
    top-down view corresponding to the ideal 180x267mm layout.
    """
    # Calculate the dimensions of the destination image in pixels
    dst_width_px = int(layout.PRINTABLE_WIDTH * px_per_mm)
    dst_height_px = int(layout.PRINTABLE_HEIGHT * px_per_mm)

    # Define the ideal coordinates of the fiducial centers in mm.
    # Fiducial is an 8mm box, so its center is 4mm from the edges.
    fiducial_center_offset = 4 # mm
    
    # Top-left fiducial center
    tl_ideal_mm = (fiducial_center_offset, fiducial_center_offset)
    # Top-right fiducial center
    tr_ideal_mm = (layout.PRINTABLE_WIDTH - fiducial_center_offset, fiducial_center_offset)
    # Bottom-left fiducial center (positioned 5mm from the bottom edge)
    bl_ideal_mm = (fiducial_center_offset, layout.PRINTABLE_HEIGHT - 5 - fiducial_center_offset)
    # Bottom-right fiducial center
    br_ideal_mm = (layout.PRINTABLE_WIDTH - fiducial_center_offset, layout.PRINTABLE_HEIGHT - 5 - fiducial_center_offset)

    # Convert the ideal mm coordinates to pixel coordinates for the destination points
    dst = np.array([
        (tl_ideal_mm[0] * px_per_mm, tl_ideal_mm[1] * px_per_mm),
        (tr_ideal_mm[0] * px_per_mm, tr_ideal_mm[1] * px_per_mm),
        (br_ideal_mm[0] * px_per_mm, br_ideal_mm[1] * px_per_mm),
        (bl_ideal_mm[0] * px_per_mm, bl_ideal_mm[1] * px_per_mm),
    ], dtype="float32")
        
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (dst_width_px, dst_height_px))
    return warped

def _ocr_student_id(warped_image, layout_data, px_per_mm, processor, model, device, debug_dir=None, page_number=None) -> str:
    """Performs OCR on the student ID boxes using transformers TrOCR."""
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    
    student_id = ""
    for i, box in enumerate(layout_data.student_id_boxes):
        tl_x, tl_y = box.top_left
        br_x, br_y = box.bottom_right
        
        x_px, y_px = int(tl_x * px_per_mm), int(tl_y * px_per_mm)
        x2_px, y2_px = int(br_x * px_per_mm), int(br_y * px_per_mm)

        # Add a more aggressive padding to avoid box borders
        padding = 5 
        roi = gray[y_px+padding:y2_px-padding, x_px+padding:x2_px-padding]
        
        if roi.size == 0: continue

        # Save the exact ROI being sent to the OCR for debugging
        if debug_dir and page_number and logging.getLogger().isEnabledFor(logging.DEBUG):
            cv2.imwrite(os.path.join(debug_dir, f"page_{page_number}_ocr_digit_{i}.png"), roi)

        # Convert to PIL Image for the transformer model
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(roi_rgb)
        
        char = "?"
        try:
            pixel_values = processor(img_pil, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values, max_length=4)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Keep only one allowed character (digits+uppercase)
            m = re.search(r"[0-9A-Z]", text.upper())
            if m:
                char = m.group(0)
                logging.debug(f"OCR for character {i}: Detected '{char}' from '{text}'.")
            else:
                logging.warning(f"OCR for character {i}: No valid character detected from '{text}'.")
        except Exception as e:
            logging.error(f"Error during OCR for character {i}: {e}")

        student_id += char
        
    return student_id

def _ocr_model_id(warped_image, layout_data, px_per_mm, processor, model, device, debug_dir=None, page_number=None) -> str:
    """Performs OCR on the model ID box using transformers TrOCR."""
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    
    model_box = layout_data.model_id_box
    tl_x, tl_y = model_box.top_left
    br_x, br_y = model_box.bottom_right
    
    x_px, y_px = int(tl_x * px_per_mm), int(tl_y * px_per_mm)
    x2_px, y2_px = int(br_x * px_per_mm), int(br_y * px_per_mm)
    
    padding = 5
    roi = gray[y_px+padding:y2_px-padding, x_px+padding:x2_px-padding]
    
    if roi.size == 0:
        return ""

    if debug_dir and page_number and logging.getLogger().isEnabledFor(logging.DEBUG):
        cv2.imwrite(os.path.join(debug_dir, f"page_{page_number}_ocr_model_id.png"), roi)

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    img_pil = Image.fromarray(roi_rgb)
    
    model_id = ""
    try:
        pixel_values = processor(img_pil, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values, max_length=4)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        m = re.search(r"[0-9A-Z]", text.strip().upper())
        if m:
            model_id = m.group(0)
            logging.debug(f"OCR for model ID: Detected '{model_id}' from raw '{text}'.")
        else:
            logging.warning(f"OCR for model ID: No valid character detected from '{text}'.")

    except Exception as e:
        logging.error(f"Error during OCR for model ID: {e}")

    return model_id

def _ocr_student_name(warped_image, layout_data, px_per_mm, processor, model, device, debug_dir=None, page_number=None) -> str:
    """Performs OCR on the student name box using transformers TrOCR."""
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    
    name_box = layout_data.student_name_box
    tl_x, tl_y = name_box.top_left
    br_x, br_y = name_box.bottom_right
    
    x_px, y_px = int(tl_x * px_per_mm), int(tl_y * px_per_mm)
    x2_px, y2_px = int(br_x * px_per_mm), int(br_y * px_per_mm)
    
    padding = 5
    roi = gray[y_px+padding:y2_px-padding, x_px+padding:x2_px-padding]
    
    if roi.size == 0:
        return ""

    if debug_dir and page_number and logging.getLogger().isEnabledFor(logging.DEBUG):
        cv2.imwrite(os.path.join(debug_dir, f"page_{page_number}_ocr_name.png"), roi)

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    img_pil = Image.fromarray(roi_rgb)
    
    student_name = ""
    try:
        pixel_values = processor(img_pil, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values, max_length=50) # Increased max_length for names
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean up the name: remove non-alphanumeric characters (except spaces) and strip whitespace
        student_name = re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()
        logging.debug(f"OCR for name: Detected '{student_name}' from raw '{text}'.")

    except Exception as e:
        logging.error(f"Error during OCR for student name: {e}")

    return student_name

def _analyze_and_score(warped_image, solutions: Dict[int, int], px_per_mm: float, questions: List[PexamQuestion]):
    """Analyzes the warped sheet to find marked answers and scores them."""
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    detected_answers = {}
    score = 0

    layout_data = layout.get_answer_sheet_layout(questions)
    num_questions = len(questions)

    for q_num in range(1, num_questions + 1):
        if q_num not in layout_data.answer_boxes: continue

        bubbled_option = -1
        max_filled = 0

        num_options = len(layout_data.answer_boxes[q_num])
        for opt_idx in range(num_options):
            if opt_idx not in layout_data.answer_boxes[q_num]: continue
            coords = layout_data.answer_boxes[q_num][opt_idx]
            tl_x, tl_y = coords.top_left
            br_x, br_y = coords.bottom_right
            
            x_px, y_px = int(tl_x * px_per_mm), int(tl_y * px_per_mm)
            x2_px, y2_px = int(br_x * px_per_mm), int(br_y * px_per_mm)

            roi = thresh[y_px:y2_px, x_px:x2_px]
            if roi.size == 0: continue

            filled_pixels = cv2.countNonZero(roi)
            
            if filled_pixels > max_filled and filled_pixels > (roi.size * 0.3):
                max_filled = filled_pixels
                bubbled_option = opt_idx

        detected_answers[q_num] = bubbled_option
        
        if q_num in solutions and solutions[q_num] == bubbled_option:
            score += 1
            
    return {"score": score, "total_questions": len(solutions), "answers": detected_answers}

def correct_exams(input_path: str, solutions_per_model: Dict[str, Dict[int, int]], output_dir: str, questions_dir: str) -> bool:
    if not OPENCV_AVAILABLE:
        logging.critical("Required libraries (OpenCV, etc.) are not installed.")
        return False

    logging.info(f"Starting pexams correction for: {input_path}")
    os.makedirs(output_dir, exist_ok=True)
    scanned_pages_dir = os.path.join(output_dir, "scanned_pages")
    os.makedirs(scanned_pages_dir, exist_ok=True)
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    images_to_process: List[np.ndarray] = []

    if os.path.isdir(input_path):
        logging.info("Input path is a directory, scanning for PNG/JPG images.")
        image_files = glob.glob(os.path.join(input_path, "*.png")) + \
                      glob.glob(os.path.join(input_path, "*.jpg")) + \
                      glob.glob(os.path.join(input_path, "*.jpeg"))
        
        for image_file in image_files:
            img = cv2.imread(image_file)
            if img is not None:
                images_to_process.append(img)
            else:
                logging.warning(f"Could not read image file: {image_file}")

    elif os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
        logging.info("Input path is a PDF file, converting pages to images.")
        try:
            pil_images = convert_from_path(input_path)
            for pil_img in pil_images:
                frame = np.array(pil_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                images_to_process.append(frame)
        except Exception as e:
            logging.error(f"Failed to convert PDF to images: {e}")
            return False
    else:
        logging.error(f"Input path '{input_path}' is not a valid PDF file or a directory.")
        return False

    if not images_to_process:
        logging.warning("No images found to process.")
        return False

    # Initialize TrOCR model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device} for OCR.")
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)
    except Exception as e:
        logging.error(f"Failed to initialize TrOCR model: {e}")
        logging.error("Please make sure you have an internet connection and the required transformers/torch/timm packages installed.")
        return False

    all_results = []

    # Determine the number of questions from the first model's solutions for consistent CSV headers
    first_model_key = next(iter(solutions_per_model))
    num_questions_for_header = len(solutions_per_model[first_model_key])

    dummy_options = [PexamOption(text=f'{i}', is_correct=(i==0)) for i in range(5)]
    dummy_question = PexamQuestion(id=1, text='d', options=dummy_options)

    for i, frame in enumerate(images_to_process):
        page_number = i + 1
        logging.info(f"Processing page {page_number}...")
        
        marker_corners = _find_fiducial_markers(frame, debug_dir, page_number)
        if marker_corners is None:
            logging.warning(f"Could not find 4 fiducial markers on page {page_number}. Skipping page.")
            continue
            
        warped_sheet = _apply_perspective_transform(frame, marker_corners, PX_PER_MM)
        
        # We need a layout object to find the model ID box. The number of questions doesn't affect its position.
        layout_for_model_ocr = layout.get_answer_sheet_layout(questions=[dummy_question])
        model_id = _ocr_model_id(warped_sheet, layout_for_model_ocr, PX_PER_MM, processor, model, device, debug_dir, page_number)

        if not model_id or model_id not in solutions_per_model:
            logging.warning(f"Could not detect a valid model ID for page {page_number} (detected: '{model_id}'). Skipping page.")
            continue
        
        logging.info(f"Page {page_number}: Detected Model ID '{model_id}'.")
        solutions = solutions_per_model[model_id]

        questions_path = Path(questions_dir) / f"exam_model_{model_id}_questions.json"
        if not questions_path.exists():
            logging.warning(f"Questions file not found for model {model_id} at {questions_path}. Skipping page.")
            continue
        
        try:
            exam_model = PexamExam.model_validate_json(questions_path.read_text(encoding="utf-8"))
            questions = exam_model.questions
        except Exception as e:
            logging.error(f"Failed to parse questions file for model {model_id}: {e}. Skipping page.")
            continue
        
        layout_data_for_page = layout.get_answer_sheet_layout(questions)
        student_id = _ocr_student_id(warped_sheet, layout_data_for_page, PX_PER_MM, processor, model, device, debug_dir, page_number)
        student_name = _ocr_student_name(warped_sheet, layout_data_for_page, PX_PER_MM, processor, model, device, debug_dir, page_number)

        if "?" in student_id or not student_id:
            student_id = f"unknown_{uuid.uuid4().hex[:6]}"
            logging.warning(f"Could not reliably OCR student ID for page {page_number}. Using random ID: {student_id}")

        page_result = _analyze_and_score(warped_sheet, solutions, PX_PER_MM, questions)
        page_result["page"] = page_number
        page_result["student_id"] = student_id
        page_result["student_name"] = student_name
        page_result["model_id"] = model_id
        
        # --- Create and save annotated image ---
        annotated_image = warped_sheet.copy()
        layout_data = layout_data_for_page
        
        # --- Drawing Constants ---
        green_color = (0, 255, 0)
        red_color = (0, 0, 255)
        blue_color = (255, 0, 0)
        thickness = 3
        
        # --- Draw Model ID Box and OCR'd text ---
        model_coords = layout_data.model_id_box
        tl_x, tl_y = int(model_coords.top_left[0] * PX_PER_MM), int(model_coords.top_left[1] * PX_PER_MM)
        br_x, br_y = int(model_coords.bottom_right[0] * PX_PER_MM), int(model_coords.bottom_right[1] * PX_PER_MM)
        cv2.rectangle(annotated_image, (tl_x, tl_y), (br_x, br_y), blue_color, thickness)
        
        if model_id:
            font_scale = 1.5
            font_thickness = 3
            (text_w, text_h), _ = cv2.getTextSize(model_id, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_x = tl_x + (br_x - tl_x - text_w) // 2
            text_y = br_y + text_h + 5
            cv2.putText(annotated_image, model_id, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, blue_color, font_thickness)

        # --- Draw Student Info Boxes (Blue) ---
        # Name Box
        name_coords = layout_data.student_name_box
        tl_x, tl_y = int(name_coords.top_left[0] * PX_PER_MM), int(name_coords.top_left[1] * PX_PER_MM)
        br_x, br_y = int(name_coords.bottom_right[0] * PX_PER_MM), int(name_coords.bottom_right[1] * PX_PER_MM)
        cv2.rectangle(annotated_image, (tl_x, tl_y), (br_x, br_y), blue_color, thickness)
        
        # Add OCR'd name below the name box
        if student_name:
            font_scale = 1.5
            font_thickness = 3
            (text_w, text_h), _ = cv2.getTextSize(student_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_x = tl_x
            text_y = br_y + text_h + 10  # 10 pixels padding below
            cv2.putText(annotated_image, student_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, blue_color, font_thickness)

        # Signature Box
        sig_coords = layout_data.student_signature_box
        tl_x, tl_y = int(sig_coords.top_left[0] * PX_PER_MM), int(sig_coords.top_left[1] * PX_PER_MM)
        br_x, br_y = int(sig_coords.bottom_right[0] * PX_PER_MM), int(sig_coords.bottom_right[1] * PX_PER_MM)
        cv2.rectangle(annotated_image, (tl_x, tl_y), (br_x, br_y), blue_color, thickness)

        # ID Boxes and OCR'd text
        ocr_id = page_result["student_id"]
        for i, box in enumerate(layout_data.student_id_boxes):
            tl_x, tl_y = int(box.top_left[0] * PX_PER_MM), int(box.top_left[1] * PX_PER_MM)
            br_x, br_y = int(box.bottom_right[0] * PX_PER_MM), int(box.bottom_right[1] * PX_PER_MM)
            cv2.rectangle(annotated_image, (tl_x, tl_y), (br_x, br_y), blue_color, thickness)
            
            if i < len(ocr_id):
                char = ocr_id[i]
                font_scale = 1.5
                font_thickness = 3
                (text_w, text_h), _ = cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                # Position the text centered horizontally and below the box
                text_x = tl_x + (br_x - tl_x - text_w) // 2
                text_y = br_y + text_h + 5 # 5 pixels padding below
                cv2.putText(annotated_image, char, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, blue_color, font_thickness)
                
        # --- Draw Answer Annotations (Green/Red) ---
        for q_num, correct_idx in solutions.items():
            student_idx = page_result["answers"].get(q_num, -1)
            
            # Draw green box for the correct answer
            if q_num in layout_data.answer_boxes and correct_idx in layout_data.answer_boxes[q_num]:
                correct_coords = layout_data.answer_boxes[q_num][correct_idx]
                tl_x = int(correct_coords.top_left[0] * PX_PER_MM)
                tl_y = int(correct_coords.top_left[1] * PX_PER_MM)
                br_x = int(correct_coords.bottom_right[0] * PX_PER_MM)
                br_y = int(correct_coords.bottom_right[1] * PX_PER_MM)
                cv2.rectangle(annotated_image, (tl_x, tl_y), (br_x, br_y), green_color, thickness)

            # If student was wrong, draw red box on their answer
            if student_idx != -1 and student_idx != correct_idx:
                 if q_num in layout_data.answer_boxes and student_idx in layout_data.answer_boxes[q_num]:
                    student_coords = layout_data.answer_boxes[q_num][student_idx]
                    tl_x = int(student_coords.top_left[0] * PX_PER_MM)
                    tl_y = int(student_coords.top_left[1] * PX_PER_MM)
                    br_x = int(student_coords.bottom_right[0] * PX_PER_MM)
                    br_y = int(student_coords.bottom_right[1] * PX_PER_MM)
                    cv2.rectangle(annotated_image, (tl_x, tl_y), (br_x, br_y), red_color, thickness)

        png_path = os.path.join(scanned_pages_dir, f"{student_id}.png")
        cv2.imwrite(png_path, annotated_image)
        logging.info(f"Saved annotated scan for page {page_number} to {png_path}")

        all_results.append(page_result)

    results_csv_path = os.path.join(output_dir, "correction_results.csv")
    try:
        if not all_results:
            logging.warning("No pages were processed successfully.")
            return True

        all_q_nums = sorted(range(1, num_questions_for_header + 1))
        headers = ["page", "student_id", "student_name", "model_id", "score", "total_questions"] + [f"answer_{q}" for q in all_q_nums]

        with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for result in all_results:
                row = {
                    "page": result.get("page", "N/A"),
                    "student_id": result.get("student_id", "N/A"),
                    "student_name": result.get("student_name", "N/A"),
                    "model_id": result.get("model_id", "N/A"),
                    "score": result.get("score", "N/A"),
                    "total_questions": result.get("total_questions", "N/A")
                }
                detected_answers = result.get("answers", {})
                for q_num in all_q_nums:
                    answer = detected_answers.get(q_num, -1)
                    row[f"answer_{q_num}"] = chr(ord('A') + answer) if answer != -1 else "NA"
                
                writer.writerow(row)
        logging.info(f"Correction complete. Results saved to: {results_csv_path}")
    except IOError as e:
        logging.error(f"Failed to write results to CSV: {e}")
        return False
    
    return True
