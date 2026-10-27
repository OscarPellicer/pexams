"""Reading the header boxes of an answer sheet (model, student ID, name): a vision LLM by
default, the local TrOCR model as the fallback."""
import logging
import os
import re
from typing import Optional, Tuple

import cv2
from PIL import Image

TROCR_MODEL = "microsoft/trocr-base-printed"
BOX_PADDING_PX = 5  # crop inside the box, to leave its printed border out

_HEADER_PROMPT = (
    "These are crops of the header boxes of one scanned paper exam, in this order: "
    "{boxes}. The contents are handwritten (the model box may be printed). Transcribe them exactly as written; "
    "do not guess missing characters (use '?' for an unreadable character). "
    'Return only JSON: {{"model_id": "...", "student_id": "...", "student_name": "..."}} '
    "(empty string for an empty box)."
)


def load_trocr_processor(model_name):
    """Loads the TrOCR processor. With transformers>=5 the automatic tokenizer resolution
    of the TrOCR checkpoints fails (it asks for sentencepiece/tiktoken), so the processor is
    assembled from its RoBERTa BPE tokenizer and image processor instead."""
    from transformers import TrOCRProcessor  # optional: pip install "pexams[trocr]"
    try:
        return TrOCRProcessor.from_pretrained(model_name)
    except ValueError:
        from transformers import AutoImageProcessor, RobertaTokenizer
        return TrOCRProcessor(
            image_processor=AutoImageProcessor.from_pretrained(model_name),
            tokenizer=RobertaTokenizer.from_pretrained(model_name),
        )


def _box_roi(image, box, px_per_mm, padding=BOX_PADDING_PX):
    """The pixels inside a layout box of a warped sheet (without its border)."""
    tl_x, tl_y = box.top_left
    br_x, br_y = box.bottom_right
    return image[int(tl_y * px_per_mm) + padding:int(br_y * px_per_mm) - padding,
                 int(tl_x * px_per_mm) + padding:int(br_x * px_per_mm) - padding]


def _box_png(warped_image, box, px_per_mm, padding=BOX_PADDING_PX) -> Optional[bytes]:
    roi = _box_roi(warped_image, box, px_per_mm, padding)
    if roi.size == 0:
        return None
    ok, png = cv2.imencode(".png", roi)
    return png.tobytes() if ok else None


def read_header_llm(warped_image, model_layout, student_layout, px_per_mm, llm, page_number=None) -> Optional[dict]:
    """Read the model, student-ID and name boxes with a vision LLM in one call."""
    from pexams.llm import parse_json_response

    crops = [
        ("model box", _box_png(warped_image, model_layout.model_id_box, px_per_mm)),
        ("student ID box", _box_png(warped_image, student_layout.student_id_box, px_per_mm)),
        ("student name box", _box_png(warped_image, student_layout.student_name_box, px_per_mm)),
    ]
    crops = [(label, data) for label, data in crops if data]
    if not crops:
        return None
    try:
        text, _ = llm.complete(
            _HEADER_PROMPT.format(boxes=", ".join(label for label, _ in crops)),
            images=[(data, "image/png") for _, data in crops],
            max_tokens=4000,
            json_output=True,
            purpose="read_exam_header",
            context={"page": page_number},
        )
        result = parse_json_response(text)
        return {key: str(result.get(key, "") or "").strip() for key in ("model_id", "student_id", "student_name")}
    except Exception as e:
        logging.error("LLM header reading failed on page %s: %s", page_number, e)
        return None


def _trocr_box_text(warped_image, box, px_per_mm, trocr, max_length, what, debug_path=None) -> Optional[str]:
    """Raw TrOCR text of a header box (None for an empty crop or if the model failed)."""
    processor, model, device = trocr
    roi = _box_roi(cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY), box, px_per_mm)
    if roi.size == 0:
        return None
    if debug_path and logging.getLogger().isEnabledFor(logging.DEBUG):
        cv2.imwrite(debug_path, roi)
    try:
        pixel_values = processor(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)),
                                 return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values, max_length=max_length)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        logging.error(f"Error during OCR for {what}: {e}")
        return None


def _debug_path(debug_dir, page_number, suffix):
    return os.path.join(debug_dir, f"page_{page_number}_{suffix}.png") if debug_dir and page_number else None


def ocr_model_id(warped_image, layout_data, px_per_mm, trocr, debug_dir=None, page_number=None) -> str:
    """TrOCR reading of the model box: its first digit or capital letter."""
    text = _trocr_box_text(warped_image, layout_data.model_id_box, px_per_mm, trocr, 4, "model ID",
                           _debug_path(debug_dir, page_number, "ocr_model_id"))
    if text is None:
        return ""
    m = re.search(r"[0-9A-Z]", text.strip().upper())
    if not m:
        logging.warning(f"OCR for model ID: No valid character detected from '{text}'.")
        return ""
    logging.debug(f"OCR for model ID: Detected '{m.group(0)}' from raw '{text}'.")
    return m.group(0)


def ocr_student_id(warped_image, layout_data, px_per_mm, trocr, debug_dir=None, page_number=None) -> str:
    """TrOCR reading of the student-ID box: its digits and capital letters."""
    text = _trocr_box_text(warped_image, layout_data.student_id_box, px_per_mm, trocr, 20, "student ID",
                           _debug_path(debug_dir, page_number, "ocr_id"))
    if text is None:
        return ""
    student_id = re.sub(r"[^0-9A-Z]", "", text.upper())
    logging.debug(f"OCR for student ID: Detected '{student_id}' from '{text}'.")
    return student_id


def ocr_student_name(warped_image, layout_data, px_per_mm, trocr, debug_dir=None, page_number=None) -> str:
    """TrOCR reading of the name box (letters, digits and spaces)."""
    text = _trocr_box_text(warped_image, layout_data.student_name_box, px_per_mm, trocr, 50, "student name",
                           _debug_path(debug_dir, page_number, "ocr_name"))
    if text is None:
        return ""
    student_name = re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()
    logging.debug(f"OCR for name: Detected '{student_name}' from raw '{text}'.")
    return student_name


class HeaderReader:
    """Reads the header of each answer sheet with a vision LLM (``id_reader="llm"``) or TrOCR,
    which is loaded on first use and is also the fallback when the LLM is unavailable or a call
    fails. ``llm`` is the LLM client (None without one), also used to resolve double marks."""

    def __init__(self, id_reader: str, model: str, effort: Optional[str], api_key_env: Optional[str],
                 output_dir: str, inputs=(), px_per_mm: float = 10.0, debug_dir: Optional[str] = None):
        self.px_per_mm = px_per_mm
        self.debug_dir = debug_dir
        self.llm = None
        self._trocr = None
        self._trocr_error = None
        if id_reader == "llm":
            from pexams.llm import LLMClient, write_run_manifest

            try:
                self.llm = LLMClient(model, effort=effort, api_key_env=api_key_env,
                                     log_path=os.path.join(output_dir, "llm_calls.jsonl"), tool="pexams")
                write_run_manifest(output_dir, "pexams correct", self.llm, inputs=list(inputs),
                                   extra={"id_reader": "llm"})
                logging.info("Reading exam headers with %s (%s).", model, self.llm.provider)
            except Exception as e:
                logging.warning("LLM header reading unavailable (%s). Falling back to TrOCR.", e)

    def trocr_ready(self) -> bool:
        """Load TrOCR on first use; False (logged once) if it cannot be loaded."""
        if self._trocr is None and self._trocr_error is None:
            try:
                import torch
                from transformers import VisionEncoderDecoderModel

                device = "cuda" if torch.cuda.is_available() else "cpu"
                logging.info(f"Loading TrOCR on {device}.")
                processor = load_trocr_processor(TROCR_MODEL)
                model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(device)
                self._trocr = (processor, model, device)
            except Exception as e:
                self._trocr_error = e
                logging.error(f"Failed to initialize TrOCR model: {e}")
                logging.error('TrOCR needs an internet connection and the optional packages: pip install "pexams[trocr]". '
                              'Or read the headers with an LLM (an API key).')
        return self._trocr is not None

    @property
    def ready(self) -> bool:
        return self.llm is not None or self.trocr_ready()

    def read_model_id(self, warped_sheet, header_layout, page_number) -> Tuple[str, Optional[dict]]:
        """The model read from the sheet, and the LLM's reading of the whole header (None with TrOCR).
        The header boxes do not depend on the questions, so one LLM call reads all three."""
        header = None
        if self.llm is not None:
            header = read_header_llm(warped_sheet, header_layout, header_layout, self.px_per_mm, self.llm, page_number)
        if header is not None:
            return re.sub(r"[^0-9A-Z]", "", header["model_id"].upper())[:1], header
        if self.trocr_ready():
            return ocr_model_id(warped_sheet, header_layout, self.px_per_mm, self._trocr, self.debug_dir, page_number), None
        return "", None

    def read_student(self, warped_sheet, sheet_layout, header: Optional[dict], page_number) -> Tuple[str, str]:
        """The student ID and name: from the LLM's ``header`` if there is one, else with TrOCR."""
        if header is not None:
            student_id = re.sub(r"[^0-9A-Z?]", "", header["student_id"].upper())
            logging.info("Page %s: LLM read ID '%s', name '%s'.", page_number, student_id, header["student_name"])
            return student_id, header["student_name"]
        if self.trocr_ready():
            args = (warped_sheet, sheet_layout, self.px_per_mm, self._trocr, self.debug_dir, page_number)
            return ocr_student_id(*args), ocr_student_name(*args)
        return "", ""
