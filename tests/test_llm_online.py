import os

import cv2
import numpy as np
import pytest

from pexams.correct_exams import _ocr_student_name_openrouter
from pexams.layout import get_answer_sheet_layout
from pexams.schemas import PexamOption, PexamQuestion


def _requires_llm():
    if os.getenv("RUN_LLM_TESTS") != "1":
        pytest.skip("set RUN_LLM_TESTS=1 to run real LLM tests")
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is required for real LLM tests")


@pytest.mark.llm
def test_openrouter_student_name_ocr_reads_printed_name():
    _requires_llm()

    px_per_mm = 8
    sheet = np.full((int(297 * px_per_mm), int(210 * px_per_mm), 3), 255, dtype=np.uint8)
    question = PexamQuestion(
        id=1,
        text="What is 2 + 2?",
        options=[PexamOption(id=1, text="4", is_correct=True), PexamOption(id=2, text="3", is_correct=False)],
    )
    layout = get_answer_sheet_layout([question])
    tl_x, tl_y = layout.student_name_box.top_left
    x_px, y_px = int(tl_x * px_per_mm), int(tl_y * px_per_mm)
    cv2.putText(
        sheet,
        "Ada Lovelace",
        (x_px + 20, y_px + 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    model = os.getenv("PEXAMS_LLM_NAME_OCR_MODEL", "google/gemini-3.8-flash")
    detected = _ocr_student_name_openrouter(sheet, layout, px_per_mm, model_name=model)

    normalized = detected.lower()
    assert "ada" in normalized
    assert "lovelace" in normalized
