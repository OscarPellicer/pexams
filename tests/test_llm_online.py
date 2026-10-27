import os

import cv2
import numpy as np
import pytest

from pexams.header_reading import read_header_llm
from pexams.layout import get_answer_sheet_layout
from pexams.llm import LLMClient
from pexams.schemas import PexamOption, PexamQuestion

MODEL = os.getenv("PEXAMS_HEADER_MODEL", "claude-opus-5-5")
KEY_ENV = os.getenv("PEXAMS_API_KEY_ENV", "ANTHROPIC_API_KEY" if MODEL.startswith("claude-") else "OPENROUTER_API_KEY")


def _requires_llm():
    if os.getenv("RUN_LLM_TESTS") != "1":
        pytest.skip("set RUN_LLM_TESTS=1 to run real LLM tests")
    if not os.getenv(KEY_ENV):
        pytest.skip(f"{KEY_ENV} is required for real LLM tests")


@pytest.mark.llm
def test_llm_reads_printed_header():
    _requires_llm()

    px_per_mm = 10
    sheet = np.full((int(297 * px_per_mm), int(210 * px_per_mm), 3), 255, dtype=np.uint8)
    question = PexamQuestion(
        id=1,
        text="What is 2 + 2?",
        options=[PexamOption(id=1, text="4", is_correct=True), PexamOption(id=2, text="3", is_correct=False)],
    )
    layout = get_answer_sheet_layout([question])
    for box, text in ((layout.student_name_box, "Ada Lovelace"), (layout.student_id_box, "12345678")):
        tl_x, tl_y = box.top_left
        cv2.putText(sheet, text, (int(tl_x * px_per_mm) + 20, int(tl_y * px_per_mm) + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)

    llm = LLMClient(MODEL, effort="low", api_key_env=KEY_ENV)
    header = read_header_llm(sheet, layout, layout, px_per_mm, llm, page_number=1)

    assert "lovelace" in header["student_name"].lower()
    assert header["student_id"].replace(" ", "") == "12345678"
