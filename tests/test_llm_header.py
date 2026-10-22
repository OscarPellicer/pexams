import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pexams import layout
from pexams.correct_exams import PX_PER_MM, _border_found, _read_header_llm
from pexams.llm import parse_json_response, resolve_provider
from pexams.schemas import PexamOption, PexamQuestion


def test_provider_routing():
    assert resolve_provider("claude-opus-5-5") == "anthropic"
    assert resolve_provider("anthropic/claude-opus-5-5") == "openrouter"
    assert resolve_provider("google/gemini-3.1-pro-preview") == "openrouter"
    assert resolve_provider("claude-haiku-5-5", "openrouter") == "openrouter"


def test_parse_json_response_tolerates_fences():
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('Sure: {"a": {"b": 2}} done') == {"a": {"b": 2}}


def _page_with_box(x_mm, y_mm, w_mm, h_mm):
    page = np.full((int(297 * PX_PER_MM), int(210 * PX_PER_MM), 3), 255, np.uint8)
    p = lambda v: int(v * PX_PER_MM)
    cv2.rectangle(page, (p(x_mm), p(y_mm)), (p(x_mm + w_mm), p(y_mm + h_mm)), (0, 0, 0), 2)
    cv2.putText(page, "handwriting", (p(x_mm + 10), p(y_mm + 20)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    return page


def test_layout_check_finds_box_borders_only_where_printed():
    page = _page_with_box(10, 100, 170, 60)
    p = lambda v: int(v * PX_PER_MM)
    assert _border_found(page, p(10), p(100), p(170), p(60), PX_PER_MM)
    # The same box expected 20 mm taller (a different version of the exam) is not found.
    assert not _border_found(page, p(10), p(100), p(170), p(80), PX_PER_MM)


class _FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def complete(self, prompt, images=(), **kwargs):
        self.calls.append((prompt, len(images), kwargs.get("purpose")))
        return self.reply, {}


def test_read_header_llm_sends_three_boxes_and_parses_reply():
    page = np.full((int(297 * PX_PER_MM), int(210 * PX_PER_MM), 3), 255, np.uint8)
    q = PexamQuestion(id=1, text="d", options=[PexamOption(text=str(i), is_correct=(i == 0)) for i in range(5)])
    lay = layout.get_answer_sheet_layout(questions=[q])
    llm = _FakeLLM('{"model_id": "1", "student_id": "12345678A", "student_name": "Ana Pérez"}')
    header = _read_header_llm(page, lay, lay, PX_PER_MM, llm, page_number=1)
    assert header == {"model_id": "1", "student_id": "12345678A", "student_name": "Ana Pérez"}
    assert llm.calls[0][1] == 3 and llm.calls[0][2] == "read_exam_header"
    # A failing call returns None so the caller falls back to TrOCR.
    assert _read_header_llm(page, lay, lay, PX_PER_MM, _FakeLLM("not json"), page_number=1) is None


def test_llm_module_identical_to_pevaluate_copy():
    pevaluate_llm = pytest.importorskip("autocorrect.llm")
    import pexams.llm as ours
    assert Path(ours.__file__).read_bytes() == Path(pevaluate_llm.__file__).read_bytes()
