import cv2
import numpy as np
import pandas as pd

from pexams import layout
from pexams.analysis import UNVOIDED_SCANS_DIR, mark_voided_questions_on_scans
from pexams.correct_exams import PX_PER_MM, _analyze_and_score, _resolve_multiple_marks
from pexams.schemas import PexamOption, PexamQuestion


def _questions(n=3):
    return [PexamQuestion(id=i, text=f"Q{i}", options=[PexamOption(text=t, is_correct=(t == "a")) for t in "abcd"])
            for i in range(1, n + 1)]


def _blank_sheet():
    return np.full((int(layout.PRINTABLE_HEIGHT * PX_PER_MM), int(layout.PRINTABLE_WIDTH * PX_PER_MM), 3), 255, np.uint8)


def _fill(image, sheet_layout, q, option):
    box = sheet_layout.answer_boxes[q][option]
    (x0, y0), (x1, y1) = box.top_left, box.bottom_right
    image[int(y0 * PX_PER_MM):int(y1 * PX_PER_MM), int(x0 * PX_PER_MM):int(x1 * PX_PER_MM)] = 20


class FakeLLM:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def complete(self, prompt, images=(), **kwargs):
        self.calls += 1
        return f'{{"answer": "{self.answer}", "reason": "C is crossed out"}}', {}


def test_two_filled_boxes_are_not_resolved_by_picking_the_darkest():
    questions = _questions()
    sheet_layout = layout.get_answer_sheet_layout(questions)
    image = _blank_sheet()
    _fill(image, sheet_layout, 1, 0)
    _fill(image, sheet_layout, 2, 1)
    _fill(image, sheet_layout, 2, 2)

    result = _analyze_and_score(image, {1: 0, 2: 1, 3: 0}, PX_PER_MM, questions)

    assert result["answers"][1] == 0
    assert result["answers"][2] == -1          # left blank until resolved
    assert result["multiple_marks"] == {2: [1, 2]}


def test_llm_resolves_a_double_mark_only_to_one_of_the_marked_options():
    questions = _questions()
    sheet_layout = layout.get_answer_sheet_layout(questions)
    image = _blank_sheet()

    assert _resolve_multiple_marks(image, sheet_layout, 2, [1, 2], FakeLLM("B"), PX_PER_MM) == 1
    assert _resolve_multiple_marks(image, sheet_layout, 2, [1, 2], FakeLLM("D"), PX_PER_MM) is None
    assert _resolve_multiple_marks(image, sheet_layout, 2, [1, 2], FakeLLM(""), PX_PER_MM) is None
    assert _resolve_multiple_marks(image, sheet_layout, 2, [1, 2], None, PX_PER_MM) is None


def test_voided_questions_are_marked_on_the_sheets_and_can_be_undone(tmp_path):
    questions = _questions()
    scans = tmp_path / "scanned_pages"
    scans.mkdir()
    cv2.imwrite(str(scans / "s1.png"), _blank_sheet())
    original = cv2.imread(str(scans / "s1.png"))
    df = pd.DataFrame([{"student_id": "s1", "model_id": "1"}])
    solutions = {"1": {q.id: q.model_dump() for q in questions}}

    assert mark_voided_questions_on_scans(str(tmp_path), df, solutions, [2], lang="es") == 1
    once = cv2.imread(str(scans / "s1.png"))
    assert (scans / UNVOIDED_SCANS_DIR / "s1.png").exists()
    assert (once != original).any()

    # Rerunning draws from the kept original: the marks do not pile up.
    mark_voided_questions_on_scans(str(tmp_path), df, solutions, [2], lang="es")
    assert (cv2.imread(str(scans / "s1.png")) == once).all()

    # Without voids the original sheet comes back.
    mark_voided_questions_on_scans(str(tmp_path), df, solutions, [], lang="es")
    assert (cv2.imread(str(scans / "s1.png")) == original).all()


def test_sheets_are_redrawn_from_the_raw_sheet_after_fixing_an_answer(tmp_path):
    from pexams.analysis import redraw_answer_sheets
    from pexams.correct_exams import RAW_SHEETS_DIR

    questions = _questions()                       # option A is correct in every question
    sheet_layout = layout.get_answer_sheet_layout(questions)
    raw_dir = tmp_path / "scanned_pages" / RAW_SHEETS_DIR
    raw_dir.mkdir(parents=True)
    cv2.imwrite(str(raw_dir / "s1.png"), _blank_sheet())
    solutions = {"1": {q.id: q.model_dump() for q in questions}}
    c_box = sheet_layout.answer_boxes[2][2]
    edge = (int(c_box.top_left[1] * PX_PER_MM) + 1, int((c_box.top_left[0] + c_box.bottom_right[0]) / 2 * PX_PER_MM))

    def redraw(answer_2):
        df = pd.DataFrame([{"student_id": "s1", "model_id": "1", "ocr_student_id": "S1", "ocr_student_name": "Ana",
                            "answer_1": "A", "answer_2": answer_2, "answer_3": "NA"}])
        assert redraw_answer_sheets(str(tmp_path), df, solutions, []) == 1
        return cv2.imread(str(tmp_path / "scanned_pages" / "s1.png"))

    misread = redraw("C")
    assert tuple(misread[edge]) == (0, 0, 255)     # wrong answer C drawn in red
    fixed = redraw("A")                            # the teacher fixes the answer in the CSV
    assert tuple(fixed[edge]) != (0, 0, 255)


def _outline(image, sheet_layout, q, option, thickness):
    box = sheet_layout.answer_boxes[q][option]
    (x0, y0), (x1, y1) = box.top_left, box.bottom_right
    cv2.rectangle(image, (int(x0 * PX_PER_MM), int(y0 * PX_PER_MM)), (int(x1 * PX_PER_MM), int(y1 * PX_PER_MM)),
                  (20, 20, 20), thickness)


def test_borders_do_not_count_as_fill_but_light_marks_do():
    questions = _questions(2)
    sheet_layout = layout.get_answer_sheet_layout(questions)
    image = _blank_sheet()
    for q in sheet_layout.answer_boxes:                     # the printed box borders
        for option in sheet_layout.answer_boxes[q]:
            _outline(image, sheet_layout, q, option, 2)
    # Q1: B covered with correction fluid and its border redrawn thick by hand; A is the answer.
    _outline(image, sheet_layout, 1, 1, 6)
    _fill(image, sheet_layout, 1, 0)
    # Q2: a light pencil mark covering about half of the inside of C.
    box = sheet_layout.answer_boxes[2][2]
    x0, y0 = int((box.top_left[0] + 0.5) * PX_PER_MM), int((box.top_left[1] + 0.5) * PX_PER_MM)
    x1, y1 = int((box.bottom_right[0] - 0.5) * PX_PER_MM), int((box.bottom_right[1] - 0.5) * PX_PER_MM)
    image[y0:y1, x0:x0 + (x1 - x0) // 2] = 60

    result = _analyze_and_score(image, {1: 0, 2: 2}, PX_PER_MM, questions)

    assert result["answers"] == {1: 0, 2: 2}
    assert result["multiple_marks"] == {}
