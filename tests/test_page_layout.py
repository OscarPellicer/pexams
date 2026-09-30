"""Real-browser layout tests: they render the exam with Playwright and read the measured boxes."""
import json

from pypdf import PdfReader

from pexams import utils
from pexams.generate_exams import PRINT_CONTENT_HEIGHT_MM, generate_exams
from pexams.schemas import PexamOption, PexamQuestion

USABLE_BOTTOM_MM = PRINT_CONTENT_HEIGHT_MM - 14  # bottom reserve kept free for the markers and footer


def _open(question_id, **answer_area):
    return PexamQuestion(
        id=question_id,
        question_type="open_answer",
        text=f"Explain concept {question_id}.",
        rubric="Correct explanation.",
        answer_area={"lines": 4, **answer_area},
    )


def _mc(question_id):
    return PexamQuestion(
        id=question_id,
        text=f"Multiple-choice question {question_id} with a statement long enough to wrap inside a column?",
        options=[PexamOption(text=f"Option {c}", is_correct=(c == "A")) for c in "ABCD"],
    )


def _areas(output_dir, questions, **kwargs):
    utils.set_seeds(seed_questions=None, seed_answers=42)
    generate_exams(questions, str(output_dir), num_models=1, **kwargs)
    areas = json.loads((output_dir / "open_answer_areas.json").read_text(encoding="utf-8"))
    pages = len(PdfReader(str(output_dir / "exam_model_1.pdf")).pages)
    return {a["original_id"]: a for a in areas}, pages


def test_fill_answer_space_grows_boxes_to_the_bottom_of_the_page(tmp_path):
    questions = [_open("a"), _open("b"), _open("fixed", height_mm=30)]

    plain, _ = _areas(tmp_path / "plain", questions)
    filled, _ = _areas(tmp_path / "filled", questions, fill_answer_space=True)

    for question_id in ("a", "b"):
        assert filled[question_id]["height_mm"] > plain[question_id]["height_mm"] + 5
    assert abs(filled["fixed"]["height_mm"] - 30) < 0.5  # explicit heights are kept
    last = max(filled.values(), key=lambda a: a["y_mm"])
    assert USABLE_BOTTOM_MM - 10 < last["y_mm"] + last["height_mm"] <= USABLE_BOTTOM_MM


def test_multi_column_open_answer_boxes_stay_within_their_pages(tmp_path):
    questions = []
    for i in range(1, 21):
        questions.append(_mc(f"mc{i}"))
        if i in (5, 12, 18):
            questions.append(_open(f"open{i}", lines=6))

    areas, pages = _areas(tmp_path / "two", questions, columns=2)

    assert len(areas) == 3
    for area in areas.values():
        assert area["y_mm"] >= 0
        assert area["y_mm"] + area["height_mm"] <= USABLE_BOTTOM_MM
        assert 1 <= area["page_index"] <= pages
