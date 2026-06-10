import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pexams import layout
from pexams import analysis
from pexams import utils
from pexams.correct_exams import _extract_open_answer_responses
from pexams.generate_exams import _generate_questions_markdown, generate_exams
from pexams.io.md_converter import load_questions_from_md, save_questions_to_md
from pexams.schemas import PexamOption, PexamQuestion
from pexams.utils import create_solutions_from_questions


def test_legacy_multiple_choice_question_defaults_to_mc():
    question = PexamQuestion(
        id="q1",
        text="What is 2 + 2?",
        options=[
            PexamOption(text="4", is_correct=True),
            PexamOption(text="3", is_correct=False),
        ],
    )

    assert question.question_type == "multiple_choice"
    assert question.is_multiple_choice
    assert question.correct_answer_index == 0


def test_multiple_choice_still_requires_exactly_one_correct_answer():
    with pytest.raises(ValueError, match="exactly one correct"):
        PexamQuestion(
            id="q1",
            text="Broken MC question",
            options=[
                PexamOption(text="A", is_correct=False),
                PexamOption(text="B", is_correct=False),
            ],
        )


def test_open_answer_question_has_no_correct_answer_index():
    question = PexamQuestion(
        id="open_1",
        question_type="open_answer",
        text="Explain why regularization can reduce overfitting.",
        points=4,
        expected_answer="Regularization penalizes overly complex models.",
        rubric="Award points for identifying complexity control and generalization.",
    )

    assert question.is_open_answer
    assert question.correct_answer_index is None
    assert question.answer_area.lines == 8


def test_markdown_loads_mixed_questions(tmp_path):
    md_path = tmp_path / "mixed.md"
    md_path.write_text(
        """## mc_1
What is 2 + 2?
* 4
* 3

## open_1 {type=open points=4 lines=12}
Explain why regularization can reduce overfitting.

**Expected answer:**
Regularization penalizes overly complex models.

**Rubric:**
2 points for complexity control; 2 points for generalization.
""",
        encoding="utf-8",
    )

    questions = load_questions_from_md(str(md_path))

    assert len(questions) == 2
    assert questions[0].question_type == "multiple_choice"
    assert questions[0].correct_answer_index == 0
    assert questions[1].question_type == "open_answer"
    assert questions[1].points == 4
    assert questions[1].answer_area.lines == 12
    assert "Regularization penalizes" in questions[1].expected_answer
    assert "complexity control" in questions[1].rubric


def test_markdown_loads_per_question_font_size(tmp_path):
    md_path = tmp_path / "font.md"
    md_path.write_text(
        """## q1 {points=2 font_size=8pt}
What is 2 + 2?
* 4
* 3
""",
        encoding="utf-8",
    )

    questions = load_questions_from_md(str(md_path))

    assert questions[0].font_size == "8pt"


def test_markdown_round_trips_open_answer_metadata(tmp_path):
    output_path = tmp_path / "roundtrip.md"
    original = [
        PexamQuestion(
            id="open_1",
            question_type="open_answer",
            text="Discuss the precision-recall tradeoff.",
            points=3,
            expected_answer="A higher threshold usually increases precision and lowers recall.",
            rubric="Award up to 3 points for threshold, precision, and recall.",
        )
    ]

    save_questions_to_md(original, str(output_path))
    loaded = load_questions_from_md(str(output_path))

    assert len(loaded) == 1
    assert loaded[0].question_type == "open_answer"
    assert loaded[0].points == 3
    assert loaded[0].expected_answer == original[0].expected_answer
    assert loaded[0].rubric == original[0].rubric


def test_solution_builder_ignores_open_answers_for_mc_scoring():
    questions = [
        PexamQuestion(
            id=1,
            text="What is 2 + 2?",
            options=[
                PexamOption(text="4", is_correct=True),
                PexamOption(text="3", is_correct=False),
            ],
        ),
        PexamQuestion(
            id=2,
            question_type="open_answer",
            text="Explain the bias-variance tradeoff.",
            points=4,
            rubric="Award points for bias, variance, and model complexity.",
        ),
    ]

    solutions_full, solutions_simple, max_score = create_solutions_from_questions(questions)

    assert list(solutions_full["1"].keys()) == [1]
    assert solutions_simple["1"] == {1: 0}
    assert max_score == 1


def test_answer_sheet_layout_uses_real_mc_question_ids():
    questions = [
        PexamQuestion(
            id=1,
            text="What is 2 + 2?",
            options=[
                PexamOption(text="4", is_correct=True),
                PexamOption(text="3", is_correct=False),
            ],
        ),
        PexamQuestion(
            id=3,
            text="What is 3 + 3?",
            options=[
                PexamOption(text="6", is_correct=True),
                PexamOption(text="5", is_correct=False),
            ],
        ),
    ]

    layout_data = layout.get_answer_sheet_layout(questions)

    assert list(layout_data.answer_boxes.keys()) == [1, 3]
    assert list(layout_data.question_numbers.keys()) == [1, 3]


def test_question_markdown_renders_open_answer_box():
    question = PexamQuestion(
        id=2,
        question_type="open_answer",
        text="Explain the bias-variance tradeoff.",
        points=4,
        rubric="Award points for bias, variance, and model complexity.",
    )

    html = _generate_questions_markdown([question])

    assert 'class="question-wrapper open-answer-question"' in html
    assert 'class="open-answer-box"' in html
    assert 'data-question-id="2"' in html
    assert 'min-height: 62mm;' in html
    assert html.count('class="open-answer-line"') == 0


def test_question_markdown_can_render_optional_writing_lines():
    question = PexamQuestion(
        id=2,
        question_type="open_answer",
        text="Explain the bias-variance tradeoff.",
        answer_area={"lines": 5, "show_lines": True},
    )

    html = _generate_questions_markdown([question])

    assert html.count('class="open-answer-line"') == 5


def test_question_markdown_renders_per_question_font_size():
    question = PexamQuestion(
        id=1,
        text="Small text question",
        font_size="8pt",
        options=[PexamOption(text="A", is_correct=True), PexamOption(text="B", is_correct=False)],
    )

    html = _generate_questions_markdown([question])

    assert 'style="font-size: 8pt;"' in html


def test_generate_mixed_exam_writes_open_answer_area_metadata(tmp_path, monkeypatch):
    class FakePage:
        def __init__(self):
            self.evaluate_calls = 0

        def goto(self, *args, **kwargs):
            return None

        def evaluate(self, script):
            self.evaluate_calls += 1
            if self.evaluate_calls == 1:
                return None
            return [{
                "question_id": "2",
                "page_index": 3,
                "x_mm": 10,
                "y_mm": 42,
                "width_mm": 160,
                "height_mm": 55,
                "lines": 8,
            }]

        def wait_for_timeout(self, *args, **kwargs):
            return None

        def pdf(self, path, **kwargs):
            Path(path).write_bytes(b"%PDF-1.4\n% fake\n")

    class FakeBrowser:
        def new_page(self):
            return FakePage()

        def close(self):
            return None

    class FakeChromium:
        def launch(self):
            return FakeBrowser()

    class FakePlaywright:
        chromium = FakeChromium()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    questions = [
        PexamQuestion(
            id="mc",
            text="What is 2 + 2?",
            options=[
                PexamOption(text="4", is_correct=True),
                PexamOption(text="3", is_correct=False),
            ],
        ),
        PexamQuestion(
            id="open",
            question_type="open_answer",
            text="Explain regularization.",
            points=4,
            rubric="Mention penalty and generalization.",
        ),
    ]

    utils.set_seeds(seed_questions=None, seed_answers=42)
    monkeypatch.setattr("pexams.generate_exams.sync_playwright", lambda: FakePlaywright())

    generate_exams(questions, str(tmp_path), num_models=1, keep_html=True)

    areas = json.loads((tmp_path / "open_answer_areas.json").read_text(encoding="utf-8"))
    exam_json = json.loads((tmp_path / "exam_model_1_questions.json").read_text(encoding="utf-8"))

    assert areas[0]["model_id"] == "1"
    assert areas[0]["question_id"] == "2"
    assert areas[0]["points"] == 4
    assert [q["question_type"] for q in exam_json["questions"]] == ["multiple_choice", "open_answer"]


def test_generate_exam_can_distribute_total_multiple_choice_points(tmp_path, monkeypatch):
    class FakePage:
        def goto(self, *args, **kwargs):
            return None

        def evaluate(self, script):
            return []

        def wait_for_timeout(self, *args, **kwargs):
            return None

        def pdf(self, path, **kwargs):
            Path(path).write_bytes(b"%PDF-1.4\n% fake\n")

    class FakeBrowser:
        def new_page(self):
            return FakePage()

        def close(self):
            return None

    class FakeChromium:
        def launch(self):
            return FakeBrowser()

    class FakePlaywright:
        chromium = FakeChromium()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    questions = [
        PexamQuestion(
            id="mc1",
            text="What is 2 + 2?",
            options=[PexamOption(text="4", is_correct=True), PexamOption(text="3", is_correct=False)],
        ),
        PexamQuestion(
            id="mc2",
            text="What is 3 + 3?",
            options=[PexamOption(text="6", is_correct=True), PexamOption(text="5", is_correct=False)],
        ),
    ]

    utils.set_seeds(seed_questions=None, seed_answers=42)
    monkeypatch.setattr("pexams.generate_exams.sync_playwright", lambda: FakePlaywright())

    generate_exams(questions, str(tmp_path), num_models=1, mc_total_points=5)

    exam_json = json.loads((tmp_path / "exam_model_1_questions.json").read_text(encoding="utf-8"))
    assert [q["points"] for q in exam_json["questions"]] == [2.5, 2.5]


def test_solution_builder_uses_multiple_choice_points():
    questions = [
        PexamQuestion(
            id=1,
            text="What is 2 + 2?",
            points=2.5,
            options=[
                PexamOption(text="4", is_correct=True),
                PexamOption(text="3", is_correct=False),
            ],
        ),
        PexamQuestion(
            id=2,
            text="What is 3 + 3?",
            points=0.5,
            options=[
                PexamOption(text="6", is_correct=True),
                PexamOption(text="5", is_correct=False),
            ],
        ),
    ]

    solutions_full, _, max_score = create_solutions_from_questions(questions)

    assert max_score == 3.0
    assert solutions_full["1"][1]["points"] == 2.5


def test_analysis_recalculates_marks_with_question_points(tmp_path):
    results_csv = tmp_path / "correction_results.csv"
    results_csv.write_text(
        "page,student_id,student_name,model_id,score,total_questions,answer_1,answer_2\n"
        "1,s1,Student One,1,2.5,2,A,B\n"
        "2,s2,Student Two,1,0.5,2,B,A\n",
        encoding="utf-8",
    )
    solutions_full = {
        "1": {
            1: {"correct_answer_index": 0, "points": 2.5, "options": [{"text": "4"}, {"text": "3"}]},
            2: {"correct_answer_index": 0, "points": 0.5, "options": [{"text": "6"}, {"text": "5"}]},
        }
    }

    analysis.analyze_results(
        csv_filepath=str(results_csv),
        output_dir=str(tmp_path),
        solutions_per_model=solutions_full,
        max_score=3.0,
    )

    final_marks = pd.read_csv(tmp_path / "final_marks.csv")
    assert final_marks.loc[0, "mark"] == pytest.approx(8.3333333333)
    assert final_marks.loc[1, "mark"] == pytest.approx(1.6666666667)


def test_extract_open_answer_responses_writes_crops_and_index(tmp_path, monkeypatch):
    questions_dir = tmp_path / "exam"
    output_dir = tmp_path / "out"
    debug_dir = output_dir / "debug"
    questions_dir.mkdir()
    output_dir.mkdir()
    debug_dir.mkdir()

    (questions_dir / "open_answer_areas.json").write_text(
        json.dumps([
            {
                "model_id": "1",
                "question_id": "2",
                "original_id": "open_q",
                "page_index": 2,
                "x_mm": 1,
                "y_mm": 1,
                "width_mm": 5,
                "height_mm": 4,
                "lines": 3,
                "points": 4,
            }
        ]),
        encoding="utf-8",
    )

    warped = np.full((100, 100, 3), 255, dtype=np.uint8)
    warped[10:50, 10:60] = 25
    images = [
        np.zeros((100, 100, 3), dtype=np.uint8),
        np.zeros((100, 100, 3), dtype=np.uint8),
    ]
    page_results = [{
        "page": 1,
        "student_id": "student-1",
        "student_name": "Student One",
        "model_id": "1",
    }]

    monkeypatch.setattr(
        "pexams.correct_exams._find_fiducial_markers",
        lambda *args, **kwargs: np.zeros((4, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        "pexams.correct_exams._apply_perspective_transform",
        lambda *args, **kwargs: warped,
    )

    _extract_open_answer_responses(images, page_results, str(questions_dir), str(output_dir), str(debug_dir))

    index_path = output_dir / "open_responses_index.csv"
    crop_path = output_dir / "open_responses" / "student-1_model_1_q_2.png"

    assert index_path.exists()
    assert crop_path.exists()
    assert "open_q" in index_path.read_text(encoding="utf-8")
