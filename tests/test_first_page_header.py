from pathlib import Path

import pytest

from pexams import utils
from pexams.generate_exams import _generate_questions_markdown, generate_exams
from pexams.io.md_converter import load_questions_from_md, save_questions_to_md
from pexams.schemas import PexamOption, PexamQuestion


class FakePage:
    def goto(self, *args, **kwargs):
        return None

    def emulate_media(self, **kwargs):
        return None

    def set_viewport_size(self, *args, **kwargs):
        return None

    def evaluate(self, script, arg=None):
        if arg is not None:  # per-page layout and header/footer passes
            return 1
        return []  # MathJax typeset and open-answer area extraction

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


def _open_question(question_id="open", lines=8):
    return PexamQuestion(
        id=question_id,
        question_type="open_answer",
        text="Explain regularization.",
        rubric="Mention penalty and generalization.",
        answer_area={"lines": lines},
    )


def _mc_question():
    return PexamQuestion(
        id="mc",
        text="What is 2 + 2?",
        options=[PexamOption(text="4", is_correct=True), PexamOption(text="3", is_correct=False)],
    )


def _generated_html(tmp_path, monkeypatch, questions):
    utils.set_seeds(seed_questions=None, seed_answers=42)
    monkeypatch.setattr("pexams.generate_exams.sync_playwright", lambda: FakePlaywright())
    generate_exams(questions, str(tmp_path), num_models=1, keep_html=True)
    html = (tmp_path / "exam_model_1.html").read_text(encoding="utf-8")
    return html.split("<body>", 1)[1]  # the page itself, without the embedded CSS


def test_open_answer_only_exam_puts_student_header_on_first_question_page(tmp_path, monkeypatch):
    html = _generated_html(tmp_path, monkeypatch, [_open_question("a"), _open_question("b")])

    assert "answer-sheet-page" not in html
    assert "instructions-box" not in html
    assert 'class="first-page-header"' in html
    assert 'class="first-page-header-spacer"' in html
    for field in ("model-id-box", "student-id-label", "student-name-box", "student-signature-box"):
        assert field in html
    # The header sits inside the questions page, before the questions themselves.
    assert html.index("questions-page") < html.index("first-page-header") < html.index("question-wrapper")
    # The spacer precedes the (possibly multi-column) container, so it pushes down every column.
    assert html.index("first-page-header-spacer") < html.index('class="questions-container')


def test_exam_with_multiple_choice_keeps_answer_sheet(tmp_path, monkeypatch):
    html = _generated_html(tmp_path, monkeypatch, [_mc_question(), _open_question()])

    assert "answer-sheet-page" in html
    assert "instructions-box" in html
    assert "first-page-header" not in html


def test_open_answer_box_accepts_fractional_lines():
    html = _generate_questions_markdown([_open_question(lines=8.5)])

    assert "min-height: 65.5mm;" in html  # 8.5 lines * 7mm + 6mm
    assert 'data-lines="8.5"' in html


def test_fractional_lines_draw_guide_lines_for_whole_lines_only():
    question = _open_question(lines=4.5)
    question.answer_area.show_lines = True

    html = _generate_questions_markdown([question])

    assert html.count('class="open-answer-line"') == 4


@pytest.mark.parametrize("lines, header", [(8.5, "lines=8.5"), (8, "lines=8")])
def test_markdown_round_trips_fractional_lines(tmp_path, lines, header):
    path = tmp_path / "questions.md"

    save_questions_to_md([_open_question(lines=lines)], str(path))
    loaded = load_questions_from_md(str(path))

    text = path.read_text(encoding="utf-8")
    assert header + " " in text or header + "}" in text
    assert loaded[0].answer_area.lines == lines
