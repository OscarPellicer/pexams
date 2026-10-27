"""HTML of the answer sheet: header, student boxes and bubble grid at the layout coordinates."""
from pexams import layout
from pexams.generate_exams import _generate_answer_sheet_html
from pexams.schemas import PexamOption, PexamQuestion


def _questions(n):
    return [
        PexamQuestion(id=i, original_id=f"q{i}", text=f"q{i}",
                      options=[PexamOption(text=t, is_correct=k == 0) for k, t in enumerate("abcde"[: 2 + i % 4])])
        for i in range(1, n + 1)
    ]


def test_full_sheet_has_header_fiducials_and_one_bubble_per_option():
    questions = _questions(23)
    html = _generate_answer_sheet_html(questions, 2, "Title", "Course", "2026-01-01", lang="es")
    data = layout.get_answer_sheet_layout(questions)
    assert html.count('class="fiducial ') == 4
    assert html.count('class="option-box"') == sum(len(v) for v in data.answer_boxes.values())
    assert html.count('class="question-number"') == len(questions)
    assert html.count('class="header-option"') == sum(len(v) for v in data.header_labels.values())
    assert '<div class="model-id-text"' in html and ">2</div>" in html
    assert "Course" in html and "2026-01-01" in html and "instructions-box" in html
    x, y = data.student_id_box.top_left
    assert f'class="id-box" style="position: absolute; left: {x}mm; top: {y}mm;' in html


def test_header_only_has_no_grid_or_instructions():
    html = _generate_answer_sheet_html(_questions(5), 1, "T", None, None, lang="en", header_only=True)
    assert html.startswith('<div class="first-page-header">')
    for absent in ("option-box", "question-number", "instructions-box", "fiducial", "<span>"):
        assert absent not in html
    for present in ("exam-title", "model-id-box", "student-name-box", "id-box", "student-signature-box"):
        assert f'class="{present}"' in html
