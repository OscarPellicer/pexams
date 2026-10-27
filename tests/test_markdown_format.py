"""Question-file Markdown features: code blocks, images with layout specs, math and newlines."""
import csv
import json
import textwrap

import cv2
import numpy as np
import pytest
from pdf2image import convert_from_path

from pexams import utils
from pexams.open_responses import extract_open_answer_responses
from pexams.generate_exams import _generate_questions_markdown, _render_markdown, generate_exams
from pexams.io.md_converter import load_questions_from_md, parse_image_alt, parse_questions_markdown
from pexams.schemas import PexamOption, PexamQuestion


def _write_image(path, color=(255, 0, 0), size=(200, 300)):
    image = np.full((*size, 3), 255, np.uint8)
    image[5:-5, 5:-5] = color
    cv2.imwrite(str(path), image)


def _load(tmp_path, text):
    md_path = tmp_path / "questions.md"
    md_path.write_text(textwrap.dedent(text), encoding="utf-8")
    return {str(q.id): q for q in load_questions_from_md(str(md_path))}


# --- Parsing ---

def test_code_fences_are_not_split_into_headers_or_options(tmp_path):
    questions = _load(tmp_path, """\
        ## code_q
        What does this print?

        ```python
        ## not a header
        - not an option
        * neither
        1. nor this
        print(x_1 * y_2)
        ```
        * 2
        * 3

        **Explanation:**
        Because.
        """)

    question = questions["code_q"]
    assert list(questions) == ["code_q"]
    assert [o.text for o in question.options] == ["2", "3"]
    assert "## not a header" in question.text and "- not an option" in question.text
    assert question.text.endswith("```")
    assert question.explanation == "Because."


def test_open_answer_markers_inside_code_are_text(tmp_path):
    questions = _load(tmp_path, """\
        ## open_q {type=open lines=3}
        ```
        **Expected answer:**
        ```

        **Expected answer:**
        Real answer.
        """)

    assert "**Expected answer:**" in questions["open_q"].text
    assert questions["open_q"].expected_answer == "Real answer."


def test_question_text_can_start_with_bold(tmp_path):
    questions = _load(tmp_path, """\
        ## bold_q
        **Careful:** what is 2 + 2?
        * 4
        * 5
        """)

    assert questions["bold_q"].text == "**Careful:** what is 2 + 2?"
    assert [o.text for o in questions["bold_q"].options] == ["4", "5"]


def test_html_comments_are_removed_outside_code(tmp_path):
    questions = _load(tmp_path, """\
        <!--
        ## hidden
        Commented-out question
        * a
        * b
        -->
        ## kept
        Visible <!-- note --> text.
        ```html
        <!-- kept in code -->
        ```
        * a
        * b
        """)

    assert list(questions) == ["kept"]
    assert "note" not in questions["kept"].text
    assert "<!-- kept in code -->" in questions["kept"].text


def test_old_image_line_becomes_an_image_below_the_text(tmp_path):
    (tmp_path / "img").mkdir()
    _write_image(tmp_path / "img" / "fig.png")
    questions = _load(tmp_path, """\
        ## main_img

        > ![Image for question](img/fig.png)
        Question text with ![w:20mm](img/fig.png) another image.
        * a
        * b
        """)

    question = questions["main_img"]
    assert question.image_source is None
    assert question.text == ("Question text with ![w:20mm](img/fig.png) another image.\n\n"
                             "![Image for question](img/fig.png)")


def test_leading_image_with_layout_spec_moves_into_the_text(tmp_path):
    questions = _load(tmp_path, """\
        ## spec_img
        > ![w:40mm right|Chest X-ray](xray.png)
        Describe the image.
        * a
        * b
        """)

    question = questions["spec_img"]
    assert question.image_source is None
    assert question.text == "![w:40mm right|Chest X-ray](xray.png) Describe the image."


def test_parse_questions_markdown_keeps_raw_paths_and_answer_area():
    parsed = parse_questions_markdown(textwrap.dedent("""\
        # Title, ignored
        ## q1 {type=open lines=4}
        > ![x](a/b.png)
        Explain.

        **Answer area:**
        lines: 6, show_lines: true
        """))

    assert len(parsed) == 1
    assert parsed[0].text == "Explain.\n\n![x](a/b.png)"
    assert parsed[0].answer_area.lines == 6
    assert parsed[0].answer_area.show_lines is True


@pytest.mark.parametrize("alt, expected", [
    ("w:60mm right|Chest X-ray", ({"width": "60mm", "align": "right"}, "Chest X-ray")),
    ("h:200 center", ({"height": "200px", "align": "center"}, "")),
    ("w:50% left", ({"width": "50%", "align": "left"}, "")),
    ("W:3CM", ({"width": "3cm"}, "")),
    ("left ventricle", ({}, "left ventricle")),
    ("Image for question", ({}, "Image for question")),
    ("w:1in|caption", ({}, "w:1in|caption")),
    ("", ({}, "")),
])
def test_parse_image_alt(alt, expected):
    assert parse_image_alt(alt) == expected


# --- Rendering ---

def test_render_code_math_and_emphasis():
    html = _render_markdown(textwrap.dedent("""\
        Inline `x_1 * y_2` and $x_1 * y_2 = a_b * c$, *it*, _it_ and snake_case_name.
        Next line costs \\$5 and $3.

        $$
        \\sum_{i=1}^{n} x_i^2 < 1
        $$

        Inline display $$a_1$$ here.

        ```python
        price = "$x_1$"
        ```"""), nl2br=True)

    assert "<code>x_1 * y_2</code>" in html
    assert r'<span class="arithmatex">\(x_1 * y_2 = a_b * c\)</span>' in html
    assert html.count("<em>it</em>") == 2 and "snake_case_name" in html
    assert "<br />" in html
    assert "&#36;5 and $3." in html
    assert '<div class="arithmatex">\\[\\sum_{i=1}^{n} x_i^2 &lt; 1\\]</div>' in html
    assert r'<span class="arithmatex">\[a_1\]</span>' in html
    assert "<pre>" in html and "$x_1$" in html  # math is not parsed inside code


def test_render_image_layout_spec():
    html = _render_markdown("A ![w:60mm right|Chest X-ray](a.png) b ![h:2cm](b.png) c ![left ventricle](c.png)")

    assert '<img alt="Chest X-ray" class="pexams-img-right" style="width: 60mm;" src="a.png" />' in html
    assert '<img alt="" style="height: 2cm; width: auto;" src="b.png" />' in html
    assert '<img alt="left ventricle" src="c.png" />' in html


def test_question_markdown_renders_code_images_and_math_in_options():
    question = PexamQuestion(
        id=1,
        text="Look:\n\n```\na = 1\n```\n\n![w:30mm left](fig.png) Text.",
        options=[PexamOption(text="`f(x)` and $x_1$", is_correct=True), PexamOption(text="no", is_correct=False)],
    )

    html = _generate_questions_markdown([question])

    assert "<pre>" in html
    assert 'class="pexams-img-left"' in html
    assert r'<code>f(x)</code> and <span class="arithmatex">\(x_1\)</span>' in html


# --- Real browser ---

def test_floated_images_keep_open_answer_boxes_where_they_are_printed(tmp_path):
    """Floated images next to open questions: the recorded boxes must be the printed ones,
    as read back by the corrector from a rasterized page."""
    _write_image(tmp_path / "tall.png", color=(255, 0, 0), size=(400, 200))
    md_path = tmp_path / "exam.md"
    md_path.write_text(textwrap.dedent("""\
        ## mc1
        ![h:40mm right](tall.png) Short multiple-choice question with $x_1$?
        * yes
        * no

        ## open1 {type=open lines=3}
        > ![h:50mm left|Tall figure](tall.png)
        Explain the figure.

        ## open2 {type=open lines=4}
        ```python
        def f(x):
            return x_1 * 2
        ```
        ![h:45mm right](tall.png) What does it return?
        """), encoding="utf-8")
    out_dir = tmp_path / "out"
    utils.set_seeds(seed_questions=None, seed_answers=1)
    generate_exams(load_questions_from_md(str(md_path)), str(out_dir), num_models=1, keep_html=True,
                   markdown_asset_base_dir=str(tmp_path))

    html = (out_dir / "exam_model_1.html").read_text(encoding="utf-8")
    assert "polyfill" not in html
    assert html.count('class="pexams-img-') == 3
    areas = json.loads((out_dir / "open_answer_areas.json").read_text(encoding="utf-8"))
    assert {a["original_id"] for a in areas} == {"open1", "open2"}

    pages = [cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) for page in convert_from_path(str(out_dir / "exam_model_1.pdf"), dpi=150)]
    crops_dir = tmp_path / "crops"
    (crops_dir / "debug").mkdir(parents=True)
    page_results = [{"page": 1, "student_id": "s1", "student_name": "S", "model_id": "1"}]
    extract_open_answer_responses(pages, page_results, str(out_dir), str(crops_dir), str(crops_dir / "debug"))

    with open(crops_dir / "open_responses_index.csv", newline="", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    assert len(index) == 2
    assert all(row["layout_check"] == "ok" for row in index), index
    # The blank exam itself has no handwriting, inside or outside the boxes.
    assert all(row["ink_outside_box"] == "no" for row in index), index
    # The boxes clear the floated images: nothing is printed inside them.
    for crop_path in (crops_dir / "open_responses").glob("*.png"):
        crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
        h, w = crop.shape
        assert crop[h // 10:-h // 10, w // 10:-w // 10].min() > 200, crop_path.name


def test_json_image_source_moves_into_the_text_and_limits_apply_to_all_images(tmp_path):
    from pexams.markdown_render import limit_image_size
    from pexams.schemas import PexamOption, PexamQuestion

    _write_image(tmp_path / "fig.png")
    question = PexamQuestion(id=1, text="Which?", image_source=str(tmp_path / "fig.png"), max_image_width="50mm",
                             options=[PexamOption(text="a", is_correct=True), PexamOption(text="b")])
    assert question.image_source is None
    assert question.text == f"Which?\n\n![]({(tmp_path / 'fig.png').as_uri()})"
    assert PexamQuestion.model_validate_json(question.model_dump_json()).text == question.text  # idempotent

    html_text = _render_markdown(question.text + " ![w:20mm](x.png)")
    limited = limit_image_size(html_text, "50mm", "30mm")
    assert limited.count("max-width: 50mm; max-height: 30mm;") == 2
    assert 'style="width: 20mm; max-width: 50mm; max-height: 30mm;"' in limited


def test_absolute_markdown_images_uses_file_uris(tmp_path):
    from pexams.markdown_render import absolute_markdown_images

    folder = tmp_path / "with space"
    text = absolute_markdown_images("A ![w:2mm](img/a.png) and ![](https://x/y.png)", str(folder))
    assert text == f"A ![w:2mm]({(folder / 'img' / 'a.png').as_uri()}) and ![](https://x/y.png)"
    assert "%20" in text
