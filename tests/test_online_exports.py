"""Moodle XML and GIFT exports embed every image of the text, as the PDF shows it."""
import textwrap
from xml.dom import minidom

from PIL import Image

from pexams.io import gift_converter, moodle_xml_converter
from pexams.io.md_converter import load_questions_from_md

MD = textwrap.dedent("""\
    ## q_inline
    Look at ![w:30mm right|a figure](figs/fig.png) and at $x_1^2$:

    ```python
    print(1)

    print(2)
    ```
    * ![h:10mm](figs/fig.png) Red
    * Blue

    **Explanation:**
    The figure ![](figs/other.png) is red.

    ## q_main
    > ![main](main.png)
    Which colour?
    * Blue
    * Red
    """)


def _questions(tmp_path):
    (tmp_path / "figs").mkdir()
    Image.new("RGB", (40, 30), "red").save(tmp_path / "figs" / "fig.png")
    Image.new("RGB", (40, 30), "green").save(tmp_path / "figs" / "other.png")
    Image.new("RGB", (40, 30), "blue").save(tmp_path / "main.png")
    (tmp_path / "q.md").write_text(MD, encoding="utf-8")
    return load_questions_from_md(str(tmp_path / "q.md"))


def test_moodle_xml_attaches_every_image(tmp_path):
    out = tmp_path / "out" / "q.xml"
    moodle_xml_converter.convert_to_moodle_xml(_questions(tmp_path), str(out), base_dir=str(tmp_path))
    doc = minidom.parse(str(out))
    first, second = doc.getElementsByTagName("question")
    qtext = first.getElementsByTagName("questiontext")[0]
    html = qtext.getElementsByTagName("text")[0].firstChild.data
    assert 'src="@@PLUGINFILE@@/fig.png"' in html and "float: right" in html and "width: 30mm" in html
    assert "\\(x_1^2\\)" in html and "<pre" in html
    assert [f.getAttribute("name") for f in qtext.getElementsByTagName("file")] == ["fig.png"]
    # Each answer and the general feedback carry their own files.
    answer_files = [f.getAttribute("name") for a in first.getElementsByTagName("answer") for f in a.getElementsByTagName("file")]
    assert answer_files == ["fig.png"]
    feedback = first.getElementsByTagName("generalfeedback")[0]
    assert [f.getAttribute("name") for f in feedback.getElementsByTagName("file")] == ["other.png"]
    main_files = second.getElementsByTagName("questiontext")[0].getElementsByTagName("file")
    assert [f.getAttribute("name") for f in main_files] == ["main.png"]


def test_gift_embeds_images_as_data_uris_without_blank_lines(tmp_path):
    out = tmp_path / "q.gift"
    gift_converter.convert_to_gift(_questions(tmp_path), str(out), base_dir=str(tmp_path))
    text = out.read_text(encoding="utf-8")
    first, second = [block for block in text.split("\n\n") if block.strip()]
    assert first.startswith("::Q1_q_inline::[html]")
    assert first.count("data\\:image/png;base64,") == 3  # question, option, explanation
    assert "float\\: right" in first and "figs/fig.png" not in first
    assert "&\\#10;&\\#10;" in first  # the code's blank line survives as &#10; (GIFT escapes '#') without ending the question
    assert "data\\:image/png;base64," in second


def test_rexams_and_wooclap_take_the_images_of_the_text(tmp_path):
    from pexams.io import rexams_converter, wooclap_converter

    questions = _questions(tmp_path)
    rexams_converter.prepare_for_rexams(questions, str(tmp_path / "rexams"), base_dir=str(tmp_path))
    rmd = (tmp_path / "rexams" / "question_q_main.Rmd").read_text(encoding="utf-8")
    assert f"include_graphics('{(tmp_path / 'main.png').as_posix()}')" in rmd and "![" not in rmd
    wooclap_converter.convert_to_wooclap(questions, str(tmp_path / "wooclap.csv"))
    assert not (tmp_path / "wooclap.csv").exists() or "Which colour" not in (tmp_path / "wooclap.csv").read_text()
