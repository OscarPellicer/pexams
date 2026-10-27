import base64
import logging
import os
import random
import re
from typing import Dict, List, Optional
from xml.dom import minidom

from pexams.markdown_render import (inline_image_styles, limit_image_size, question_image_limits, render_markdown,
                                     replace_image_sources)
from pexams.schemas import PexamQuestion


def _moodle_html(text: str, base_dir: Optional[str], files: Dict[str, str], nl2br: bool = False) -> str:
    """Question text, option or feedback as Moodle HTML: Markdown rendered like the PDF, math
    for Moodle's MathJax filter, image layout as inline styles, and every local image attached
    to the element as ``@@PLUGINFILE@@/<name>`` (``files``: name -> path)."""
    html_text = inline_image_styles(render_markdown(text or "", nl2br=nl2br))
    return replace_image_sources(html_text, base_dir, lambda path: "@@PLUGINFILE@@/" + _attach(files, path))


def _attach(files: Dict[str, str], path: str) -> str:
    """Register ``path`` among an element's files under a unique name."""
    for name, existing in files.items():
        if existing == path:
            return name
    stem, ext = os.path.splitext(os.path.basename(path))
    name, n = stem + ext, 1
    while name in files:
        n += 1
        name = f"{stem}_{n}{ext}"
    files[name] = path
    return name


def _text_element(doc: minidom.Document, parent, name: str, text):
    """``<name>text</name>`` (empty when ``text`` is None)."""
    elem = doc.createElement(name)
    if text is not None:
        elem.appendChild(doc.createTextNode(str(text)))
    parent.appendChild(elem)
    return elem


def _html_element(doc: minidom.Document, parent, name: str, html_text: str, files: Optional[Dict[str, str]] = None):
    """``<name format="html"><text>CDATA</text><file .../>...</name>``, the files embedded in base64."""
    elem = doc.createElement(name)
    elem.setAttribute("format", "html")
    text_elem = doc.createElement("text")
    text_elem.appendChild(doc.createCDATASection(html_text))
    elem.appendChild(text_elem)
    for file_name, path in (files or {}).items():
        file_elem = doc.createElement("file")
        file_elem.setAttribute("name", file_name)
        file_elem.setAttribute("path", "/")
        file_elem.setAttribute("encoding", "base64")
        with open(path, "rb") as f:
            file_elem.appendChild(doc.createTextNode(base64.b64encode(f.read()).decode("ascii")))
        elem.appendChild(file_elem)
    parent.appendChild(elem)
    return elem


def _add_name(doc: minidom.Document, question_elem, index: int, q: PexamQuestion) -> None:
    name = doc.createElement("name")
    question_elem.appendChild(name)
    safe_name = re.sub(r'[^\w\s-]', '', q.text[:50])
    _text_element(doc, name, "text", f"Q{index + 1}: {safe_name}")


def _add_question_text(doc: minidom.Document, question_elem, q: PexamQuestion, base_dir: Optional[str],
                       max_image_width: Optional[int], max_image_height: Optional[int]) -> None:
    """The question text, its images attached and limited to the question's maximum size."""
    files: Dict[str, str] = {}
    q_text_html = _moodle_html(q.text, base_dir, files, nl2br=True)
    q_text_html = limit_image_size(q_text_html, *question_image_limits(q, max_image_width, max_image_height))
    _html_element(doc, question_elem, "questiontext", q_text_html, files)


def _add_general_feedback(doc: minidom.Document, question_elem, q: PexamQuestion, base_dir: Optional[str]) -> None:
    files: Dict[str, str] = {}
    html_text = _moodle_html(q.explanation, base_dir, files) if q.explanation else ""
    _html_element(doc, question_elem, "generalfeedback", html_text, files)


_STANDARD_FEEDBACK = (
    ("correctfeedback", "<p>Your answer is correct.</p>"),
    ("partiallycorrectfeedback", "<p>Your answer is partially correct.</p>"),
    ("incorrectfeedback", "<p>Your answer is incorrect.</p>"),
)


def _add_settings_and_feedback(doc: minidom.Document, question_elem, q: PexamQuestion) -> None:
    """Grade, penalty and shuffle settings, and the standard per-outcome feedback."""
    settings = (
        ("defaultgrade", "1.0"),
        ("penalty", "0.3333333"),
        ("hidden", "0"),
        ("idnumber", str(q.id)),
        ("single", "true"),
        ("shuffleanswers", "true"),
        ("answernumbering", "abc"),
    )
    for name, value in settings:
        _text_element(doc, question_elem, name, value)
    for name, text in _STANDARD_FEEDBACK:
        _html_element(doc, question_elem, name, text)


def _add_answers(doc: minidom.Document, question_elem, q: PexamQuestion, base_dir: Optional[str]) -> None:
    """The options in random order, each with its own files and an empty feedback."""
    options_list = list(q.options)
    random.shuffle(options_list)
    for option in options_list:
        files: Dict[str, str] = {}
        answer = _html_element(doc, question_elem, "answer", _moodle_html(option.text, base_dir, files), files)
        answer.setAttribute("fraction", "100" if option.is_correct else "0")
        _html_element(doc, answer, "feedback", "")


def _question_element(doc: minidom.Document, index: int, q: PexamQuestion, base_dir: Optional[str],
                      max_image_width: Optional[int], max_image_height: Optional[int]):
    question_elem = doc.createElement("question")
    question_elem.setAttribute("type", "multichoice")
    _add_name(doc, question_elem, index, q)
    _add_question_text(doc, question_elem, q, base_dir, max_image_width, max_image_height)
    _add_general_feedback(doc, question_elem, q, base_dir)
    _add_settings_and_feedback(doc, question_elem, q)
    _add_answers(doc, question_elem, q, base_dir)
    return question_elem


def convert_to_moodle_xml(questions: List[PexamQuestion], output_file: str, max_image_width: Optional[int] = None,
                          max_image_height: Optional[int] = None, base_dir: Optional[str] = None):
    """
    Converts questions to Moodle XML format using PexamQuestion, building the XML
    with minidom to properly handle CDATA sections for formatted text. Images in the text
    (paths relative to ``base_dir``, the folder of the questions file) are embedded.
    """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    logging.info(f"Converting {len(questions)} questions to Moodle XML: {output_file}")

    doc = minidom.Document()
    quiz = doc.createElement("quiz")
    doc.appendChild(quiz)

    for i, q in enumerate(questions):
        if q.is_open_answer:
            logging.warning("Skipping open-answer question %s: Moodle XML export currently supports multiple choice only.", q.id)
            continue
        quiz.appendChild(_question_element(doc, i, q, base_dir, max_image_width, max_image_height))

    try:
        pretty_xml = doc.toprettyxml(indent="  ", encoding='utf-8')
        with open(output_file, 'wb') as f:
            f.write(pretty_xml)
        logging.info(f"Successfully converted questions to Moodle XML: {output_file}")
    except Exception as e:
        logging.error(f"Failed to write Moodle XML file {output_file}: {e}", exc_info=True)
