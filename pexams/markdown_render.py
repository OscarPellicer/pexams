"""Markdown to HTML for exam text: questions, options, the custom header and online exports.

Math (``$..$``, ``$$..$$``) is protected from Markdown and kept for MathJax; fenced and
inline code are highlighted; images take a layout spec in their alt text
(``![w:60mm right|Caption](img.png)``). The PDF uses CSS classes for the layout; online
exports (Moodle, GIFT) get inline styles and embedded image files.
"""
import base64
import html
import mimetypes
import os
import re
from pathlib import Path
from typing import Callable, Optional

import markdown

from pexams.io.md_converter import _code_fence_mask, parse_image_alt


def _is_relative_media_path(src: str) -> bool:
    """True if src should be resolved against a base directory (not http(s), file:, data:, or absolute)."""
    s = src.strip()
    if not s or s.startswith("#"):
        return False
    low = s.lower()
    if low.startswith(("http://", "https://", "data:", "file:")):
        return False
    if s.startswith("//"):
        return False
    try:
        return not Path(s).is_absolute()
    except Exception:
        return True


def _resolve_relative_media_uris(html: str, base_dir: str) -> str:
    """
    Rewrite relative img/video/audio src attributes to file:// URIs.

    When the exam HTML is opened via file:///.../exam_model_1.html, relative paths are
    resolved against the HTML file's directory, not the source markdown folder — so
    ![](fig.png) beside the .md file must become an absolute file URI.
    """
    if not html or not base_dir:
        return html
    base_dir = os.path.abspath(base_dir)

    def repl(match) -> str:
        quote = match.group(1)
        path = match.group(2)
        if not _is_relative_media_path(path):
            return match.group(0)
        abs_path = os.path.normpath(os.path.join(base_dir, path))
        try:
            uri = Path(abs_path).as_uri()
        except ValueError:
            return match.group(0)
        return f"src={quote}{uri}{quote}"

    return re.sub(r"\bsrc=(['\"])([^'\"]+)\1", repl, html, flags=re.IGNORECASE)

_MARKDOWN_EXTENSIONS = ['pymdownx.inlinehilite', 'fenced_code', 'codehilite']

# Scanned left to right outside fenced code: inline code spans are kept as they are, so
# '$' inside code is never math. Inline '$...$' follows Pandoc: no space after the opening
# or before the closing '$', no digit right after it, no blank line inside.
_MATH_RE = re.compile(
    r'(?P<code>(`+)[\s\S]*?(?<!`)\2(?!`))'
    r'|(?P<escaped>\\\$)'
    r'|\$\$(?P<display>[\s\S]+?)\$\$'
    r'|\\\[(?P<display_brackets>[\s\S]+?)\\\]'
    r'|\\\((?P<inline_parens>[\s\S]+?)\\\)'
    r'|\$(?=[^\s$])(?P<inline>(?:\\.|[^$\\\n]|\n(?![ \t]*\n))+?)(?<=[^\s\\])\$(?!\d)'
)
_MATH_PLACEHOLDER = "PEXAMSMATH{}END"


def _protect_math(text: str):
    r"""Replace math spans by placeholders so Markdown cannot mangle them (``_``, ``*``, newlines).

    Returns the text and the HTML for each placeholder, in MathJax's ``\(..\)``/``\[..\]`` form.
    ``\$`` becomes a literal dollar sign.
    """
    spans = []

    def repl(match):
        if match.group("code"):
            return match.group(0)
        if match.group("escaped"):
            return "&#36;"
        display = match.group("display") or match.group("display_brackets")
        tex = display if display is not None else (match.group("inline_parens") or match.group("inline"))
        delimiters = ("\\[", "\\]") if display is not None else ("\\(", "\\)")
        spans.append((display is not None, f"{delimiters[0]}{html.escape(tex.strip(), quote=False)}{delimiters[1]}"))
        return _MATH_PLACEHOLDER.format(len(spans) - 1)

    lines = text.split("\n")
    mask = _code_fence_mask(lines)
    out, chunk, chunk_is_code = [], [], False
    for line, is_code in zip(lines + [None], mask + [None]):
        if line is None or (chunk and is_code != chunk_is_code):
            joined = "\n".join(chunk)
            out.append(joined if chunk_is_code else _MATH_RE.sub(repl, joined))
            chunk = []
        if line is not None:
            chunk.append(line)
            chunk_is_code = is_code
    return "\n".join(out), spans


def _restore_math(html_text: str, spans) -> str:
    for index, (is_display, tex) in enumerate(spans):
        token = re.escape(_MATH_PLACEHOLDER.format(index))
        if is_display:
            # Display math alone in a paragraph is a block; elsewhere drop the line breaks around it.
            html_text, n = re.subn(rf'<p>\s*{token}\s*</p>', lambda m: f'<div class="arithmatex">{tex}</div>', html_text)
            if n:
                continue
            token = rf'(?:<br />\s*)?{token}(?:\s*<br />)?'
        html_text = re.sub(token, lambda m: f'<span class="arithmatex">{tex}</span>', html_text)
    return html_text


def _apply_image_layout(html_text: str) -> str:
    """Turn an image's alt layout spec (``![w:60mm right|Caption](img.png)``) into style and class."""

    def repl(match):
        tag = match.group(0)
        alt_match = re.search(r'\balt="([^"]*)"', tag)
        if not alt_match:
            return tag
        layout_spec, caption = parse_image_alt(html.unescape(alt_match.group(1)))
        if not layout_spec:
            return tag
        styles = []
        if "width" in layout_spec:
            styles.append(f"width: {layout_spec['width']};")
        if "height" in layout_spec:
            styles.append(f"height: {layout_spec['height']};")
            if "width" not in layout_spec:
                styles.append("width: auto;")
        attrs = f'alt="{html.escape(caption)}"'
        if "align" in layout_spec:
            attrs += f' class="pexams-img-{layout_spec["align"]}"'
        if styles:
            attrs += f' style="{" ".join(styles)}"'
        return tag[:alt_match.start()] + attrs + tag[alt_match.end():]

    return re.sub(r'<img\b[^>]*>', repl, html_text, flags=re.IGNORECASE)


def _render_markdown(text: str, nl2br: bool = False) -> str:
    """Markdown to HTML for question text, options and the custom header.

    Fenced and inline code, ``$``/``$$`` math (kept intact for MathJax) and images with an
    alt layout spec. ``nl2br`` turns single newlines into line breaks (question text).
    """
    protected, math_spans = _protect_math(text)
    extensions = _MARKDOWN_EXTENSIONS + (['nl2br'] if nl2br else [])
    html_text = markdown.markdown(protected, extensions=extensions).strip()
    return _apply_image_layout(_restore_math(html_text, math_spans))


render_markdown = _render_markdown
resolve_relative_media_uris = _resolve_relative_media_uris

# Inline equivalents of the pexams.css image classes, for HTML shown without that stylesheet.
_ALIGN_STYLES = {
    "left": "float: left; margin: 0 1em 0.5em 0;",
    "right": "float: right; margin: 0 0 0.5em 1em;",
    "center": "display: block; margin: 0.5em auto;",
}


def inline_image_styles(html_text: str) -> str:
    """Replace the ``pexams-img-<align>`` classes by inline styles (Moodle has no pexams.css)."""

    def repl(match):
        tag = match.group(0)
        cls = re.search(r'\sclass="pexams-img-(left|right|center)"', tag)
        if not cls:
            return tag
        tag = tag[:cls.start()] + tag[cls.end():]
        style = _ALIGN_STYLES[cls.group(1)]
        existing = re.search(r'\sstyle="([^"]*)"', tag)
        if existing:
            return tag[:existing.start(1)] + existing.group(1) + " " + style + tag[existing.end(1):]
        return tag.replace("<img", f'<img style="{style}"', 1)

    return re.sub(r'<img\b[^>]*>', repl, html_text, flags=re.IGNORECASE)


def limit_image_size(html_text: str, max_width: Optional[str] = None, max_height: Optional[str] = None) -> str:
    """Add ``max-width``/``max-height`` to every ``<img>`` (the question's image size limits)."""
    limits = (f"max-width: {max_width};" if max_width else "") + (" " if max_width and max_height else "") + \
             (f"max-height: {max_height};" if max_height else "")
    if not limits:
        return html_text

    def repl(match):
        tag = match.group(0)
        existing = re.search(r'\sstyle="([^"]*)"', tag)
        if existing:
            return tag[:existing.end(1)] + " " + limits + tag[existing.end(1):]
        return tag.replace("<img", f'<img style="{limits}"', 1)

    return re.sub(r'<img\b[^>]*>', repl, html_text, flags=re.IGNORECASE)


def question_image_limits(question, default_width_px: Optional[int] = None,
                          default_height_px: Optional[int] = None) -> tuple:
    """A question's image size limits, falling back to the export's defaults in pixels."""
    width = question.max_image_width or (f"{default_width_px}px" if default_width_px else None)
    height = question.max_image_height or (f"{default_height_px}px" if default_height_px else None)
    return width, height


MARKDOWN_IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)\s]+)\)')


def absolute_markdown_images(text: Optional[str], base_dir: Optional[str]) -> Optional[str]:
    """Make the relative paths of the Markdown images in ``text`` absolute file URIs (for files read elsewhere)."""
    if not text or not base_dir:
        return text

    def repl(match):
        path = match.group(2)
        if not _is_relative_media_path(path):
            return match.group(0)
        path = Path(os.path.normpath(os.path.join(os.path.abspath(base_dir), path)))
        return f"![{match.group(1)}]({path.as_uri()})"  # a file URI: no spaces to break the Markdown

    return MARKDOWN_IMAGE_RE.sub(repl, text)


def local_image_path(src: str, base_dir: Optional[str]) -> Optional[str]:
    """The file an ``<img src>`` points to (relative to ``base_dir``, absolute, or file://), if it exists."""
    src = html.unescape(src.strip())
    if src.lower().startswith("file:"):
        from urllib.parse import unquote, urlparse
        path = unquote(urlparse(src).path)
    elif _is_relative_media_path(src):
        path = os.path.join(base_dir or os.getcwd(), src)
    elif os.path.isabs(src):
        path = src
    else:
        return None  # http(s), data: URIs
    path = os.path.normpath(path)
    return path if os.path.isfile(path) else None


def replace_image_sources(html_text: str, base_dir: Optional[str], new_src: Callable[[str], Optional[str]]) -> str:
    """Rewrite each local ``<img src>`` with ``new_src(path)`` (left as is when it returns None)."""

    def repl(match):
        path = local_image_path(match.group(2), base_dir)
        src = new_src(path) if path else None
        return match.group(0) if src is None else f'src={match.group(1)}{src}{match.group(1)}'

    return re.sub(r"\bsrc=(['\"])([^'\"]+)\1", repl, html_text, flags=re.IGNORECASE)


def image_data_uri(path: str) -> Optional[str]:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        return None
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode('ascii')}"
