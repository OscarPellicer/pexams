"""HTML to PDF with headless Chromium (Playwright), and the <head> assets of the HTML reports.

Self-contained (standard library and Playwright only) so that pexams and pevaluate can share
this file byte for byte.
"""
import html
import os
from typing import Callable, Optional

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

__all__ = ["PlaywrightError", "head_assets", "html_document", "file_url", "html_to_pdf",
           "OPEN_SANS_URL", "MATHJAX_URL", "MERMAID_URL", "MATHJAX_ARITHMATEX_CONFIG"]

OPEN_SANS_URL = "https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,300..800;1,300..800&display=swap"
MATHJAX_URL = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
MERMAID_URL = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs"

# Typeset only the math spans written by Python-Markdown's arithmatex extension (class
# "arithmatex"): a stray '$' anywhere else stays literal.
MATHJAX_ARITHMATEX_CONFIG = """window.MathJax = {
    tex: { inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] },
    options: { ignoreHtmlClass: '.*|', processHtmlClass: 'arithmatex' }
};"""


def head_assets(fonts_url: Optional[str] = OPEN_SANS_URL, mathjax: bool = True,
                mathjax_config: Optional[str] = None, mermaid: bool = False) -> str:
    """The <link>/<script> tags of a report: web fonts, MathJax (optionally configured) and Mermaid."""
    tags = []
    if fonts_url:
        tags.append(f'<link rel="stylesheet" href="{fonts_url}">')
    if mathjax and mathjax_config:
        tags.append(f"<script>\n{mathjax_config}\n</script>")
    if mathjax:
        tags.append(f'<script id="MathJax-script" async src="{MATHJAX_URL}"></script>')
    if mermaid:
        tags.append(f"<script type=\"module\">import mermaid from '{MERMAID_URL}'; "
                    "mermaid.initialize({startOnLoad: true}); window.mermaid = mermaid;</script>")
    return "\n".join(tags)


def html_document(body: str, title: Optional[str] = None, css: str = "", **assets) -> str:
    """A complete HTML page: ``<head>`` with ``head_assets(**assets)`` and ``css``, then ``body``."""
    title_tag = f"<title>{html.escape(title)}</title>\n" if title is not None else ""
    return (f'<!DOCTYPE html>\n<html>\n<head>\n<meta charset="UTF-8">\n{title_tag}{head_assets(**assets)}\n'
            f"<style>{css}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n")


def file_url(path: str) -> str:
    """``file:///`` URL of a local path (also on Windows)."""
    return "file:///" + os.path.abspath(path).replace("\\", "/")


def html_to_pdf(html_path: str, pdf_path: str, *, typeset_math: bool = True, math_required: bool = True,
                settle_ms: int = 1000, before_pdf: Optional[Callable] = None, **pdf_options):
    """Render a local HTML file to PDF with headless Chromium.

    The page is loaded (``networkidle``), its MathJax math typeset (``typeset_math``; with
    ``math_required=False`` a missing or failing MathJax is ignored), and after ``settle_ms``
    ``before_pdf(page)`` may lay it out further. ``pdf_options`` go to ``page.pdf`` (default
    ``format="A4"``, ``print_background=True``). Returns what ``before_pdf`` returned.
    Playwright errors propagate.
    """
    pdf_options = {"format": "A4", "print_background": True, **pdf_options}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(file_url(html_path), wait_until="networkidle")
            if typeset_math:
                try:
                    page.evaluate("() => MathJax.typesetPromise()")
                except Exception:
                    if math_required:
                        raise
            page.wait_for_timeout(settle_ms)
            result = before_pdf(page) if before_pdf is not None else None
            page.pdf(path=pdf_path, **pdf_options)
            return result
        finally:
            browser.close()
