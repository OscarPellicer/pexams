"""Browser-side layout of the exam pages (run in Playwright before printing): questions split
into page-sized containers, running header/footer on every sheet, and the measured positions of
the open-answer boxes."""
from typing import List

from pexams.schemas import PexamQuestion

PX_PER_MM_CSS = 96 / 25.4
PRINT_CONTENT_WIDTH_MM = 180  # A4 width (210mm) minus 15mm margins
PRINT_CONTENT_HEIGHT_MM = 267  # A4 height (297mm) minus 15mm margins


def wrap_pages_with_header_footer(page, header_text: str, footer_prefix: str) -> int:
    """Wraps each page container in an A4 sheet (15mm margins) with its own header and footer."""
    return page.evaluate(
        """([headerText, footerPrefix]) => {
            const style = document.createElement('style');
            style.textContent = `
                @page { size: A4; margin: 0; }
                body { margin: 0; }
                .pexams-sheet { position: relative; width: 210mm; height: 297mm; padding: 15mm;
                                box-sizing: border-box; overflow: hidden; break-after: page; }
                .pexams-sheet.last { break-after: auto; }
                .pexams-sheet > .page-container { break-after: auto !important; page-break-after: auto !important; }
                .pexams-running { position: absolute; left: 15mm; right: 15mm; text-align: center;
                                  font-size: 9px; line-height: 1.2; color: #888; white-space: nowrap;
                                  overflow: hidden; text-overflow: ellipsis; }`;
            document.head.appendChild(style);
            const containers = Array.from(document.body.children).filter((el) => el.classList.contains('page-container'));
            containers.forEach((container, index) => {
                const sheet = document.createElement('div');
                sheet.className = 'pexams-sheet' + (index === containers.length - 1 ? ' last' : '');
                container.parentNode.insertBefore(sheet, container);
                sheet.appendChild(container);
                for (const [edge, text] of [['top', headerText], ['bottom', `${footerPrefix} - Page ${index + 1} of ${containers.length}`]]) {
                    const running = document.createElement('div');
                    running.className = 'pexams-running';
                    running.style[edge] = '6mm';
                    running.textContent = text;
                    sheet.appendChild(running);
                }
            });
            return containers.length;
        }""",
        [header_text, footer_prefix],
    )


def keep_questions_within_pages(page, fill_answer_space: bool = False) -> int:
    """Splits the questions into one fixed-height container per printed page.

    A single tall container is fragmented by Chromium's print engine in ways that
    getBoundingClientRect cannot report, so boxes ended up split across pages and the
    open-answer coordinates drifted. Instead, questions are distributed over page-sized
    containers (like the answer sheet): no question or open-answer box crosses a page,
    every page carries its own four fiducial markers, and measured coordinates are the
    printed ones. Each container gets the height left on its page, so with several columns
    a question that does not fit shows up below the page or in an extra column to the
    right, and is moved to the next page.

    With ``fill_answer_space`` the free space left at the bottom of each page is shared
    equally among that page's open-answer boxes (except those with an explicit height).
    Returns the number of question pages.
    """
    return page.evaluate(
        """([pageHeightMm, bottomReserveMm, fillAnswerSpace, fillMarginMm]) => {
            const mm = 96 / 25.4;
            const limit = (pageHeightMm - bottomReserveMm) * mm;
            const original = document.querySelector('.questions-page');
            if (!original) return 0;
            const originalContent = original.querySelector('.questions-container');
            const items = Array.from(originalContent.children);
            // The student header and its spacer (exams without answer sheet) belong to the first page only.
            const firstPageHeader = original.querySelector('.first-page-header');
            const firstPageSpacer = original.querySelector('.first-page-header-spacer');
            if (firstPageHeader) firstPageHeader.remove();
            if (firstPageSpacer) firstPageSpacer.remove();
            const template = original.cloneNode(true);
            template.querySelector('.questions-container').innerHTML = '';
            const pages = [];
            const usableBottom = (pageEl) => pageEl.getBoundingClientRect().top + limit;
            const overflows = (pageEl, container) => {
                const bottom = usableBottom(pageEl) + 1;
                const right = container.getBoundingClientRect().right + 1;
                return Array.from(container.children).some((child) => {
                    const rect = child.getBoundingClientRect();
                    return rect.bottom > bottom || rect.right > right;
                });
            };
            const newPage = () => {
                const pageEl = template.cloneNode(true);
                pageEl.style.height = pageHeightMm + 'mm';
                pageEl.style.minHeight = '0';
                pageEl.style.overflow = 'hidden';
                pageEl.style.pageBreakAfter = 'always';
                original.parentNode.insertBefore(pageEl, original);
                pages.push(pageEl);
                const container = pageEl.querySelector('.questions-container');
                if (pages.length === 1 && firstPageSpacer) pageEl.insertBefore(firstPageSpacer, container);
                const height = usableBottom(pageEl) - container.getBoundingClientRect().top;
                container.style.height = Math.max(0, height) + 'px';
                return container;
            };
            let current = newPage();
            if (firstPageHeader) pages[0].appendChild(firstPageHeader);
            for (const item of items) {
                current.appendChild(item);
                // Check every item: adding one can rebalance the columns of the others.
                if (current.children.length > 1 && overflows(pages[pages.length - 1], current)) {
                    current = newPage();
                    current.appendChild(item);
                }
            }
            original.remove();
            pages[pages.length - 1].style.pageBreakAfter = 'auto';

            if (fillAnswerSpace) {
                for (const pageEl of pages) {
                    const container = pageEl.querySelector('.questions-container');
                    const boxes = Array.from(container.querySelectorAll('.open-answer-box:not([data-fixed-height])'));
                    if (!boxes.length) continue;
                    const contentBottom = Math.max(...Array.from(container.children).map((c) => c.getBoundingClientRect().bottom));
                    const free = usableBottom(pageEl) - fillMarginMm * mm - contentBottom;
                    if (free <= 0) continue;
                    const heights = boxes.map((box) => box.getBoundingClientRect().height);
                    const previous = boxes.map((box) => box.style.minHeight);
                    boxes.forEach((box, i) => { box.style.minHeight = (heights[i] + free / boxes.length) + 'px'; });
                    if (overflows(pageEl, container)) boxes.forEach((box, i) => { box.style.minHeight = previous[i]; });
                }
            }
            return pages.length;
        }""",
        [PRINT_CONTENT_HEIGHT_MM, 14, fill_answer_space, 2],
    )


def extract_open_answer_area_metadata(page, model_questions: List[PexamQuestion], exam_model: int) -> List[dict]:
    """Read rendered open-answer box positions from the browser page."""
    question_lookup = {str(q.id): q for q in model_questions}
    raw_boxes = page.evaluate(
        """() => {
            const pxPerMm = 96 / 25.4;
            const containers = Array.from(document.querySelectorAll('.page-container'));
            return Array.from(document.querySelectorAll('.open-answer-box')).map((box) => {
                const container = containers.find((candidate) => candidate.contains(box));
                const boxRect = box.getBoundingClientRect();
                const containerRect = container.getBoundingClientRect();
                return {
                    question_id: box.dataset.questionId,
                    page_index: containers.indexOf(container) + 1,
                    x_mm: (boxRect.left - containerRect.left) / pxPerMm,
                    y_mm: (boxRect.top - containerRect.top) / pxPerMm,
                    width_mm: boxRect.width / pxPerMm,
                    height_mm: boxRect.height / pxPerMm,
                    lines: Number(box.dataset.lines || 0),
                };
            });
        }"""
    )

    metadata = []
    for box in raw_boxes:
        question = question_lookup.get(str(box["question_id"]))
        metadata.append({
            "model_id": str(exam_model),
            "question_id": box["question_id"],
            "original_id": str(question.original_id) if question and question.original_id is not None else None,
            # y is measured from the top of a (possibly multi-page) container: convert it to
            # the printed page and the offset within that page.
            "page_index": int(box["page_index"]) + int(float(box["y_mm"]) // PRINT_CONTENT_HEIGHT_MM),
            "x_mm": round(float(box["x_mm"]), 3),
            "y_mm": round(float(box["y_mm"]) % PRINT_CONTENT_HEIGHT_MM, 3),
            "width_mm": round(float(box["width_mm"]), 3),
            "height_mm": round(float(box["height_mm"]), 3),
            "lines": float(box["lines"]),
            "points": float(question.points) if question else None,
        })
    return metadata
