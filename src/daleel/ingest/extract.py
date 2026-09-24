"""Extract a PDF's text layer, page by page, with pypdfium2.

The backend was chosen by measurement against hand-transcribed ground truth.
pdfplumber writes Arabic in visual order, which reverses every word, and its
right-to-left option fixes the words by reversing every number instead.
pypdfium2 gets both right: 99 to 100% of words on sound pages, and every date
on the calendar intact. It also returns base letters where pdfplumber and
pdftotext return presentation forms. pdfplumber remains the tool for
character geometry.

pdfplumber stays available as a second backend, for comparison: the inventory
can measure a corpus through either one, and the difference between them is
itself one of this project's findings.

PDFium is a C library, so every handle opened here is closed explicitly rather
than left for the garbage collector.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pypdfium2

# The default comes first: the gate was calibrated on it.
BACKENDS = ("pypdfium2", "pdfplumber")

# Called with (page number, page count) after each page, for progress reports.
PageCallback = Callable[[int, int], None]


def _text_of(document: pypdfium2.PdfDocument, index: int) -> str:
    page = document[index]
    textpage = page.get_textpage()
    try:
        # PDFium ends lines with \r\n; the rest of the project uses \n.
        return textpage.get_text_range().replace("\r\n", "\n")
    finally:
        textpage.close()
        page.close()


def page_text(path: Path, page: int) -> str:
    """The text layer of one page, numbered from 1 like the annotation guidelines."""
    if page < 1:
        raise ValueError(f"pages are numbered from 1, not {page}")
    document = pypdfium2.PdfDocument(path)
    try:
        if page > len(document):
            raise IndexError(f"{path} has {len(document)} pages, so there is no page {page}")
        return _text_of(document, page - 1)
    finally:
        document.close()


def _read_pages(
    count: int,
    read: Callable[[int], str],
    errors: list[str] | None,
    on_page: PageCallback | None,
) -> list[str]:
    texts = []
    for index in range(count):
        try:
            texts.append(read(index))
        except Exception as exc:
            # With an error list, one unreadable page must not abort the
            # document: a page that cannot be read is itself a finding.
            if errors is None:
                raise
            errors.append(f"page {index + 1}: {type(exc).__name__}: {exc}")
            texts.append("")
        if on_page is not None:
            on_page(index + 1, count)
    return texts


def page_texts(
    path: Path,
    backend: str = "pypdfium2",
    *,
    errors: list[str] | None = None,
    on_page: PageCallback | None = None,
) -> list[str]:
    """Every page's text layer, in page order.

    `backend` is "pypdfium2", the default, or "pdfplumber", kept for
    comparison. With `errors`, a page that cannot be read is recorded there
    and read as empty instead of aborting the document.
    """
    if backend == "pypdfium2":
        document = pypdfium2.PdfDocument(path)
        try:
            return _read_pages(
                len(document), lambda index: _text_of(document, index), errors, on_page
            )
        finally:
            document.close()

    if backend == "pdfplumber":
        # Imported here so that code using only pypdfium2 never pays for it.
        import pdfplumber

        with pdfplumber.open(path) as pdf:

            def read(index: int) -> str:
                page = pdf.pages[index]
                try:
                    return page.extract_text() or ""
                finally:
                    # pdfplumber caches parsed objects per page. Without this a
                    # long document holds every page in memory at once.
                    page.flush_cache()

            return _read_pages(len(pdf.pages), read, errors, on_page)

    raise ValueError(f"unknown backend {backend!r}; choose from {', '.join(BACKENDS)}")
