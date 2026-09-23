"""Extract a PDF's text layer, page by page, with pypdfium2.

The backend was chosen by measurement against hand-transcribed ground truth.
pdfplumber writes Arabic in visual order, which reverses every word, and its
right-to-left option fixes the words by reversing every number instead.
pypdfium2 gets both right: 99 to 100% of words on sound pages, and every date
on the calendar intact. It also returns base letters where pdfplumber and
pdftotext return presentation forms. pdfplumber remains the tool for
character geometry.

PDFium is a C library, so every handle opened here is closed explicitly rather
than left for the garbage collector.
"""

from __future__ import annotations

from pathlib import Path

import pypdfium2


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


def page_texts(path: Path) -> list[str]:
    """Every page's text layer, in page order."""
    document = pypdfium2.PdfDocument(path)
    try:
        return [_text_of(document, index) for index in range(len(document))]
    finally:
        document.close()
