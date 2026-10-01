"""Extract a PDF's text layer, page by page, with pypdfium2.

The backend was chosen by measurement against hand-transcribed ground truth.
pdfplumber writes Arabic in visual order, which reverses every word, and its
right-to-left option fixes the words by reversing every number instead.
pypdfium2 gets both right: 99 to 100% of words on sound pages, and every date
on the calendar intact. It also returns base letters where pdfplumber and
pdftotext return presentation forms, and gives every character's box, from
which the calendar's grid is rebuilt (daleel.ingest.calendar).

Not every PDFium build reads Arabic in order. pypdfium2 5.6.0, 5.7.1 and 5.12.1
read the whole corpus identically, but 5.13.0, with PDFium 153.0.7999.0, puts
the words of an Arabic line in the order they are drawn, last word first. It
still spells each word right, so no measure of words notices. pyproject.toml
holds pypdfium2 below 5.13, and before any page is read, a generated Arabic
line is read to check that the installed build keeps its words in order.

pdfplumber stays available as a second backend, for comparison: the inventory
can measure a corpus through either one, and the difference between them is
itself one of this project's findings.

PDFium is a C library, so every handle opened here is closed explicitly rather
than left for the garbage collector.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path

import pypdfium2

# The default comes first: the gate was calibrated on it.
BACKENDS = ("pypdfium2", "pdfplumber")

# Called with (page number, page count) after each page, for progress reports.
PageCallback = Callable[[int, int], None]

# Three Arabic words, in the order they are read: what the check must read back.
ORDER_PROBE = "نهاية فترة حذف"


class UnsupportedPdfiumError(RuntimeError):
    """The installed PDFium puts Arabic words out of order, so no text layer can be read."""


def arabic_line_pdf(line: str) -> bytes:
    """A one-page PDF whose text layer holds `line`, a line of Arabic letters and
    spaces, drawn from its left end as Adobe's software draws Arabic.

    Each letter is drawn as a box, in a Type3 font whose ToUnicode map names the
    letter, so no Arabic font is needed: only the text layer matters here.
    """
    letters = sorted(set(line))
    code = {letter: number for number, letter in enumerate(letters, start=1)}
    drawn = "".join(f"{code[letter]:02X}" for letter in reversed(line))
    content = f"BT /F1 12 Tf 20 150 Td <{drawn}> Tj ET"
    names = " ".join("/space" if letter == " " else "/box" for letter in letters)
    widths = " ".join("250" if letter == " " else "500" for letter in letters)
    mappings = "\n".join(f"<{code[letter]:02X}> <{ord(letter):04X}>" for letter in letters)
    to_unicode = (
        "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /Adobe-Identity-UCS def /CMapType 2 def\n"
        "1 begincodespacerange <00> <FF> endcodespacerange\n"
        f"{len(letters)} beginbfchar\n{mappings}\nendbfchar\n"
        "endcmap CMapName currentdict /CMap defineresource pop end end"
    )
    box = "500 0 0 0 450 700 d1 0 0 450 700 re f"
    space = "250 0 0 0 0 0 d1"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type3 /FontBBox [0 0 500 700] "
        "/FontMatrix [0.001 0 0 0.001 0 0] /CharProcs << /box 6 0 R /space 7 0 R >> "
        f"/Encoding << /Type /Encoding /Differences [1 {names}] >> "
        f"/FirstChar 1 /LastChar {len(letters)} /Widths [{widths}] /ToUnicode 8 0 R >>",
        f"<< /Length {len(box)} >>\nstream\n{box}\nendstream",
        f"<< /Length {len(space)} >>\nstream\n{space}\nendstream",
        f"<< /Length {len(to_unicode)} >>\nstream\n{to_unicode}\nendstream",
    ]
    body, offsets = b"%PDF-1.4\n", []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{number} 0 obj\n{obj}\nendobj\n".encode("ascii")
    xref = len(body)
    body += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    body += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    body += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    body += f"startxref\n{xref}\n%%EOF\n".encode()
    return body


def _read_probe() -> str:
    """What the installed PDFium reads from a generated line of ORDER_PROBE."""
    document = pypdfium2.PdfDocument(arabic_line_pdf(ORDER_PROBE))
    try:
        page = document[0]
        textpage = page.get_textpage()
        try:
            return textpage.get_text_range()
        finally:
            textpage.close()
            page.close()
    finally:
        document.close()


@functools.cache
def check_reading_order() -> None:
    """Raise UnsupportedPdfiumError if the installed PDFium reads an Arabic line's words
    out of order. It reads a generated line once, before the first real page."""
    read = _read_probe()
    if read != ORDER_PROBE:
        raise UnsupportedPdfiumError(
            f"pypdfium2 {pypdfium2.version.PYPDFIUM_INFO} with PDFium "
            f"{pypdfium2.version.PDFIUM_INFO} reads the Arabic line {ORDER_PROBE!r} "
            f"as {read!r}, so it would put the words of every Arabic line out of order. "
            'Install the pypdfium2 this project holds to: uv pip install -e ".[dev]"'
        )


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
    check_reading_order()
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
        check_reading_order()
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
