"""Render PDF pages to images for OCR.

What an engine needs is enough pixels per line of text, so a page is rendered
at a resolution in dots per inch of the size it is printed at. That is usually
the size its PDF declares. The organizational regulations are the exception:
every page is declared as 59.5 x 84.2 points, A4 drawn at a tenth of its size,
with no UserUnit entry to say so. At 300 DPI of that size a page would be 248
pixels wide, and a renderer asked for A4's pixels instead stamps the image
with 3,000 DPI, which Tesseract rejects as invalid and replaces with a guess.

So a page declared smaller than any real document is treated as the A4 page it
draws, and every image carries the resolution it was rendered at.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pypdfium2
from PIL import Image

POINTS_PER_INCH = 72
A4_LONG_EDGE = 841.89  # points
# No real document is under two inches on its long edge. A page declared that
# small is a drawing of a larger one, as in organizational_regulations.pdf.
SMALLEST_REAL_PAGE = 2 * POINTS_PER_INCH

# The version of how render_page draws a page. The OCR cache keys on it with
# pypdfium2's version and the resolution, so a change to render_page that
# changes any page's pixels must raise it, or pages would be read back that
# were read from other pixels.
RENDERING = 1


def printed_scale(width: float, height: float) -> float:
    """How many times larger than its declared size, in points, a page prints."""
    long_edge = max(width, height)
    if long_edge >= SMALLEST_REAL_PAGE:
        return 1.0
    return A4_LONG_EDGE / long_edge


@dataclass(frozen=True)
class RenderedPage:
    """A page as an engine will see it, and the resolution it was rendered at."""

    image: Image.Image
    dpi: int
    # Width and height as the PDF declares them, in points: the page's own
    # coordinates, which pdfplumber measures its characters in.
    page_size: tuple[float, float]

    def png(self) -> bytes:
        """The image as PNG, stamped with its resolution so no engine has to guess it."""
        buffer = io.BytesIO()
        self.image.save(buffer, format="PNG", dpi=(self.dpi, self.dpi))
        return buffer.getvalue()


def render_page(path: Path, page: int, *, dpi: int = 300) -> RenderedPage:
    """Render one page, numbered from 1, at dpi of its printed size."""
    if page < 1:
        raise ValueError(f"pages are numbered from 1, not {page}")
    if dpi < 1:
        raise ValueError(f"a resolution must be positive, not {dpi}")
    document = pypdfium2.PdfDocument(path)
    try:
        if page > len(document):
            raise IndexError(f"{path} has {len(document)} pages, so there is no page {page}")
        pdf_page = document[page - 1]
        try:
            width, height = pdf_page.get_size()
            bitmap = pdf_page.render(scale=dpi / POINTS_PER_INCH * printed_scale(width, height))
            try:
                # For some pixel formats the image shares the bitmap's memory,
                # so it is copied before PDFium frees the bitmap.
                image = bitmap.to_pil().copy()
            finally:
                bitmap.close()
        finally:
            pdf_page.close()
    finally:
        document.close()
    return RenderedPage(image=image, dpi=dpi, page_size=(width, height))
