"""Tests for text-layer extraction.

The PDFs here are built byte by byte, so the tests need no fixture files. They
use a standard Latin font, since embedding an Arabic font in a test is not
practical: these tests cover the wrapper's behaviour, while Arabic accuracy is
established by the calibration against ground truth.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daleel.ingest.extract import page_text, page_texts


def tiny_pdf(pages: list[str]) -> bytes:
    """A minimal valid PDF with one line of Helvetica text per page ("" for blank)."""
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))
    font_id = 3 + 2 * len(pages)
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>",
    ]
    for i, text in enumerate(pages):
        lines = text.split("\n") if text else []
        shows = " ".join(f"({line}) Tj 0 -14 Td" for line in lines)
        stream = f"BT /F1 12 Tf 20 150 Td {shows} ET" if lines else ""
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
            f"/Contents {4 + 2 * i} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    body, offsets = b"%PDF-1.4\n", []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{number} 0 obj\n{obj}\nendobj\n".encode("latin-1")
    xref = len(body)
    body += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    body += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    body += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    body += f"startxref\n{xref}\n%%EOF\n".encode()
    return body


@pytest.fixture
def three_pages(tmp_path: Path) -> Path:
    path = tmp_path / "three.pdf"
    path.write_bytes(tiny_pdf(["first page", "GPA 3.75 (DN)", ""]))
    return path


def test_pages_are_numbered_from_one(three_pages: Path) -> None:
    assert page_text(three_pages, 1) == "first page"


def test_the_requested_page_is_read(three_pages: Path) -> None:
    assert page_text(three_pages, 2) == "GPA 3.75 (DN)"


def test_blank_page_has_empty_text(three_pages: Path) -> None:
    assert page_text(three_pages, 3) == ""


def test_page_zero_is_an_error(three_pages: Path) -> None:
    with pytest.raises(ValueError, match="numbered from 1"):
        page_text(three_pages, 0)


def test_page_past_the_end_is_an_error(three_pages: Path) -> None:
    with pytest.raises(IndexError, match="has 3 pages"):
        page_text(three_pages, 4)


def test_every_page_in_order(three_pages: Path) -> None:
    assert page_texts(three_pages) == ["first page", "GPA 3.75 (DN)", ""]


def test_line_breaks_are_newlines(tmp_path: Path) -> None:
    path = tmp_path / "two_lines.pdf"
    path.write_bytes(tiny_pdf(["upper line\nlower line"]))
    text = page_text(path, 1)
    assert "\r" not in text
    assert text.split("\n") == ["upper line", "lower line"]
