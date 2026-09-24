"""Fixtures shared across test files.

pytest makes every fixture defined here available to every test file, without
an import.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest


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
def make_pdf(tmp_path: Path) -> Callable[..., Path]:
    """Write a small PDF with one line of Latin text per page, and return its path."""

    def make(pages: list[str], name: str = "doc.pdf") -> Path:
        path = tmp_path / name
        path.write_bytes(tiny_pdf(pages))
        return path

    return make
