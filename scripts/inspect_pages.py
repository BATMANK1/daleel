#!/usr/bin/env python3
"""Inspect chosen pages of a PDF through both text backends.

    python3 scripts/inspect_pages.py <pdf> <page> [<page> ...]

For each page, prints what the page is made of (characters, images and vector
curves, as pdfplumber sees its structure) and how much Arabic each backend
extracts from it, counting both letters and words. Letters show whether a
backend loses text; words show whether it groups the text correctly, since
the same letters split into more fragments make more words.

pdfplumber groups characters into lines and words with fixed tolerances, sized
for standard pages. Some PDFs define their pages in scaled-down units, so the
script also reruns pdfplumber with its tolerances scaled to the page's width
relative to A4. That is how the organizational regulations' word counts were
reconciled.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pdfplumber

from daleel.ingest.extract import page_text
from daleel.ingest.quality import arabic_tokens
from daleel.normalize.arabic import for_comparison

A4_WIDTH = 595.276  # points
PDFPLUMBER_TOLERANCE = 3  # pdfplumber's default x_tolerance and y_tolerance
ARABIC_LETTER = re.compile(r"[ء-غف-ي]")


def letters(text: str) -> int:
    return len(ARABIC_LETTER.findall(for_comparison(text)))


def words(text: str) -> int:
    return len(arabic_tokens(for_comparison(text)))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 2 or not all(a.isdigit() for a in args[1:]):
        print(__doc__, file=sys.stderr)
        return 2
    path, pages = Path(args[0]), [int(a) for a in args[1:]]
    if not path.is_file():
        print(f"error: {path} is not a file", file=sys.stderr)
        return 2

    with pdfplumber.open(path) as pdf:
        first = pdf.pages[0]
        scale = first.width / A4_WIDTH
        tolerance = PDFPLUMBER_TOLERANCE * scale
        print(
            f"{path.name}: {len(pdf.pages)} pages of {first.width:.1f} x {first.height:.1f} "
            f"units, {scale:.2f} of A4 width; pdfplumber rerun with tolerance "
            f"{tolerance:.2f} instead of {PDFPLUMBER_TOLERANCE}"
        )
        print()
        print(f"{'':>4}   {'structure':^21}   {'Arabic letters':^17}   {'Arabic words':^25}")
        print(
            f"{'page':>4}   {'chars':>6} {'images':>6} {'curves':>7}   "
            f"{'pdfium':>8} {'plumber':>8}   {'pdfium':>8} {'plumber':>8} {'scaled':>7}"
        )
        for number in pages:
            if not 1 <= number <= len(pdf.pages):
                print(f"{number:>4}   no such page")
                continue
            page = pdf.pages[number - 1]
            pdfium = page_text(path, number)
            default = page.extract_text() or ""
            scaled = page.extract_text(x_tolerance=tolerance, y_tolerance=tolerance) or ""
            print(
                f"{number:>4}   {len(page.chars):>6} {len(page.images):>6} {len(page.curves):>7}   "
                f"{letters(pdfium):>8} {letters(default):>8}   "
                f"{words(pdfium):>8} {words(default):>8} {words(scaled):>7}"
            )
            page.flush_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
