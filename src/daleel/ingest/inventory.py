"""Measure what a PDF's text layer actually contains.

This corpus has three distinct extraction failure modes and none of them
announce themselves, so the decision about which extraction path to use -- and
whether to trust a text layer at all -- has to rest on numbers rather than on
glancing at a couple of pages.

This module only reports. It deliberately passes no judgement: thresholds and
pass/fail verdicts belong to the quality gate, and setting those thresholds
before looking at the measurements would mean calibrating against assumptions
instead of against the corpus.

The character counting is kept separate from the PDF reading. `analyse_text`
is a pure function over a string, which makes the part that matters unit
testable without needing a PDF fixture.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

from daleel.ingest.extract import PageCallback, page_texts
from daleel.ingest.metadata import metadata_from
from daleel.normalize.arabic import (
    BIDI_CONTROLS,
    DIACRITIC_RANGES,
    PRESENTATION_FORM_RANGES,
    TATWEEL,
    in_ranges,
)

# Base Arabic letters, and the extended set used for non-Arabic languages.
ARABIC_LETTER_RANGES = ((0x0620, 0x064A), (0x0671, 0x06D3))

ARABIC_INDIC_DIGIT_RANGES = ((0x0660, 0x0669), (0x06F0, 0x06F9))

REPLACEMENT_CHAR = 0xFFFD

# Layout, not corruption: tab, newline, carriage return, and form feed,
# which pdftotext writes after every page as a page separator.
BENIGN_C0 = frozenset({0x09, 0x0A, 0x0C, 0x0D})


@dataclass
class CharStats:
    """Counts of the character classes that distinguish good text from damage."""

    chars: int = 0
    presentation_forms: int = 0
    arabic_letters: int = 0
    arabic_diacritics: int = 0
    tatweel: int = 0
    arabic_indic_digits: int = 0
    bidi_controls: int = 0
    c0_controls: int = 0
    replacement_chars: int = 0
    latin_letters: int = 0
    ascii_digits: int = 0

    @property
    def arabic_total(self) -> int:
        """Every character that is Arabic script, however it was encoded."""
        return self.presentation_forms + self.arabic_letters

    @property
    def presform_ratio(self) -> float:
        """Share of Arabic stored as display forms. High means failure mode A."""
        total = self.arabic_total
        return self.presentation_forms / total if total else 0.0

    @property
    def bidi_per_1k(self) -> float:
        return 1000 * self.bidi_controls / self.chars if self.chars else 0.0

    @property
    def c0_per_1k(self) -> float:
        """Control characters where letters belong. High means failure mode C."""
        return 1000 * self.c0_controls / self.chars if self.chars else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "chars": self.chars,
            "presentation_forms": self.presentation_forms,
            "arabic_letters": self.arabic_letters,
            "arabic_total": self.arabic_total,
            "arabic_diacritics": self.arabic_diacritics,
            "tatweel": self.tatweel,
            "arabic_indic_digits": self.arabic_indic_digits,
            "bidi_controls": self.bidi_controls,
            "c0_controls": self.c0_controls,
            "replacement_chars": self.replacement_chars,
            "latin_letters": self.latin_letters,
            "ascii_digits": self.ascii_digits,
            "presform_ratio": round(self.presform_ratio, 4),
            "bidi_per_1k": round(self.bidi_per_1k, 2),
            "c0_per_1k": round(self.c0_per_1k, 2),
        }


def analyse_text(text: str) -> CharStats:
    """Count character classes in extracted text.

    Pure function, no I/O. Every interesting property of the corpus is
    measurable through this, which is why it is tested directly.
    """
    stats = CharStats(chars=len(text))

    for char in text:
        codepoint = ord(char)

        if char == TATWEEL:
            stats.tatweel += 1
        elif codepoint == REPLACEMENT_CHAR:
            stats.replacement_chars += 1
        elif codepoint in BIDI_CONTROLS:
            stats.bidi_controls += 1
        elif in_ranges(codepoint, PRESENTATION_FORM_RANGES):
            stats.presentation_forms += 1
        elif in_ranges(codepoint, DIACRITIC_RANGES):
            stats.arabic_diacritics += 1
        elif in_ranges(codepoint, ARABIC_INDIC_DIGIT_RANGES):
            stats.arabic_indic_digits += 1
        elif in_ranges(codepoint, ARABIC_LETTER_RANGES):
            stats.arabic_letters += 1
        elif codepoint < 0x20 and codepoint not in BENIGN_C0:
            stats.c0_controls += 1
        elif char.isascii() and char.isalpha():
            stats.latin_letters += 1
        elif char.isascii() and char.isdigit():
            stats.ascii_digits += 1

    return stats


@dataclass
class PdfInventory:
    """What one document's text layer looks like, measured."""

    path: Path
    pages: int
    producer: str
    creator: str
    stats: CharStats
    fonts: int = 0
    font_sample_pages: int = 0
    errors: list[str] = field(default_factory=list)
    backend: str = "pypdfium2"

    @property
    def chars_per_page(self) -> float:
        """Low values mean the page is mostly graphics, so OCR is needed
        regardless of whether the text that did extract is sound."""
        return self.stats.chars / self.pages if self.pages else 0.0

    def to_dict(self) -> dict:
        return {
            "file": self.path.name,
            "backend": self.backend,
            "pages": self.pages,
            "producer": self.producer,
            "creator": self.creator,
            "fonts_sampled": self.fonts,
            "font_sample_pages": self.font_sample_pages,
            "chars_per_page": round(self.chars_per_page, 1),
            "errors": self.errors,
            **self.stats.to_dict(),
        }


def _progress(path: Path) -> PageCallback:
    def report(number: int, count: int) -> None:
        print(f"\r  {path.name}: page {number}/{count}", end="", file=sys.stderr, flush=True)

    return report


def inspect_pdf(
    path: Path,
    font_sample_pages: int = 5,
    progress: bool = False,
    backend: str = "pypdfium2",
) -> PdfInventory:
    """Extract a PDF's text layer and measure it.

    Text comes from `backend`, as daleel.ingest.extract reads it. Metadata and
    font names always come from pdfplumber, which remains the tool for a PDF's
    structure. Font names are collected from the first `font_sample_pages`
    pages only: reading `page.chars` materialises a dict per glyph, which on an
    86-page document costs hundreds of megabytes for a diagnostic that a
    handful of pages answers just as well. The sample size is recorded so the
    number is never mistaken for a full count.
    """
    errors: list[str] = []
    report = _progress(path) if progress and sys.stderr.isatty() else None
    texts = page_texts(path, backend, errors=errors, on_page=report)

    font_names: set[str] = set()
    with pdfplumber.open(path) as pdf:
        meta = metadata_from(pdf)
        page_count = len(pdf.pages)
        for number, page in enumerate(pdf.pages[:font_sample_pages], start=1):
            try:
                font_names.update(char["fontname"] for char in page.chars if "fontname" in char)
            except Exception as exc:  # unreadable fonts are a finding, not a crash
                errors.append(f"page {number} fonts: {type(exc).__name__}: {exc}")
            finally:
                page.flush_cache()

    if report is not None:
        print(f"\r  {path.name}: {page_count} pages read", file=sys.stderr)

    return PdfInventory(
        path=path,
        pages=page_count,
        producer=meta.producer,
        creator=meta.creator,
        stats=analyse_text("\n".join(texts)),
        fonts=len(font_names),
        font_sample_pages=min(font_sample_pages, page_count),
        errors=errors,
        backend=backend,
    )


def inventory_dir(
    directory: Path, progress: bool = False, backend: str = "pypdfium2"
) -> list[PdfInventory]:
    """Inspect every PDF in a directory, in a stable order."""
    pdfs = sorted(p for p in directory.glob("*.pdf") if p.is_file())
    return [inspect_pdf(path, progress=progress, backend=backend) for path in pdfs]


_COLUMNS: tuple[tuple[str, str], ...] = (
    ("file", "file"),
    ("pages", "pages"),
    ("producer", "producer"),
    ("chars", "chars"),
    ("chars_per_page", "ch/page"),
    ("presform_ratio", "presform"),
    ("bidi_per_1k", "bidi/1k"),
    ("c0_per_1k", "ctrl/1k"),
    ("replacement_chars", "U+FFFD"),
)


def format_table(inventories: list[PdfInventory]) -> str:
    """Render the inventory as an aligned plain-text table."""
    if not inventories:
        return "No PDFs found."

    rows = [inv.to_dict() for inv in inventories]
    headers = [label for _, label in _COLUMNS]

    cells: list[list[str]] = []
    for row in rows:
        line = []
        for key, _ in _COLUMNS:
            value = row[key]
            if key == "presform_ratio":
                line.append(f"{value:.0%}")
            elif isinstance(value, float):
                line.append(f"{value:.1f}")
            else:
                line.append(str(value))
        cells.append(line)

    widths = [max(len(headers[i]), *(len(line[i]) for line in cells)) for i in range(len(headers))]
    divider = "  ".join("-" * width for width in widths)

    out = ["  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)), divider]
    out.extend("  ".join(c.ljust(w) for c, w in zip(line, widths, strict=True)) for line in cells)

    # The same PDF measures very differently through different extractors.
    backends = ", ".join(sorted({inv.backend for inv in inventories}))
    out.append("")
    out.append(f"Text layers read with {backends}.")

    notes = [f"{inv.path.name}: {err}" for inv in inventories for err in inv.errors]
    if notes:
        out.append("")
        out.append("Page errors:")
        out.extend(f"  {note}" for note in notes)

    return "\n".join(out)


def to_json(inventories: list[PdfInventory]) -> str:
    return json.dumps([inv.to_dict() for inv in inventories], indent=2, ensure_ascii=False)
