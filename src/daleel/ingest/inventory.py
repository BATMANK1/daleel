# Measure what a PDF's text layer actually contains

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

"""Arabic Presentation Forms-A and -B hold the contextual glyph variants a
renderer selects per letter position. A text layer emitting these has stored
display forms rather than characters, so a query typed in base letters can
never match it different codepoints entirely."""

PRESENTATION_FORM_RANGES = ((0xFB50, 0xFDFF), (0xFE70, 0xFEFF))

# Base Arabic letters, and the extended set used for non-Arabic languages.
ARABIC_LETTER_RANGES = ((0x0620, 0x064A), (0x0671, 0x06D3))

# Harakat and other combining marks. Noise for retrieval; stripped later.
ARABIC_DIACRITIC_RANGES = ((0x064B, 0x065F), (0x0670, 0x0670))

ARABIC_INDIC_DIGIT_RANGES = ((0x0660, 0x0669), (0x06F0, 0x06F9))

TATWEEL = 0x0640  # U+0640, the justification stroke
REPLACEMENT_CHAR = 0xFFFD

# Explicit bidirectional formatting characters. Meaningful to a renderer,
# pure noise to an index.
BIDI_CONTROLS = frozenset(
    {
        0x061C,  # Arabic letter mark
        0x200E,  # LTR mark
        0x200F,  # RTL mark
        0x202A,  # LTR embedding
        0x202B,  # RTL embedding
        0x202C,  # pop directional formatting
        0x202D,  # LTR override
        0x202E,  # RTL override
        0x2066,  # LTR isolate
        0x2067,  # RTL isolate
        0x2068,  # first strong isolate
        0x2069,  # pop directional isolate
    }
)

# Tab, newline and carriage return are legitimate layout, not corruption.
BENIGN_C0 = frozenset({0x09, 0x0A, 0x0D})


def _in_ranges(codepoint: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(low <= codepoint <= high for low, high in ranges)


@dataclass
class CharStats:
    # Counts of the character classes that distinguish good text from damage

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
        return self.presentation_forms + self.arabic_letters

    @property
    def presform_ratio(self) -> float:
        # hare of Arabic stored as display forms. High means failure mode A
        total = self.arabic_total
        return self.presentation_forms / total if total else 0.0

    @property
    def bidi_per_1k(self) -> float:
        return 1000 * self.bidi_controls / self.chars if self.chars else 0.0

    @property
    def c0_per_1k(self) -> float:
        # Control characters where letters belong. High means failure mode C
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
    # Count character classes in extracted text
    stats = CharStats(chars=len(text))

    for char in text:
        codepoint = ord(char)

        if codepoint == TATWEEL:
            stats.tatweel += 1
        elif codepoint == REPLACEMENT_CHAR:
            stats.replacement_chars += 1
        elif codepoint in BIDI_CONTROLS:
            stats.bidi_controls += 1
        elif _in_ranges(codepoint, PRESENTATION_FORM_RANGES):
            stats.presentation_forms += 1
        elif _in_ranges(codepoint, ARABIC_DIACRITIC_RANGES):
            stats.arabic_diacritics += 1
        elif _in_ranges(codepoint, ARABIC_INDIC_DIGIT_RANGES):
            stats.arabic_indic_digits += 1
        elif _in_ranges(codepoint, ARABIC_LETTER_RANGES):
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
    # What one document's text layer looks like measured

    path: Path
    pages: int
    producer: str
    creator: str
    stats: CharStats
    fonts: int = 0
    font_sample_pages: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def chars_per_page(self) -> float:
        """Low values mean the page is mostly graphics, so OCR is needed
        regardless of whether the text that did extract is sound."""
        return self.stats.chars / self.pages if self.pages else 0.0

    def to_dict(self) -> dict:
        return {
            "file": self.path.name,
            "pages": self.pages,
            "producer": self.producer,
            "creator": self.creator,
            "fonts_sampled": self.fonts,
            "font_sample_pages": self.font_sample_pages,
            "chars_per_page": round(self.chars_per_page, 1),
            "errors": self.errors,
            **self.stats.to_dict(),
        }


def _metadata_value(metadata: dict | None, key: str) -> str:
    if not metadata:
        return ""
    raw = metadata.get(key)
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return str(raw).strip()


def inspect_pdf(path: Path, font_sample_pages: int = 5, progress: bool = False) -> PdfInventory:
    # Extract a PDF's text layer and measure it
    errors: list[str] = []
    parts: list[str] = []
    font_names: set[str] = set()

    with pdfplumber.open(path) as pdf:
        producer = _metadata_value(pdf.metadata, "Producer")
        creator = _metadata_value(pdf.metadata, "Creator")
        page_count = len(pdf.pages)

        for number, page in enumerate(pdf.pages, start=1):
            try:
                parts.append(page.extract_text() or "")
                if number <= font_sample_pages:
                    font_names.update(char["fontname"] for char in page.chars if "fontname" in char)
            except Exception as exc:  # one bad page must not abort the
                # inventory; a page that cannot be read is itself a finding.
                errors.append(f"page {number}: {type(exc).__name__}: {exc}")
            finally:
                # pdfplumber caches parsed objects per page. Without this a
                # long document holds every page in memory at once.
                page.flush_cache()
                if progress and sys.stderr.isatty():
                    print(
                        f"\r  {path.name}: page {number}/{page_count}",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )

    if progress and sys.stderr.isatty():
        print(f"\r  {path.name}: {page_count} pages read", file=sys.stderr)

    return PdfInventory(
        path=path,
        pages=page_count,
        producer=producer,
        creator=creator,
        stats=analyse_text("\n".join(parts)),
        fonts=len(font_names),
        font_sample_pages=min(font_sample_pages, page_count),
        errors=errors,
    )


def inventory_dir(directory: Path, progress: bool = False) -> list[PdfInventory]:
    # Inspect every PDF in a directory, in a stable order
    pdfs = sorted(p for p in directory.glob("*.pdf") if p.is_file())
    return [inspect_pdf(path, progress=progress) for path in pdfs]


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
    # Render the inventory as an aligned plain text table
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

    notes = [f"{inv.path.name}: {err}" for inv in inventories for err in inv.errors]
    if notes:
        out.append("")
        out.append("Page errors:")
        out.extend(f"  {note}" for note in notes)

    return "\n".join(out)


def to_json(inventories: list[PdfInventory]) -> str:
    return json.dumps([inv.to_dict() for inv in inventories], indent=2, ensure_ascii=False)
