"""Rebuild the academic calendar's events, a row for each, from its text layer.

The academic calendar (academic_weeks_1448) prints every event as a card: its
title in Arabic and in English, then a box with the day or days, in Arabic and
in English, beside the Gregorian date and the Hijri date. The gate trusts its
text layer, and PDFium reads each card's lines whole and in logical order,
every date intact. What the text layer loses is the grid. It gives the cards
in the order they were drawn, not the order they are read, and on page 1 it
gives one card's days and dates before its title, with the page's heading in
between. Read as a page of text, a date can sit beside the wrong event.

So the grid is rebuilt from where each line sits. PDFium's lines are split
where a space separates text far apart, as a card's date and its day are when
they share a baseline. Every card has exactly one Gregorian date, so each date
anchors a card, and every other line goes to the card whose date lies in its
column and level with it or below it, within reach of a title. The page's
heading, above every card, and its footer, below them, are within reach of
none, and the heading names the semester. Lines above the date box are the
titles, Arabic and English by their letters; in the box, the days and the Hijri
date are told by their form. Cards are read as the annotation guidelines read
a page (rule 4.1): rows top to bottom, each row right to left. Page 1 comes out
identical to its hand-transcribed ground truth, all 84 cells of its 14 rows.

Two faults of the text layer are mended. Its English font maps the fi
ligature to U+001F, so finalizing comes out with a control character for its
first two letters; one followed by a Latin letter is read as fi. And five times
a word lost the space before it, as in الدراسيالأول: in each page's heading,
and in two titles on page 3. A word the lexicon does not hold, which splits
into exactly one pair of words it does hold, is split. Of the 376 Arabic words
on the calendar's pages, those five are the only ones the lexicon does not hold.

Rows follow the guidelines for tables (section 6), so they can be checked
against ground truth and the calendar's .ics (scripts/check_calendar_csv.py):
lines joined with single spaces, tatweel dropped, dates as printed.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Protocol

import pypdfium2

from daleel.normalize.arabic import for_comparison

# The columns of a calendar row, as the ground truth names them.
COLUMNS = ("title_ar", "title_en", "day_ar", "day_en", "date_gregorian", "date_hijri")

# Where a calendar's rows go, a CSV per page, out of git with the rest of
# data/interim/: they reproduce the calendar's text.
CALENDAR = Path("data/interim/calendar")

# Distances in points. A space wider than SPLIT parts a line into two segments.
SPLIT = 10.0
# A card's titles sit at most TITLES above its Gregorian date, and its Hijri
# date at most BELOW under it; its lines lie within COLUMN of the date
# sideways. Cards whose dates start within ROW of each other share a row.
TITLES = 60.0
BELOW = 12.0
COLUMN = 120.0
ROW = 20.0

# The Gregorian date, its suffix meem; the Hijri, its suffix heh, which the
# text layer writes with a tatweel. A range is printed end first and may cross
# a month; the second line of a Hijri range printed on two lines ends in a dash.
_DATE = r"\d{4}/\d{1,2}/\d{1,2}(?:-(?:\d{1,2}/)?\d{1,2})?"
GREGORIAN = re.compile(rf"{_DATE} ?\u0645")
HIJRI = re.compile(rf"{_DATE}-? ?\u0647\u0640?")
_DAYS_AR = "الأحد|الاحد|الإثنين|الاثنين|الثلاثاء|الأربعاء|الاربعاء|الخميس|الجمعة|السبت"
_DAYS_EN = "Sun|Mon|Tue|Wed|Thu|Fri|Sat"
DAYS_AR = re.compile(rf"(?:{_DAYS_AR})(?: ?- ?(?:{_DAYS_AR}))?")
DAYS_EN = re.compile(rf"(?:{_DAYS_EN})(?: ?- ?(?:{_DAYS_EN}))?")

_ARABIC_LETTER = re.compile("[\u0621-\u064a]")
_LATIN_LETTER = re.compile("[A-Za-z]")
_ARABIC_WORD = re.compile("[\u0621-\u064a]+")

# What this calendar's English font writes for its fi ligature.
LIGATURE_FI = "\x1f"
TATWEEL = "\u0640"

Box = tuple[float, float, float, float]


class TextPage(Protocol):
    """What segments() needs of a PDFium text page."""

    def count_chars(self) -> int: ...

    def get_text_range(self) -> str: ...

    def get_charbox(self, index: int) -> tuple[float, float, float, float]: ...


@dataclass(frozen=True)
class Segment:
    """A run of text on one line, and its box: left, top, right and bottom in
    points from the page's top left corner."""

    text: str
    box: Box

    @property
    def middle(self) -> float:
        return (self.box[0] + self.box[2]) / 2


def _chars(textpage: TextPage) -> list[tuple[str, Box | None]]:
    """The page's characters in PDFium's order, each with its box, or None for
    a line break or a box of no width.

    Each box is found by its character's place in the text, so the text must
    hold exactly one character for each of PDFium's. PDFium can leave one out
    of its text or put one in (pypdfium2 warns of both for get_text_range),
    and then no box can be trusted to be its character's.
    """
    text = textpage.get_text_range()
    count = textpage.count_chars()
    if len(text) != count:
        raise ValueError(
            f"its text has {len(text)} characters for PDFium's {count}, "
            "so their boxes cannot be matched to them"
        )
    found: list[tuple[str, Box | None]] = []
    for index, char in enumerate(text):
        if char in "\r\n":
            found.append(("\n", None))
            continue
        left, bottom, right, top = textpage.get_charbox(index)
        box = (left, bottom, right, top) if right > left else None
        following = text[index + 1 : index + 2]
        if char == LIGATURE_FI and following.isascii() and following.isalpha():
            char = "fi"
        found.append((char, box))
    return found


def segments(textpage: TextPage, height: float) -> list[Segment]:
    """The page's lines, split where a space parts text more than SPLIT apart."""
    found: list[Segment] = []

    def close(run: list[tuple[str, Box | None]]) -> None:
        boxes = [box for char, box in run if box is not None and not char.isspace()]
        text = "".join(char for char, _ in run).strip()
        if boxes and text:
            left = min(box[0] for box in boxes)
            right = max(box[2] for box in boxes)
            top = height - max(box[3] for box in boxes)
            bottom = height - min(box[1] for box in boxes)
            found.append(Segment(text, (left, top, right, bottom)))

    def next_on_line(chars: list[tuple[str, Box | None]], place: int) -> Box | None:
        for char, box in chars[place + 1 :]:
            if char == "\n":
                return None
            if box is not None and not char.isspace():
                return box
        return None

    chars = _chars(textpage)
    run: list[tuple[str, Box | None]] = []
    for place, (char, box) in enumerate(chars):
        if char == "\n":
            close(run)
            run = []
            continue
        before = [b for c, b in run if b is not None and not c.isspace()]
        after = next_on_line(chars, place) if char.isspace() and before else None
        if after is not None:
            left = min(b[0] for b in before)
            right = max(b[2] for b in before)
            if max(after[0] - right, left - after[2]) > SPLIT:
                close(run)
                run = []
                continue
        run.append((char, box))
    close(run)
    return found


def kind(text: str) -> str:
    """What a segment is: a Gregorian or Hijri date, days in Arabic or English,
    or Arabic or English text."""
    if GREGORIAN.fullmatch(text):
        return "gregorian"
    if HIJRI.fullmatch(text):
        return "hijri"
    if DAYS_AR.fullmatch(text):
        return "days_ar"
    if DAYS_EN.fullmatch(text):
        return "days_en"
    arabic = len(_ARABIC_LETTER.findall(text))
    latin = len(_LATIN_LETTER.findall(text))
    if arabic > latin:
        return "arabic"
    return "english" if latin else "other"


def _known(word: str, lexicon: frozenset[str]) -> bool:
    normalized = for_comparison(word)
    return bool(normalized) and normalized in lexicon


def split_glued(text: str, lexicon: frozenset[str]) -> str:
    """The text with each glued pair of words parted: a word the lexicon does
    not hold, which splits into exactly one pair of words it does."""

    def part(match: re.Match[str]) -> str:
        word = match[0]
        if _known(word, lexicon):
            return word
        pairs = [
            (word[:at], word[at:])
            for at in range(2, len(word) - 1)
            if _known(word[:at], lexicon) and _known(word[at:], lexicon)
        ]
        return " ".join(pairs[0]) if len(pairs) == 1 else word

    return _ARABIC_WORD.sub(part, text)


def cell(texts: Sequence[str], lexicon: frozenset[str]) -> str:
    """Lines joined into one cell: single spaces, no tatweel, glued words parted."""
    text = " ".join(" ".join(texts).replace(TATWEEL, "").split())
    return split_glued(text, lexicon)


@dataclass(frozen=True)
class CalendarRow:
    """One event of the calendar, as one card prints it."""

    title_ar: str
    title_en: str
    day_ar: str
    day_en: str
    date_gregorian: str
    date_hijri: str
    # The card's lines that fit none of the columns, which a check should see.
    unplaced: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, str]:
        return dict(zip(COLUMNS, astuple(self)[: len(COLUMNS)], strict=True))


def _card_for(part: Segment, anchors: Sequence[Segment]) -> Segment | None:
    """The card a segment belongs to: the nearest Gregorian date in its column
    that is level with it or below it, within reach of a title."""
    reach = [
        anchor
        for anchor in anchors
        if part.box[1] <= anchor.box[3] + BELOW
        and anchor.box[1] - part.box[3] <= TITLES
        and abs(anchor.middle - part.middle) <= COLUMN
    ]
    return min(reach, key=lambda anchor: abs(anchor.middle - part.middle), default=None)


def card_row(anchor: Segment, parts: Sequence[Segment], lexicon: frozenset[str]) -> CalendarRow:
    """The row a card makes: its titles above its date box, its days and dates in it."""
    parts = sorted(parts, key=lambda part: part.box[1])
    titles = [part for part in parts if part.box[1] < anchor.box[1] - 3]
    kinds = {id(part): kind(part.text) for part in parts}

    def texts(*wanted: str, among: Sequence[Segment] = parts) -> list[str]:
        return [part.text for part in among if kinds[id(part)] in wanted]

    used = {id(part) for part in titles if kinds[id(part)] in ("arabic", "english")}
    used |= {id(part) for part in parts if kinds[id(part)] in ("hijri", "days_ar", "days_en")}
    return CalendarRow(
        title_ar=cell(texts("arabic", among=titles), lexicon),
        title_en=cell(texts("english", among=titles), lexicon),
        day_ar=cell(texts("days_ar"), lexicon),
        day_en=cell(texts("days_en"), lexicon),
        date_gregorian=cell([anchor.text], lexicon),
        date_hijri=cell(texts("hijri"), lexicon),
        unplaced=tuple(part.text for part in parts if id(part) not in used),
    )


def page_rows(found: Sequence[Segment], lexicon: frozenset[str]) -> list[CalendarRow]:
    """A page's cards as rows, read top to bottom and each row right to left."""
    anchors = sorted(
        (part for part in found if kind(part.text) == "gregorian"), key=lambda a: a.box[1]
    )
    parts: dict[int, list[Segment]] = {id(anchor): [] for anchor in anchors}
    for part in found:
        if id(part) in parts:
            continue
        anchor = _card_for(part, anchors)
        if anchor is not None:
            parts[id(anchor)].append(part)

    rows: list[list[Segment]] = []
    for anchor in anchors:
        if rows and anchor.box[1] - rows[-1][0].box[1] <= ROW:
            rows[-1].append(anchor)
        else:
            rows.append([anchor])
    return [
        card_row(anchor, parts[id(anchor)], lexicon)
        for row in rows
        for anchor in sorted(row, key=lambda anchor: anchor.middle, reverse=True)
    ]


_SEMESTER_AR = re.compile(r"الفصل الدراسي\s*([\u0621-\u064a]+)\s*\(\s*(\d+)\s*\)")
_SEMESTER_EN = re.compile(r"([A-Z][a-z]+) Semester\s*\(\s*(\d+)\s*\)")


@dataclass(frozen=True)
class CalendarPage:
    """One page of the calendar: its semester, as its heading names it, and its rows."""

    page: int
    semester_ar: str
    semester_en: str
    rows: tuple[CalendarRow, ...]


def semester(found: Sequence[Segment]) -> tuple[str, str]:
    """The semester the page's heading names, in Arabic and in English, or blanks.

    The heading is what lies above every card: a card's title can name a
    semester too, as the last card of the first semester names the second.
    """
    dates = [part.box[1] for part in found if kind(part.text) == "gregorian"]
    top = min(dates, default=float("inf")) - TITLES
    heading = " ".join(part.text for part in found if part.box[3] < top)
    arabic = _SEMESTER_AR.search(heading)
    english = _SEMESTER_EN.search(heading)
    return (
        f"الفصل الدراسي {arabic[1]} ({arabic[2]})" if arabic else "",
        f"{english[1]} Semester ({english[2]})" if english else "",
    )


def read_calendar(path: Path, lexicon: frozenset[str]) -> list[CalendarPage]:
    """Every page of a calendar PDF, its cards rebuilt as rows.

    Pass the lexicon from load_gate_lexicon: it decides which glued words to part.
    Raises ValueError, naming the page, where a page's characters and their
    boxes cannot be matched.
    """
    document = pypdfium2.PdfDocument(path)
    try:
        pages = []
        for index in range(len(document)):
            page = document[index]
            textpage = page.get_textpage()
            try:
                found = segments(textpage, page.get_height())
            except ValueError as error:
                raise ValueError(f"page {index + 1}: {error}") from error
            finally:
                textpage.close()
                page.close()
            arabic, english = semester(found)
            rows = tuple(page_rows(found, lexicon))
            pages.append(CalendarPage(index + 1, arabic, english, rows))
        return pages
    finally:
        document.close()


def write_csv(rows: Sequence[CalendarRow], path: Path) -> None:
    """Write rows as the ground truth writes a table: a header, then a line per row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(astuple(row)[: len(COLUMNS)] for row in rows)
