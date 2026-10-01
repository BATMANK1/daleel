"""Tests for rebuilding the academic calendar's events from its text layer.

Segments are placed as PDFium places the calendar's own: the boxes follow the
first card of academic_weeks_1448's page 1, in points from the top left. A
small generated PDF checks that PDFium's boxes are read the right way up. The
calendar itself, which CI does not have, came out identical to its ground
truth on page 1, and every row of its three pages passed the checks against
the .ics and the weekdays (README, Calendar).

The era suffixes and the ligature's control character are built from their
code points, so each string reads the same in any editor.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import pypdfium2
import pytest

from daleel.ingest.calendar import (
    COLUMNS,
    CalendarPage,
    CalendarRow,
    Segment,
    cell,
    kind,
    page_rows,
    read_calendar,
    segments,
    semester,
    split_glued,
    write_csv,
)
from daleel.normalize.arabic import for_comparison

MEEM = chr(0x0645)
HEH = chr(0x0647)
TATWEEL = chr(0x0640)
LIGATURE_FI = chr(0x1F)


def segment(text: str, left: float, top: float, right: float, bottom: float) -> Segment:
    return Segment(text, (left, top, right, bottom))


def card(
    middle: float,
    top: float,
    title_ar: str,
    title_en: str,
    days: tuple[str, str],
    gregorian: str,
    hijri: str,
) -> list[Segment]:
    """One card's segments, placed as on the page: the date box at `top`, the
    titles above it, the days right of the dates."""
    day_ar, day_en = days
    return [
        segment(title_ar, middle - 47, top - 35, middle + 47, top - 24),
        segment(title_en, middle - 42, top - 22, middle + 42, top - 15),
        segment(day_ar, middle + 28, top - 2, middle + 46, top + 6),
        segment(day_en, middle + 30, top + 9, middle + 45, top + 16),
        segment(gregorian + MEEM, middle - 47, top, middle + 3, top + 8),
        segment(hijri + HEH + TATWEEL, middle - 48, top + 9, middle + 3, top + 15),
    ]


SUNDAY = ("الأحد", "Sun")
THURSDAY = ("الخميس", "Thu")


def test_cards_are_read_top_to_bottom_and_each_row_right_to_left() -> None:
    right = card(568, 267, "بداية الفصل", "Start", SUNDAY, "2026/08/23", "1448/03/10")
    middle = card(350, 267, "نهاية الحذف", "Add and drop", THURSDAY, "2026/08/27", "1448/03/14")
    left = card(130, 268, "بداية الاعتذار", "Withdrawal", SUNDAY, "2026/08/30", "1448/03/17")
    below = card(568, 395, "الاختبارات", "Exams", SUNDAY, "2026/10/11", "1448/04/30")
    # The text layer gives cards in the order they were drawn, and one card's
    # days and dates before its title.
    found = middle + below[2:] + left + right + below[:2]
    rows = page_rows(found, frozenset())
    assert [row.title_en for row in rows] == ["Start", "Add and drop", "Withdrawal", "Exams"]
    assert rows[0] == CalendarRow(
        title_ar="بداية الفصل",
        title_en="Start",
        day_ar="الأحد",
        day_en="Sun",
        date_gregorian="2026/08/23" + MEEM,
        date_hijri="1448/03/10" + HEH,
    )
    assert rows[3].date_gregorian == "2026/10/11" + MEEM


def test_titles_over_several_lines_are_joined_in_order() -> None:
    date_box = card(568, 267, "", "", THURSDAY, "2026/12/10", "1448/07/01")[2:]
    titles = [
        segment("major selection", 535, 253, 595, 260),
        segment("نهاية فترة", 530, 220, 600, 230),
        segment("Last day for", 530, 244, 600, 251),
        segment("وإدخال الرغبات", 535, 232, 595, 242),
    ]
    (row,) = page_rows(titles + date_box, frozenset())
    assert row.title_ar == "نهاية فترة وإدخال الرغبات"
    assert row.title_en == "Last day for major selection"


def test_a_hijri_range_over_two_lines_is_one_cell() -> None:
    parts = card(350, 652, "الاختبارات", "Final Exams", THURSDAY, "2027/06/14-01", "1448/12/26")
    parts.append(segment("1449/01/09-" + HEH + TATWEEL, 291, 669, 345, 675))
    (row,) = page_rows(parts, frozenset())
    assert row.date_hijri == "1448/12/26" + HEH + " 1449/01/09-" + HEH


def test_a_line_that_fits_no_column_is_kept_apart() -> None:
    parts = card(568, 267, "بداية الفصل", "Start", SUNDAY, "2026/08/23", "1448/03/10")
    parts.append(segment("12:00 PM", 540, 270, 560, 276))
    (row,) = page_rows(parts, frozenset())
    assert row.unplaced == ("12:00 PM",)
    assert row.title_en == "Start"


def test_a_heading_above_every_card_joins_none_and_names_the_semester() -> None:
    heading = [
        segment("الفصل الدراسيالأول (481)", 61, 131, 200, 145),
        segment("First Semester ( 481 )", 74, 146, 165, 155),
    ]
    # The last card of the first semester names the second.
    parts = card(
        350,
        808,
        "تأجيل الفصل الدراسي الثاني (482)",
        "Postponement",
        THURSDAY,
        "2027/01/14",
        "1448/08/06",
    )
    rows = page_rows(heading + parts, frozenset())
    assert [row.title_ar for row in rows] == ["تأجيل الفصل الدراسي الثاني (482)"]
    assert semester(heading + parts) == ("الفصل الدراسي الأول (481)", "First Semester (481)")


def test_a_page_without_a_heading_names_no_semester() -> None:
    assert semester(card(568, 267, "بداية", "Start", SUNDAY, "2026/08/23", "1448/03/10")) == (
        "",
        "",
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026/08/23" + MEEM, "gregorian"),
        ("2027/01/17 " + MEEM, "gregorian"),
        ("2027/03/13-2/26" + MEEM, "gregorian"),
        ("1448/03/10" + HEH + TATWEEL, "hijri"),
        ("1449/01/15 " + HEH + TATWEEL, "hijri"),
        ("1449/01/09-" + HEH + TATWEEL, "hijri"),
        ("الأحد", "days_ar"),
        ("الأربعاء-السبت", "days_ar"),
        ("الجمعة -السبت", "days_ar"),
        ("Sun", "days_en"),
        ("Wed - Sat", "days_en"),
        ("بداية الفصل الدراسي الأول", "arabic"),
        ("- الساعة 12 م", "arabic"),
        ("Start of First Semester", "english"),
        ("study for Semester(482) - 12:00 PM", "english"),
        ("2026 -", "other"),
    ],
)
def test_each_line_is_told_by_its_form(text: str, expected: str) -> None:
    assert kind(text) == expected


# A character and its box: left, bottom, right and top, from the page's bottom left.
Char = tuple[str, tuple[float, float, float, float]]


class FakeTextPage:
    """A PDFium text page reduced to its text and a box for each character."""

    def __init__(self, chars: list[Char]) -> None:
        self.text = "".join(char for char, _ in chars)
        self.boxes = [box for _, box in chars]

    def count_chars(self) -> int:
        return len(self.boxes)

    def get_text_range(self) -> str:
        return self.text

    def get_charbox(self, index: int) -> tuple[float, float, float, float]:
        return self.boxes[index]


def line(text: str, left: float, bottom: float, width: float = 4.0) -> list[Char]:
    """A line's characters, each `width` wide, boxed as PDFium boxes them."""
    return [
        (char, (left + i * width, bottom, left + (i + 1) * width, bottom + 8))
        for i, char in enumerate(text)
    ]


BREAK: list[Char] = [("\r", (0, 0, 0, 0)), ("\n", (0, 0, 0, 0))]


def test_a_date_and_a_day_on_one_baseline_are_two_segments() -> None:
    chars = line("2026/10/11" + MEEM + " ", 83, 580) + line("Sun", 151, 580)
    chars += BREAK + line("Start of Midterm Exams", 86, 600)
    found = segments(FakeTextPage(chars), height=985)
    assert [part.text for part in found] == ["2026/10/11" + MEEM, "Sun", "Start of Midterm Exams"]
    # Boxes are measured from the top of the page.
    assert found[1].box == (151, 985 - 588, 163, 985 - 580)


def test_the_fi_ligature_is_read_as_fi() -> None:
    chars = line("Last day for entering and " + LIGATURE_FI + "nalizing", 50, 300)
    (found,) = segments(FakeTextPage(chars), height=985)
    assert found.text == "Last day for entering and finalizing"


def test_a_text_that_does_not_line_up_with_its_boxes_is_refused() -> None:
    # A character PDFium put into its text that has no box: every box after it
    # would belong to the character before.
    page = FakeTextPage(line("Sun", 151, 580))
    page.text = "S" + "-" + "un"
    with pytest.raises(ValueError, match="4 characters for PDFium's 3"):
        segments(page, height=985)


def test_a_real_text_layer_is_boxed_from_the_top_of_its_page(
    make_pdf: Callable[..., Path],
) -> None:
    # Helvetica at 12 points, its baseline 150 points up a page 200 high, so 50
    # down from the top, with capitals 8.6 high. PDFium reads the six spaces as
    # one, but they are 20 points wide, so the line parts in two.
    document = pypdfium2.PdfDocument(make_pdf(["Final Exams      Thu"]))
    try:
        page = document[0]
        textpage = page.get_textpage()
        try:
            found = segments(textpage, page.get_height())
        finally:
            textpage.close()
            page.close()
    finally:
        document.close()
    assert [part.text for part in found] == ["Final Exams", "Thu"]
    left, top, right, bottom = found[0].box
    # PDFium boxes the ink, and the F's ink starts a point right of the text's origin.
    assert left == pytest.approx(21, abs=1)
    assert top == pytest.approx(50 - 8.6, abs=1)
    assert bottom == pytest.approx(50, abs=1)
    assert found[1].box[0] - right > 10


def test_a_page_without_a_date_gives_no_rows(make_pdf: Callable[..., Path]) -> None:
    path = make_pdf(["Start of First Semester", ""])
    assert read_calendar(path, frozenset()) == [
        CalendarPage(1, "", "", ()),
        CalendarPage(2, "", "", ()),
    ]


LEXICON = frozenset(for_comparison(word) for word in ["الفصل", "الدراسي", "الأول", "بداية"])


def test_a_glued_pair_of_words_is_parted() -> None:
    assert split_glued("بداية الفصل الدراسيالأول (491)", LEXICON) == (
        "بداية الفصل الدراسي الأول (491)"
    )


def test_a_word_the_lexicon_holds_is_left_whole() -> None:
    assert split_glued("الدراسي", LEXICON | {for_comparison("الدرا"), "سي"}) == "الدراسي"


def test_a_word_that_parts_two_ways_is_left_whole() -> None:
    lexicon = LEXICON | {for_comparison("الدراسيا"), for_comparison("لأول")}
    assert split_glued("الدراسيالأول", lexicon) == "الدراسيالأول"


def test_cells_join_lines_with_single_spaces_and_drop_tatweel() -> None:
    assert cell(["Last day to request postponement of ", " study for Semester(482)"], LEXICON) == (
        "Last day to request postponement of study for Semester(482)"
    )
    assert cell(["1448/03/10" + HEH + TATWEEL], LEXICON) == "1448/03/10" + HEH


def test_rows_are_written_as_the_ground_truth_writes_a_table(tmp_path: Path) -> None:
    row = CalendarRow(
        "بداية", "Start, first day", "الأحد", "Sun", "2026/08/23" + MEEM, "1448/03/10" + HEH
    )
    path = tmp_path / "calendar" / "academic_weeks_1448_p01.csv"
    write_csv([row], path)
    with path.open(encoding="utf-8", newline="") as file:
        read = list(csv.reader(file))
    assert read == [list(COLUMNS), list(row.to_dict().values())]
    assert '"Start, first day"' in path.read_text(encoding="utf-8")
    assert list(row.to_dict()) == list(COLUMNS)
