"""Tests for putting detected text boxes in Arabic reading order.

Coordinates are pixels on a 300 DPI render, top-left origin, with lines about
55 pixels high and 70 apart, as on the organizational regulations' pages.
"""

from __future__ import annotations

from daleel.ocr.reading_order import Box, arabic_reading_order, rows


def box(text: str, left: float, top: float, right: float, bottom: float) -> Box:
    return Box(text, left, top, right, bottom)


def test_a_row_is_read_right_to_left() -> None:
    # PaddleOCR returns a split line left box first.
    left = box("الدراسية.", 130, 1000, 900, 1055)
    right = box("المادة الثانية والخمسون:", 1000, 1000, 2340, 1055)
    assert arabic_reading_order([left, right]) == "المادة الثانية والخمسون: الدراسية."


def test_rows_are_read_top_to_bottom() -> None:
    second = box("الثاني", 130, 1070, 2340, 1125)
    first = box("الأول", 130, 1000, 2340, 1055)
    # The newline sits in a literal of its own: glued to Arabic, ruff reads the
    # escape as Latin text and flags the Arabic letters as look-alikes.
    assert arabic_reading_order([second, first]) == "الأول" + "\n" + "الثاني"


def test_boxes_a_little_higher_or_lower_share_a_row() -> None:
    # Boxes of one line differ in height with ascenders and descenders.
    tall = box("أ", 1500, 995, 2340, 1060)
    short = box("ب", 130, 1008, 1400, 1052)
    assert rows([short, tall]) == [[tall, short]]


def test_a_small_box_above_a_line_does_not_split_it() -> None:
    # Page 21, item 6: PaddleOCR read a background stroke as "II" just above
    # the line, and the line's first word as a box of its own. Anchored on the
    # stroke, the row took the first word and left the rest for a new row.
    stroke = box("II", 700, 1870, 720, 1905)
    first_word = box("ألا", 2180, 1880, 2340, 1930)
    rest = box("يقل تقدير الطالب", 130, 1890, 2150, 1945)
    assert rows([rest, first_word, stroke]) == [[first_word, rest, stroke]]


def test_a_badge_below_a_line_starts_its_own_row() -> None:
    line = box("نسبة الغياب", 400, 1000, 2340, 1055)
    badge = box("01", 1200, 1100, 1260, 1140)
    assert rows([badge, line]) == [[line], [badge]]


def test_a_polygon_becomes_its_upright_box() -> None:
    quad = [(130.0, 1001.5), (2340.0, 1000.0), (2341.0, 1055.0), (131.0, 1056.0)]
    assert Box.around("سطر", quad) == box("سطر", 130.0, 1000.0, 2341.0, 1056.0)


def test_empty_text_and_empty_pages_add_nothing() -> None:
    assert arabic_reading_order([]) == ""
    blank = box("", 130, 1000, 2340, 1055)
    assert arabic_reading_order([blank, box("نص", 130, 1070, 2340, 1125)]) == "نص"
