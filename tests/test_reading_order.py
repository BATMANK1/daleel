"""Tests for putting what an OCR engine read in Arabic reading order.

PaddleOCR's text boxes are in pixels on a 300 DPI render, top-left origin, with
lines about 55 pixels high and 70 apart, as on the organizational regulations'
pages. A layout model's blocks are placed as dots.mocr placed them on the pages
named, in points on the library deck's slides and in the model's pixels on the
other pages.
"""

from __future__ import annotations

from daleel.ocr.engine import Block
from daleel.ocr.reading_order import Box, arabic_reading_order, block_order, rows


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


# Blocks a layout model ordered


def block(category: str, text: str, left: float, top: float, right: float, bottom: float) -> Block:
    # The order only compares sides along one axis, so any unit serves.
    return Block(category, text, (left, top, right, bottom))


def test_two_columns_read_left_first_are_read_right_first() -> None:
    # Library deck, page 8: the policy's second part, on the left under its
    # own heading, came before the slide's title and the first part.
    page = [
        block("Page-header", "الهيئة الملكية", 1140, 28, 1414, 105),
        block("Section-header", "ثانيا الجزاءات والغرامات", 478, 117, 723, 147),
        block("Text", "للمكتبة الحق في أن تحرم من الاستعارة", 53, 215, 694, 303),
        block("Text", "يجوز للمكتبة حرمان المستعير من الإعارة", 53, 308, 694, 363),
        block("Section-header", "سياسة الاعارة", 946, 206, 1220, 273),
        block("Section-header", "أولا مدة الاعارة للمستفيدين", 946, 335, 1220, 370),
        block("Table", "الفئة عدد الكتب مدة الاعارة", 834, 433, 1334, 647),
        block("Page-footer", "6", 1346, 754, 1359, 774),
    ]
    assert block_order(page) == [0, 4, 5, 6, 1, 2, 3, 7]


def test_rows_of_cards_read_left_to_right_are_read_right_to_left() -> None:
    # The academic calendar: each card is an event above its dates, and the
    # model read each row from the left, against the order of the dates. The
    # second row has no card on the left, and the first row's left card stays
    # in its row.
    page = [
        block("Text", "بداية فترة الاعتذار", 47, 635, 568, 701),
        block("Table", "الأحد", 148, 707, 534, 800),
        block("Text", "نهاية فترة الحذف والإضافة", 809, 639, 1147, 704),
        block("Table", "الخميس", 786, 707, 1168, 800),
        block("Text", "بداية الفصل الدراسي", 1452, 641, 1722, 704),
        block("Table", "الأحد", 1411, 707, 1795, 800),
        block("Text", "نهاية فترة التأجيل", 640, 960, 1197, 1090),
        block("Table", "الخميس", 792, 1097, 1173, 1190),
        block("Text", "إجازة منتصف العام", 1452, 1026, 1751, 1090),
        block("Table", "الجمعة", 1415, 1097, 1815, 1190),
    ]
    assert block_order(page) == [4, 5, 2, 3, 0, 1, 8, 9, 6, 7]


def test_rows_of_label_and_value_read_right_first_keep_the_model_s_order() -> None:
    # Student guide, page 30: across a wide gap, each label sits right of its
    # value, and the model read them row by row.
    page = [
        block("Text", "حساب الحد الأقصى للغياب", 1196, 1073, 1372, 1147),
        block("Text", "إجمالي الساعات الدراسية للمقرر", 327, 1071, 1095, 1153),
        block("Text", "أعذار الغياب عن الاختبارات", 1187, 1221, 1377, 1299),
        block("Text", "لوكالة الشؤون التعليمية قبول الأعذار", 323, 1221, 1095, 1300),
    ]
    assert block_order(page) == [0, 1, 2, 3]


def test_cards_read_right_to_left_keep_the_model_s_order() -> None:
    # Student guide, page 30: each card's number sits below its text, and the
    # model read each number before its card's text.
    page = [
        block("Text", "01", 1295, 947, 1346, 986),
        block("Text", "يمنح الطالب تقدير محروم", 1207, 707, 1430, 924),
        block("Text", "02", 797, 947, 854, 986),
        block("Text", "يتم احتساب غياب الطالب", 524, 715, 1125, 915),
        block("Text", "03", 298, 947, 353, 986),
        block("Text", "تمنح الطالبة إجازة وضع", 213, 731, 438, 909),
    ]
    assert block_order(page) == [0, 1, 2, 3, 4, 5]


def test_a_picture_beside_text_stays_where_the_model_put_it() -> None:
    # Library deck, page 4: a QR code on the left of three steps.
    page = [
        block("Section-header", "طريقة اصدار عضوية المكتبة", 623, 125, 1084, 202),
        block("Picture", "", 639, 285, 900, 544),
        block("Text", "تعبئة نموذج عضوية المكتبة", 1043, 277, 1400, 310),
        block("Text", "وصول رابط لتعيين كلمة المرور", 1082, 402, 1372, 464),
        block("Text", "تعيين كلمة المرور", 1066, 539, 1382, 602),
    ]
    assert block_order(page) == [0, 1, 2, 3, 4]


def test_a_page_mostly_in_latin_script_keeps_the_model_s_order() -> None:
    page = [
        block("Text", "Left column, read first", 53, 215, 694, 303),
        block("Text", "Right column", 946, 206, 1220, 273),
    ]
    assert block_order(page) == [0, 1]


def test_a_block_without_a_box_keeps_the_model_s_order() -> None:
    page = [
        block("Text", "للمكتبة الحق في أن تحرم من الاستعارة", 53, 215, 694, 303),
        block("Section-header", "سياسة الاعارة", 946, 206, 1220, 273),
        Block("Text", "نص بلا موضع"),
    ]
    assert block_order(page) == [0, 1, 2]
