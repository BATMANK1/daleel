"""Tests for tables as rows: HTML from the OCR layout, calendar cards, and
tables typed in by hand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daleel.chunk.tables import (
    MANUAL_TABLES,
    Cell,
    calendar_sentence,
    column_names,
    grid,
    html_rows,
    load_manual_tables,
    sentence,
    table_sentences,
)
from daleel.corpus import DOCUMENTS
from daleel.ingest.calendar import CalendarRow

MEEM, HEH = chr(0x0645), chr(0x0647)


def html(*rows: str, head: str = "") -> str:
    body = "".join(f"<tr>{row}</tr>" for row in rows)
    return (
        f"<table>{f'<thead><tr>{head}</tr></thead>' if head else ''}<tbody>{body}</tbody></table>"
    )


def test_rows_are_read_with_their_spans_and_line_breaks() -> None:
    rows = html_rows(
        html(
            '<td rowspan="2">مقيم</td><td>برنامج<br>الدراسة</td>',
            head='<th colspan="2">الفئة</th>',
        )
    )
    assert [row.header for row in rows] == [True, False]
    assert rows[0].cells == (Cell("الفئة", 1, 2),)
    assert rows[1].cells == (Cell("مقيم", 2, 1), Cell("برنامج الدراسة"))


def test_a_row_of_header_cells_is_a_header_outside_the_head_too() -> None:
    rows = html_rows("<table><tr><th>CODE</th><th>CH</th></tr><tr><td>ENG 001</td><td>8</td></tr>")
    assert [row.header for row in rows] == [True, False]


def test_a_cell_merged_down_is_repeated_in_every_row_it_covers() -> None:
    table = grid(
        html_rows(
            html(
                '<td rowspan="2">مقيم</td><td>الفرص الإضافية</td><td rowspan="2">1100</td>',
                "<td>برنامج الدراسة</td>",
            )
        )
    )
    assert [cells for _, cells in table] == [
        ["مقيم", "الفرص الإضافية", "1100"],
        ["مقيم", "برنامج الدراسة", "1100"],
    ]


def test_a_cell_merged_across_names_its_columns_but_fills_one_value() -> None:
    table = grid(
        html_rows(
            html(
                '<td colspan="2">Total</td><td>14</td>',
                head='<th colspan="2">Course</th><th>CH</th>',
            )
        )
    )
    assert table == [(True, ["Course", "Course", "CH"]), (False, ["Total", "", "14"])]


def test_columns_are_named_by_every_header_row() -> None:
    table = grid(
        html_rows(
            "<table><thead>"
            '<tr><th rowspan="2">Course Code</th><th colspan="2">Details</th></tr>'
            "<tr><th>Sec</th><th>Sun</th></tr>"
            "</thead><tbody><tr><td>SCM426</td><td>1</td><td>80,44</td></tr></tbody></table>"
        )
    )
    assert column_names(table) == ["Course Code", "Details Sec", "Details Sun"]


def test_each_row_of_values_becomes_a_sentence_naming_its_columns() -> None:
    found = table_sentences(
        html(
            "<td>الطلبة</td><td>5 كتب</td><td>14 يوم</td>",
            head="<td>الفئة</td><td>عدد الكتب</td><td>مدة الاعارة</td>",
        )
    )
    assert found == ["الفئة: الطلبة، عدد الكتب: 5 كتب، مدة الاعارة: 14 يوم"]


def test_a_table_with_no_header_joins_its_cells() -> None:
    found = table_sentences(html("<td>رئيساً</td><td>وكيل شؤون الطلاب</td>"))
    assert found == ["رئيساً، وكيل شؤون الطلاب"]


def test_empty_cells_and_empty_rows_are_left_out() -> None:
    found = table_sentences(
        html("<td>A</td><td></td><td>3</td>", "<td></td><td></td><td></td>", head="<td>x</td>" * 3)
    )
    assert found == ["x: A، x: 3"]
    assert table_sentences("") == []


def test_a_caption_leads_each_row() -> None:
    assert sentence(["المبلغ"], ["250"], "تستوفى الرسوم حسب الآتي:") == (
        "تستوفى الرسوم حسب الآتي، المبلغ: 250"
    )


def test_a_calendar_card_is_a_sentence_with_its_semester() -> None:
    row = CalendarRow(
        "بداية الفصل الدراسي الأول",
        "Start of First Semester",
        "الأحد",
        "Sun",
        "2026/08/23" + MEEM,
        "1448/03/10" + HEH,
    )
    assert calendar_sentence("الفصل الدراسي الأول (481)", row) == (
        "الفصل الدراسي الأول (481)، بداية الفصل الدراسي الأول (Start of First Semester): "
        f"الأحد (Sun) 2026/08/23{MEEM}، 1448/03/10{HEH}"
    )
    bare = CalendarRow("إجازة", "", "الأحد", "", "2026/11/20" + MEEM, "")
    assert calendar_sentence("", bare) == f"إجازة: الأحد 2026/11/20{MEEM}"


def test_the_tables_typed_by_hand_are_whole() -> None:
    tables = load_manual_tables(MANUAL_TABLES)
    assert tables
    for table in tables:
        assert table.doc_id in DOCUMENTS
        assert table.replaces and table.checked
        assert all(len(row) == len(table.columns) for row in table.rows)
        assert len(table.sentences()) == len(table.rows)


def test_a_table_typed_by_hand_must_give_every_row_a_cell_per_column(tmp_path: Path) -> None:
    entry = {
        "doc_id": "student_guide_2025",
        "page": 28,
        "columns": ["الفئة", "المبلغ"],
        "rows": [["مقيم"]],
        "replaces": ["1100"],
        "checked": "by hand",
    }
    path = tmp_path / "tables.json"
    path.write_text(json.dumps([entry], ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="every row needs 2 cells"):
        load_manual_tables(path)
