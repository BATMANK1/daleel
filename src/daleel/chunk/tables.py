"""Tables as rows, each row a sentence that can be cited on its own.

A retrieved fragment of a table is no use to an answer: the number 750 means
nothing apart from its program and its category of student. So every row of a
table becomes a sentence that names each of its cells by its column, as
"الفئة: الطالب السعودي أو من في حكمه، البرنامج: برنامج الدراسة بمقابل مالي،
المبلغ: 750".

Tables come from three places. On a page read by OCR, the layout gives each
table as HTML, whose merged cells are spread back over the rows and columns
they cover. The academic calendar's cards are rebuilt from where their lines
sit on the page (daleel.ingest.calendar), and each card is a row. And a table
whose text layer scatters its cells, where no reading puts them back, is
typed in by hand from the rendered page: MANUAL_TABLES holds those, each with
the lines of the text layer it stands for, and their chunks are marked as
manually verified.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from itertools import zip_longest
from pathlib import Path
from typing import Any

MANUAL_TABLES = Path("data/manual/tables.json")

# Between a column's name and its value, and between cells.
NAMED = ": "
JOINED = "، "


@dataclass(frozen=True)
class Cell:
    text: str
    rowspan: int = 1
    colspan: int = 1


@dataclass(frozen=True)
class Row:
    cells: tuple[Cell, ...]
    # A row of column names: in the table's head, or made of header cells.
    header: bool


class _TableParser(HTMLParser):
    """The rows of an HTML table, as the layout writes them."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[Row] = []
        self._cells: list[Cell] | None = None
        self._headers: list[bool] = []
        self._text: list[str] | None = None
        self._spans = (1, 1)
        self._head = False

    @staticmethod
    def _span(attrs: list[tuple[str, str | None]], name: str) -> int:
        value = dict(attrs).get(name) or "1"
        return int(value) if value.isdigit() and int(value) > 0 else 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "thead":
            self._head = True
        elif tag in ("tbody", "tfoot"):
            self._head = False
        elif tag == "tr":
            self._cells, self._headers = [], []
        elif tag in ("td", "th") and self._cells is not None:
            self._text = []
            self._spans = (self._span(attrs, "rowspan"), self._span(attrs, "colspan"))
            self._headers.append(tag == "th")
        elif tag == "br" and self._text is not None:
            self._text.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cells is not None and self._text is not None:
            text = " ".join("".join(self._text).split())
            self._cells.append(Cell(text, *self._spans))
            self._text = None
        elif tag == "tr" and self._cells is not None:
            if self._cells:
                header = self._head or all(self._headers)
                self.rows.append(Row(tuple(self._cells), header))
            self._cells = None
        elif tag == "thead":
            self._head = False

    def handle_data(self, data: str) -> None:
        if self._text is not None:
            self._text.append(data)


def html_rows(html: str) -> list[Row]:
    """The rows of an HTML table."""
    parser = _TableParser()
    parser.feed(html)
    parser.close()
    return parser.rows


def _carried(below: dict[int, list], column: int) -> str:
    """The text a cell above carries down into this row's column, and one row less."""
    text, left = below[column]
    if left == 1:
        del below[column]
    else:
        below[column][1] = left - 1
    return text


def grid(rows: Sequence[Row]) -> list[tuple[bool, list[str]]]:
    """A table's rows with merged cells spread over what they cover.

    A cell merged down the rows below it is repeated in each, so every row is
    whole on its own. A cell merged across columns names every column it
    covers in a header row, and fills only the first in a row of values, so
    a value is not said twice.
    """
    below: dict[int, list] = {}
    table = []
    for row in rows:
        line: dict[int, str] = {}
        column = 0
        for cell in row.cells:
            while column in below:
                line[column] = _carried(below, column)
                column += 1
            for offset in range(cell.colspan):
                text = cell.text if offset == 0 or row.header else ""
                line[column + offset] = text
                if cell.rowspan > 1:
                    below[column + offset] = [text, cell.rowspan - 1]
            column += cell.colspan
        for place in sorted(set(below) - set(line)):
            line[place] = _carried(below, place)
        width = max(line, default=-1) + 1
        table.append((row.header, [line.get(place, "") for place in range(width)]))
    return table


def column_names(table: Sequence[tuple[bool, list[str]]]) -> list[str]:
    """Each column's name: the distinct names its header rows give it, top to bottom."""
    width = max((len(cells) for _, cells in table), default=0)
    names = []
    for place in range(width):
        parts: list[str] = []
        for header, cells in table:
            text = cells[place] if header and place < len(cells) else ""
            if text and text not in parts:
                parts.append(text)
        names.append(" ".join(parts))
    return names


def sentence(names: Sequence[str], cells: Sequence[str], caption: str = "") -> str:
    """One row as a sentence: each value after its column's name, empty cells left out."""
    parts = [
        f"{name}{NAMED}{value}" if name else value
        for name, value in zip_longest(names[: len(cells)], cells, fillvalue="")
        if value
    ]
    text = JOINED.join(parts)
    return f"{caption.rstrip(' :')}{JOINED}{text}" if caption and text else text


def table_sentences(html: str) -> list[str]:
    """Every row of values of an HTML table as a sentence."""
    table = grid(html_rows(html))
    names = column_names(table)
    found = [sentence(names, cells) for header, cells in table if not header]
    return [text for text in found if text]


def calendar_sentence(semester: str, row: Any) -> str:
    """A calendar card as a sentence: its semester, its event, its days and its dates."""
    title = f"{row.title_ar} ({row.title_en})" if row.title_en else row.title_ar
    days = f"{row.day_ar} ({row.day_en})" if row.day_en else row.day_ar
    when = " ".join(part for part in (days, row.date_gregorian) if part)
    text = f"{title}: {when}{JOINED}{row.date_hijri}" if row.date_hijri else f"{title}: {when}"
    return f"{semester}{JOINED}{text}" if semester else text


@dataclass(frozen=True)
class ManualTable:
    """A table typed in by hand from its rendered page, for one its text layer scatters."""

    doc_id: str
    page: int
    caption: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    # The lines of the page's text layer the table stands for, as extracted.
    replaces: tuple[str, ...]
    checked: str

    def sentences(self) -> list[str]:
        return [sentence(self.columns, row, self.caption) for row in self.rows]


def load_manual_tables(path: Path = MANUAL_TABLES) -> list[ManualTable]:
    """The tables typed in by hand, as the file lists them."""
    with path.open(encoding="utf-8") as file:
        entries = json.load(file)
    tables = []
    for entry in entries:
        table = _manual_table(entry)
        widths = {len(row) for row in table.rows}
        if widths != {len(table.columns)}:
            raise ValueError(
                f"{path}: {table.doc_id} page {table.page}: every row needs "
                f"{len(table.columns)} cells, one per column"
            )
        tables.append(table)
    return tables


def _manual_table(entry: Mapping[str, Any]) -> ManualTable:
    return ManualTable(
        doc_id=entry["doc_id"],
        page=entry["page"],
        caption=entry.get("caption", ""),
        columns=tuple(entry["columns"]),
        rows=tuple(tuple(row) for row in entry["rows"]),
        replaces=tuple(entry["replaces"]),
        checked=entry["checked"],
    )
