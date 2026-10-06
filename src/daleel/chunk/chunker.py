"""Chunks cut along the documents' own structure: headings, articles and clauses.

A chunk cut through the middle of a clause cites nothing, so pages are split
where the documents split themselves. Each page's text is read as lines, and
each line is a heading, the start of a unit, or the continuation of one.
Headings come from an OCR page's layout, from articles' labels, from numbered
parts such as أولاً:, and from the student guide's table of contents. A heading
is a chunk's section_heading, not part of its text. Units start at numbered
clauses, bullets and layout blocks, and are merged when too short and split
when too long, but never across a page, since a chunk cites one page.

Two habits of the text layers shape the rules. The regulations and the
conduct code print an article's label in the margin, which their text layers
read at the end of the article's first line (PageReader). The student guide
prints the numbers of its illustrated lists beside their items, so a line of
bare numbers ends a unit; it is not taken as a clause number, because whether
it belongs to the text before it or after it differs between documents.

Chunks keep the extracted text as it is, diacritics and tatweel included:
normalizing it is the index's choice, and one the evaluation measures.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from daleel.corpus import DOCUMENTS
from daleel.ingest.records import read_records
from daleel.normalize.arabic import for_comparison

CHUNKS = Path("data/processed/chunks.jsonl")

# Sizes are counted in words: whitespace-separated tokens holding a letter or
# a digit.
MIN_WORDS = 40
FRAGMENT_WORDS = 12
MAX_WORDS = 180
OVERLAP = 30
# A chunk with fewer words holds nothing to retrieve: a stray year or a slogan.
MIN_CHUNK_WORDS = 3

# A line is a running header or footer when it has at most BOILERPLATE_WORDS
# words and appears on BOILERPLATE_SHARE of its document's pages, and on at
# least three. A document needs BOILERPLATE_PAGES pages for that to be told.
BOILERPLATE_WORDS = 8
BOILERPLATE_SHARE = 0.25
BOILERPLATE_PAGES = 6

CLAUSE, PROSE, TABLE = "clause", "prose", "table"

# The layout's categories, as dots.mocr names them.
HEADING_BLOCKS = frozenset({"Title", "Section-header"})
LEFT_OUT_BLOCKS = frozenset({"Page-header", "Page-footer", "Picture", "Formula"})
LIST_BLOCKS = frozenset({"List-item"})
TABLE_BLOCKS = frozenset({"Table"})
# A block called a heading that has more words than this is read as text.
HEADING_WORDS = 15
# A numbered part (ثانياً: ...) is a heading when it has at most this many words.
PART_WORDS = 8
# An article's label may skip this many numbers past the last one found.
ARTICLE_GAP = 3

BULLETS = "•●▪◦"
SENTENCE_ENDS = ".:;!?؛؟"

_HAS_WORD = re.compile(r"\w")
_ARABIC = re.compile(f"[{chr(0x0621)}-{chr(0x064A)}]")
# A line of numbers of one or two digits: page numbers, and the numbers of
# illustrated lists. An amount such as 750 on a line of its own is text.
_NUMBERS = re.compile(r"\d{1,2}(?: \d{1,2})*")
# Matched on a line normalized for comparison, so diacritics and tatweel are
# gone and digits are ASCII.
_NUMBERED = re.compile(r"(?:[.\-](\d{1,2})|(\d{1,2}) ?[.\-)])(?!\d)")

_ARTICLE = for_comparison("المادة")


def _normalized(text: str) -> frozenset[str]:
    """The words of a text, each normalized for comparison."""
    return frozenset(for_comparison(word) for word in text.split())


# The ordinal words that number articles, and what each adds to the number:
# الحادية عشرة is 1 and 10, الثانية والعشرون is 2 and 20.
_ORDINALS = {
    for_comparison(spelling): value
    for spellings, value in (
        ("الأولى الأول الحادية الحادي", 1),
        ("الثانية الثاني", 2),
        ("الثالثة الثالث", 3),
        ("الرابعة الرابع", 4),
        ("الخامسة الخامس", 5),
        ("السادسة السادس", 6),
        ("السابعة السابع", 7),
        ("الثامنة الثامن", 8),
        ("التاسعة التاسع", 9),
        ("العاشرة العاشر عشرة عشر", 10),
        ("العشرون العشرين", 20),
        ("الثلاثون الثلاثين", 30),
        ("الأربعون الأربعين", 40),
        ("الخمسون الخمسين", 50),
        ("الستون الستين", 60),
        ("السبعون السبعين", 70),
        ("الثمانون الثمانين", 80),
        ("التسعون التسعين", 90),
        ("المائة المئة", 100),
    )
    for spelling in spellings.split()
}
# Words that make an article named after them, with no colon, a reference.
_REFERRING = _normalized("في من وفق وفقا حسب بحسب بموجب لأحكام نص نصت")
_PARTS = _normalized("أولا ثانيا ثالثا رابعا خامسا سادسا سابعا ثامنا تاسعا عاشرا")
# The words a table of contents labels itself and its columns with.
_CONTENTS_LABELS = _normalized("الفهرس المحتوى الصفحة")


def words(text: str) -> int:
    """How many words a text holds: tokens with a letter or a digit."""
    return sum(1 for token in text.split() if _HAS_WORD.search(token))


@dataclass(frozen=True)
class Line:
    """One line of a page, and what it does to the units around it."""

    text: str
    heading: bool = False
    # It starts a unit: a clause, a bullet, or a block of the layout.
    starts: bool = False
    clause_no: str | None = None
    # It is part of a clause: numbered, bulleted, or a list item of the layout.
    clause: bool = False
    table: bool = False
    # A line of numbers: it ends the unit before it and holds nothing.
    stop: bool = False


@dataclass(frozen=True)
class Label:
    """An article's label found on a line, with the rest of the line."""

    label: str
    number: int
    # The line's other text: before a label that ends the line, or after one
    # that starts it.
    rest: str
    # A label with a colon, or alone on its line, cannot be a reference.
    certain: bool


@dataclass
class Unit:
    """A clause, a paragraph or a table: what a chunk is made of."""

    page: int
    heading: str | None
    content_type: str
    lines: list[str] = field(default_factory=list)
    clause_no: str | None = None

    @property
    def text(self) -> str:
        return ("\n" if self.content_type == TABLE else " ").join(self.lines)


@dataclass
class Draft:
    """Units merged into one chunk, before it has an id."""

    page: int
    heading: str | None
    content_type: str
    texts: list[str]
    clause_nos: list[str | None]

    @property
    def words(self) -> int:
        return sum(words(text) for text in self.texts)

    @property
    def clause_no(self) -> str | None:
        numbers = [number for number in self.clause_nos if number]
        if not numbers or numbers[0] == numbers[-1]:
            return numbers[0] if numbers else None
        return f"{numbers[0]}-{numbers[-1]}"


# --- lines ---------------------------------------------------------------------


def _ordinal(token: str) -> int:
    word = token.strip(":-")
    if word not in _ORDINALS and word.startswith("و"):
        word = word[1:]
    return _ORDINALS.get(word, 0)


def article_label(text: str) -> Label | None:
    """An article's label on a line, such as المادة الخامسة والعشرون:, or None.

    A label ends the line or is the whole of it, or starts it and ends in a
    colon. One in brackets, or named with no colon after في, من, وفق and the
    like, is a reference to an article, not its label.
    """
    tokens = text.split()
    probes = [for_comparison(token) for token in tokens]
    for start, probe in enumerate(probes):
        if probe.lstrip("-") != _ARTICLE:
            continue
        end, number = start + 1, 0
        while end < len(tokens) and end - start <= 3 and _ordinal(probes[end]):
            number += _ordinal(probes[end])
            end += 1
        if not number:
            continue
        while end < len(tokens) and probes[end] in (":", "-", ""):
            end += 1
        label = " ".join(tokens[start:end])
        colon = label.endswith(":")
        before = [probe for probe in probes[:start] if probe]
        if before and (
            "(" in before[-1] or ")" in before[-1] or (before[-1] in _REFERRING and not colon)
        ):
            continue
        if end == len(tokens):
            return Label(label, number, " ".join(tokens[:start]), colon or not before)
        if start == 0 and colon:
            return Label(label, number, " ".join(tokens[end:]), True)
    return None


def numbered_part(text: str) -> tuple[str, str] | None:
    """The text before a short numbered part (ثانياً: ...) ending the line, and the part."""
    tokens = text.split()
    for start, token in enumerate(tokens):
        probe = for_comparison(token)
        colon = probe.endswith(":") or (
            start + 1 < len(tokens) and tokens[start + 1].startswith(":")
        )
        if probe.rstrip(":") in _PARTS and colon and len(tokens) - start <= PART_WORDS:
            return " ".join(tokens[:start]), " ".join(tokens[start:])
    return None


def unit_line(text: str, starts: bool = False) -> Line:
    """A line of a unit: the start of a clause if it is numbered or bulleted."""
    probe = for_comparison(text)
    numbered = _NUMBERED.match(probe)
    if numbered:
        number = str(int(numbered.group(1) or numbered.group(2)))
        return Line(text, starts=True, clause_no=number, clause=True)
    if text[0] in BULLETS:
        return Line(text, starts=True, clause=True)
    return Line(text, starts=starts)


def _headed(heading: str, rest: str) -> list[Line]:
    line = Line(heading, heading=True)
    return [line, unit_line(rest, starts=True)] if rest else [line]


class PageReader:
    """Reads a document's pages as lines, carrying what runs from page to page.

    That is the number of the last article whose label was read. A label that
    could be a reference, one without a colon at the end of a line of text, is
    taken as a label only when it is the first article, or follows the last one
    found within ARTICLE_GAP, as articles do even where a label was lost in
    extraction. A reference to an article long before or after is passed by.
    """

    def __init__(
        self, boilerplate: frozenset[str] = frozenset(), contents: frozenset[str] = frozenset()
    ) -> None:
        self.boilerplate = boilerplate
        self.contents = contents
        self.article = 0

    def _label(self, text: str) -> Label | None:
        found = article_label(text)
        if found is None:
            return None
        follows = self.article < found.number <= self.article + ARTICLE_GAP
        if not (found.certain or follows or found.number == 1):
            return None
        self.article = found.number
        return found

    def line(self, raw: str) -> list[Line]:
        """What one line of extracted text is: nothing, a stop, a heading, or a unit's line."""
        text = " ".join(raw.split())
        if not _HAS_WORD.search(text):
            return []
        probe = for_comparison(text)
        if probe in self.boilerplate:
            return []
        if _NUMBERS.fullmatch(probe):
            return [Line("", stop=True)]
        if probe in self.contents:
            return [Line(text, heading=True)]
        label = self._label(text)
        if label:
            return _headed(label.label, label.rest)
        part = numbered_part(text)
        if part:
            return _headed(part[1], part[0])
        return [unit_line(text)]

    def block(self, block: Mapping[str, Any]) -> Iterator[Line]:
        """The lines of one block of a page's layout."""
        category, text = block.get("category"), block.get("text") or ""
        if category in LEFT_OUT_BLOCKS or not _HAS_WORD.search(text):
            return
        if category in TABLE_BLOCKS:
            rows = [" ".join(row.split()) for row in text.split("\n")]
            yield Line("\n".join(row for row in rows if row), table=True)
            return
        if category in HEADING_BLOCKS and words(text) <= HEADING_WORDS:
            heading = " ".join(text.split())
            label = article_label(heading)
            if label is not None:
                self.article = label.number
            yield Line(heading, heading=True)
            return
        first = True
        for raw in text.split("\n"):
            for line in self.line(raw):
                if first and not (line.heading or line.stop):
                    clause = line.clause or category in LIST_BLOCKS
                    line = Line(line.text, starts=True, clause_no=line.clause_no, clause=clause)
                    first = False
                yield line

    def page(self, record: Mapping[str, Any]) -> Iterator[Line]:
        """Every line of a page's record, read from its layout where OCR gave one."""
        blocks = (record.get("ocr") or {}).get("blocks") or []
        if blocks:
            for block in blocks:
                yield from self.block(block)
            return
        for raw in (record.get("text") or "").split("\n"):
            yield from self.line(raw)


def _all_lines(record: Mapping[str, Any]) -> Iterator[str]:
    blocks = (record.get("ocr") or {}).get("blocks") or []
    for text in [block.get("text") or "" for block in blocks] or [record.get("text") or ""]:
        yield from text.split("\n")


def boilerplate_lines(records: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    """A document's running headers and footers, normalized for comparison."""
    if len(records) < BOILERPLATE_PAGES:
        return frozenset()
    pages: Counter[str] = Counter()
    for record in records:
        probes = {for_comparison(raw) for raw in _all_lines(record)}
        pages.update(
            probe
            for probe in probes
            if probe and words(probe) <= BOILERPLATE_WORDS and not _NUMBERS.fullmatch(probe)
        )
    least = max(3, BOILERPLATE_SHARE * len(records))
    return frozenset(line for line, count in pages.items() if count >= least)


def contents_entries(
    records: Sequence[Mapping[str, Any]], page: int | None, boilerplate: frozenset[str]
) -> frozenset[str]:
    """The entries of a document's table of contents, normalized for comparison.

    The table's own labels (الفهرس, الصفحة) are not entries. A document with no
    such page, or without its records, has none.
    """
    record = next((record for record in records if record["page"] == page), None)
    if record is None:
        return frozenset()
    return frozenset(
        probe
        for probe in map(for_comparison, _all_lines(record))
        if _ARABIC.search(probe)
        and probe not in boilerplate
        and not _CONTENTS_LABELS & set(probe.split())
    )


# --- units and chunks ----------------------------------------------------------


def _heading(text: str) -> str:
    return text.strip(" :-")


def document_units(records: Sequence[Mapping[str, Any]]) -> list[Unit]:
    """A document's units in reading order, each under the heading before it."""
    doc_id = records[0]["doc_id"]
    boilerplate = boilerplate_lines(records)
    contents = DOCUMENTS[doc_id].contents_page
    reader = PageReader(boilerplate, contents_entries(records, contents, boilerplate))
    heading: str | None = None
    units: list[Unit] = []
    for record in records:
        page, current = record["page"], None
        if page == contents:
            continue
        for line in reader.page(record):
            if line.heading or line.stop or line.table:
                current = None
                if line.heading:
                    heading = _heading(line.text)
                elif line.table:
                    units.append(Unit(page, heading, TABLE, [line.text]))
                continue
            if current is None or line.starts:
                article = heading is not None and for_comparison(heading).startswith(_ARTICLE)
                kind = CLAUSE if line.clause or article else PROSE
                current = Unit(page, heading, kind, [], line.clause_no)
                units.append(current)
            current.lines.append(line.text)
    return units


def windows(count: int, ends: Sequence[bool]) -> list[tuple[int, int]]:
    """Where to cut `count` tokens into pieces of at most MAX_WORDS, overlapping.

    `ends` says which tokens end a sentence. A piece ends at the last sentence
    end in its second half, or at its full length where there is none.
    """
    spans, start = [], 0
    while True:
        end = min(start + MAX_WORDS, count)
        if end < count:
            for cut in range(end, start + MAX_WORDS // 2, -1):
                if ends[cut - 1]:
                    end = cut
                    break
        spans.append((start, end))
        if end >= count:
            return spans
        start = end - OVERLAP


def pieces(unit: Unit) -> list[Unit]:
    """A unit, or the overlapping pieces of one longer than MAX_WORDS."""
    if unit.content_type == TABLE or words(unit.text) <= MAX_WORDS:
        return [unit]
    tokens = unit.text.split()
    ends = [token[-1] in SENTENCE_ENDS for token in tokens]
    return [
        Unit(unit.page, unit.heading, unit.content_type, [" ".join(tokens[a:b])], unit.clause_no)
        for a, b in windows(len(tokens), ends)
    ]


def _joins(draft: Draft, unit: Unit) -> bool:
    size = words(unit.text)
    return (
        draft.page == unit.page
        and draft.heading == unit.heading
        and TABLE not in (draft.content_type, unit.content_type)
        and draft.words + size <= MAX_WORDS
        and (draft.words < MIN_WORDS or size < FRAGMENT_WORDS)
    )


def merge(units: Sequence[Unit]) -> list[Draft]:
    """Units merged where they are too short to stand alone.

    A unit shorter than MIN_WORDS takes in the units after it under the same
    heading on the same page, and a fragment shorter than FRAGMENT_WORDS joins
    the unit before it. No merge makes a chunk longer than MAX_WORDS, and
    tables are never merged.
    """
    drafts: list[Draft] = []
    for unit in units:
        if drafts and _joins(drafts[-1], unit):
            draft = drafts[-1]
            draft.texts.append(unit.text)
            draft.clause_nos.append(unit.clause_no)
            if unit.content_type == CLAUSE:
                draft.content_type = CLAUSE
        else:
            drafts.append(
                Draft(unit.page, unit.heading, unit.content_type, [unit.text], [unit.clause_no])
            )
    return drafts


def document_chunks(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """A document's chunks, from its records in page order, with their metadata.

    That is what a citation needs: the document's title, scope and date
    (daleel.corpus), the chunk's page, heading and clause number, whether it is
    a clause, prose or a table, and how its page was extracted, with the gate's
    score for the page's text layer, the share of its Arabic words the word
    list knows, so a wrong answer can be traced to its page.
    """
    if not records:
        return []
    doc_id = records[0]["doc_id"]
    if doc_id not in DOCUMENTS:
        raise ValueError(f"{doc_id} is not a document of the corpus")
    document = DOCUMENTS[doc_id]
    pages = {record["page"]: record for record in records}
    units = [piece for unit in document_units(records) for piece in pieces(unit)]
    chunks: list[dict[str, Any]] = []
    counts: Counter[int] = Counter()
    for draft in merge(units):
        if draft.words < MIN_CHUNK_WORDS:
            continue
        counts[draft.page] += 1
        record = pages[draft.page]
        score = record.get("gate", {}).get("token_validity")
        chunks.append(
            {
                "chunk_id": f"{doc_id}_p{draft.page}_c{counts[draft.page]}",
                "doc_id": doc_id,
                "doc_title": document.title,
                "scope": document.scope,
                "effective": document.effective,
                "page": draft.page,
                "section_heading": draft.heading,
                "clause_no": draft.clause_no,
                "content_type": draft.content_type,
                "extraction_method": record["method"],
                "gate_score": None if score is None else round(score, 3),
                "text": "\n".join(draft.texts),
            }
        )
    return chunks


def corpus_chunks(folder: Path) -> list[dict[str, Any]]:
    """The chunks of every document whose records are in `folder`, document by document."""
    chunks = []
    for path in sorted(folder.glob("*.jsonl")):
        records = sorted(read_records(path), key=lambda record: record["page"])
        chunks += document_chunks(records)
    return chunks


def write_chunks(chunks: Sequence[Mapping[str, Any]], path: Path = CHUNKS) -> Path:
    """Write chunks as JSON lines, whole or not at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        Path(name).replace(path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return path


def read_chunks(path: Path = CHUNKS) -> list[dict[str, Any]]:
    """Chunks as written, one dictionary each."""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
