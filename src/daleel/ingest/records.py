"""Every page's extracted text, with the method that produced it and why.

Each page becomes one record: its text, the method that produced it, and the
evidence behind the choice. A page's text layer is used only where the gate
trusts the page and its document as a whole takes the text layer. Every other
page is read by OCR: pages the gate rejects, pages with no Arabic in their text
layer, and the pages of a document the gate sends to OCR, even one it trusts,
since a page broken into fragments can still pass the lexicon
(eval/results/gate_calibration.md, section 7).

A record keeps the gate's measurements of the page's text layer whichever
method won, so that a wrong answer can be traced to its page, its method and
the evidence for it. A page read by OCR also keeps the engine's tag, the
resolution it was read at, and its layout: the blocks in reading order, each
box in the page's own points from its top left corner, as pdfplumber measures
characters, so a box and the text layer's characters can be laid side by side.
Reading order is the engine's, but for parts of the page it read left to
right, side by side, which are read right to left (daleel.ocr.reading_order),
and the page's text follows the blocks. Each block keeps its place in the
engine's own order, so a change to it shows.

Letters of other scripts the engine writes inside Arabic words are repaired
where they are known, and reported (daleel.ocr.letters).

A document's records go to data/interim/extracted/<document>.jsonl, a line
per page, and stay out of git like the rest of data/interim/.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daleel.ingest.extract import page_texts
from daleel.ingest.gate import PageVerdict, Verdict, document_path, judge_pages
from daleel.ingest.metadata import read_metadata
from daleel.ingest.router import ExtractionPath
from daleel.ocr.cache import CachedReader
from daleel.ocr.engine import Block, Box
from daleel.ocr.letters import repair_letters
from daleel.ocr.reading_order import block_order

RECORDS = Path("data/interim/extracted")


def in_points(box: Box, page_size: tuple[float, float]) -> list[float]:
    """A box given as fractions of the page, in the page's points."""
    width, height = page_size
    left, top, right, bottom = box
    return [
        round(value, 2) for value in (left * width, top * height, right * width, bottom * height)
    ]


@dataclass(frozen=True)
class OcrReading:
    """How a page was read by OCR: by which engine, at what resolution, into which layout."""

    engine: str
    dpi: int
    seconds: float
    warnings: str
    # The page's width and height in points, which its blocks' boxes are measured in.
    page_size: tuple[float, float]
    # The blocks in reading order, and each one's place in the engine's own
    # order, counting from 1: by default, the same order.
    blocks: tuple[Block, ...] = ()
    engine_order: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        places = self.engine_order or range(1, len(self.blocks) + 1)
        blocks = [
            {
                "category": block.category,
                "text": block.text,
                "box": None if block.box is None else in_points(block.box, self.page_size),
                "html": block.html,
                "engine_order": place,
            }
            for block, place in zip(self.blocks, places, strict=True)
        ]
        return {
            "engine": self.engine,
            "dpi": self.dpi,
            "seconds": round(self.seconds, 1),
            "warnings": self.warnings,
            "page_size": [round(value, 2) for value in self.page_size],
            "blocks": blocks,
        }


@dataclass(frozen=True)
class PageRecord:
    """One page's text, where it came from, and the gate's evidence for the choice."""

    doc_id: str
    page: int
    method: ExtractionPath
    producer: str
    text: str
    # The gate's verdict on this page's text layer, and on its document's.
    gate: PageVerdict
    document: ExtractionPath
    ocr: OcrReading | None = None

    def to_dict(self) -> dict[str, Any]:
        verdict = self.gate
        return {
            "doc_id": self.doc_id,
            "page": self.page,
            "method": self.method.value,
            "producer": self.producer,
            "text": self.text,
            "gate": {
                **verdict.decision.to_dict(),
                "document": self.document.value,
                **verdict.quality.to_dict(),
            },
            "ocr": None if self.ocr is None else self.ocr.to_dict(),
        }


def page_methods(verdicts: Sequence[PageVerdict]) -> tuple[ExtractionPath, list[ExtractionPath]]:
    """The path the document takes, and the method each of its pages is read by."""
    document = document_path(verdicts)
    methods = [
        ExtractionPath.TEXT_LAYER
        if document is ExtractionPath.TEXT_LAYER and verdict.decision.verdict is Verdict.TRUSTED
        else ExtractionPath.OCR
        for verdict in verdicts
    ]
    return document, methods


# Reads one page, numbered from 1, by OCR: its text and how it was read.
ReadPage = Callable[[int], tuple[str, OcrReading]]


def document_records(
    doc_id: str,
    producer: str,
    texts: Sequence[str],
    verdicts: Sequence[PageVerdict],
    read: ReadPage,
    on_record: Callable[[PageRecord], None] | None = None,
) -> list[PageRecord]:
    """A record for every page, reading by OCR each page the text layer cannot serve."""
    document, methods = page_methods(verdicts)
    records = []
    for text, verdict, method in zip(texts, verdicts, methods, strict=True):
        ocr = None
        if method is ExtractionPath.OCR:
            text, ocr = read(verdict.page)
        record = PageRecord(doc_id, verdict.page, method, producer, text, verdict, document, ocr)
        records.append(record)
        if on_record is not None:
            on_record(record)
    return records


def _repaired(block: Block, notes: list[str]) -> Block:
    text, found = repair_letters(block.text)
    if not found:
        return block
    notes.extend(found)
    return dataclasses.replace(block, text=text, html=repair_letters(block.html)[0])


def read_page(reader: CachedReader, path: Path, page: int) -> tuple[str, OcrReading]:
    """One page read by OCR, or read back from the cache, its blocks in reading order."""
    reading = reader.read(path, page)
    result = reading.result
    notes = [result.warnings] if result.warnings else []
    order = block_order(result.blocks)
    blocks = tuple(_repaired(result.blocks[place], notes) for place in order)
    # The engine's text is its blocks' text, so it follows them when they change.
    text, found = repair_letters(result.text)
    if blocks and (found or order != sorted(order)):
        text = "\n\n".join(block.text for block in blocks if block.text)
    elif not blocks:
        notes.extend(found)
    ocr = OcrReading(
        engine=reader.tag(),
        dpi=reader.dpi,
        seconds=result.seconds,
        warnings="; ".join(notes),
        page_size=reading.page_size,
        blocks=blocks,
        engine_order=tuple(place + 1 for place in order),
    )
    return text, ocr


def extract_document(
    path: Path,
    lexicon: frozenset[str],
    ocr: Callable[[], CachedReader],
    *,
    on_record: Callable[[PageRecord], None] | None = None,
) -> list[PageRecord]:
    """Every page of a PDF as a record, through the gate and, where needed, OCR.

    `ocr` gives the reader, and is called only when a page needs it, so a
    document whose text layer serves every page needs no engine at all. Pass the
    lexicon from load_gate_lexicon, as the gate was calibrated with it.
    """
    texts = page_texts(path)
    verdicts = judge_pages(texts, lexicon)
    producer = read_metadata(path).producer
    return document_records(
        path.stem,
        producer,
        texts,
        verdicts,
        lambda page: read_page(ocr(), path, page),
        on_record,
    )


def write_records(records: Sequence[PageRecord], folder: Path = RECORDS) -> Path:
    """Write one document's records as JSON lines, whole or not at all."""
    documents = {record.doc_id for record in records}
    if len(documents) != 1:
        raise ValueError(f"records are written a document at a time, not {len(documents)}")
    (doc_id,) = documents
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{doc_id}.jsonl"
    handle, name = tempfile.mkstemp(dir=folder, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        Path(name).replace(target)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return target


def read_records(path: Path) -> list[dict[str, Any]]:
    """A document's records as written, one dictionary per page."""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
