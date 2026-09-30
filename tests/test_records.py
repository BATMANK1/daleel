"""Tests for page records: which method reads each page, and what a record keeps.

A stand-in engine reads the pages that need OCR, so no model runs.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

from daleel.ingest import records as records_module
from daleel.ingest.gate import PageVerdict, decide
from daleel.ingest.quality import PageQuality
from daleel.ingest.records import (
    OcrReading,
    PageRecord,
    document_records,
    extract_document,
    in_points,
    page_methods,
    read_records,
    write_records,
)
from daleel.ingest.router import ExtractionPath
from daleel.ocr.cache import CachedReader
from daleel.ocr.engine import Block, Result

TEXT_LAYER, OCR = ExtractionPath.TEXT_LAYER, ExtractionPath.OCR


def page(number: int, validity: float | None, tokens: int = 100) -> PageVerdict:
    """A page the gate trusts at validity 1.0, rejects at 0.5, or finds no Arabic on at None."""
    quality = PageQuality(
        content_chars=500,
        presform_ratio=0.0,
        bidi_per_1k=0.0,
        c0_per_1k=0.0,
        replacement_chars=0,
        cid_placeholders=0,
        arabic_tokens=tokens if validity is not None else 0,
        token_validity=validity,
        single_letter_share=0.0,
    )
    return PageVerdict(page=number, quality=quality, decision=decide(quality))


class StandInEngine:
    """Reads every page as the same short layout, and keeps the images it was given."""

    def __init__(self) -> None:
        self.images: list[Image.Image] = []

    def tag(self) -> str:
        return "stand-in-1.0"

    def describe(self) -> str:
        return "a stand-in engine"

    def recognize(self, png: bytes) -> Result:
        self.images.append(Image.open(io.BytesIO(png)))
        return Result(
            text="نص مقروء",
            seconds=120.5,
            blocks=(
                Block("Section-header", "عنوان", (0.25, 0.0, 0.75, 0.25)),
                Block("Table", "أ ب", (0.0, 0.5, 1.0, 1.0), "<table></table>"),
            ),
        )


def reading(page_size: tuple[float, float] = (600, 800)) -> OcrReading:
    return OcrReading(
        engine="stand-in-1.0",
        dpi=200,
        seconds=120.5,
        warnings="",
        page_size=page_size,
        blocks=(Block("Text", "نص", (0.5, 0.25, 1.0, 0.75)),),
    )


# Choosing the method


def test_a_trusted_page_of_a_trusted_document_keeps_its_text_layer() -> None:
    verdicts = [page(1, 1.0), page(2, 1.0), page(3, 0.5), page(4, None)]
    document, methods = page_methods(verdicts)
    assert document is TEXT_LAYER
    assert methods == [TEXT_LAYER, TEXT_LAYER, OCR, OCR]


def test_every_page_of_a_document_sent_to_ocr_is_read_even_a_trusted_one() -> None:
    # The guidance manual: its one trusted page, 11, is broken into fragments that
    # still pass the lexicon (gate_calibration.md, section 7).
    verdicts = [page(n, 0.5) for n in range(1, 11)] + [page(11, 0.98)]
    verdicts += [page(n, 0.5) for n in range(12, 20)]
    document, methods = page_methods(verdicts)
    assert document is OCR
    assert set(methods) == {OCR}


def test_only_the_pages_the_text_layer_cannot_serve_are_read() -> None:
    asked: list[int] = []

    def read(number: int) -> tuple[str, OcrReading]:
        asked.append(number)
        return f"page {number} by OCR", reading()

    seen: list[int] = []
    records = document_records(
        "guide",
        "PDFium",
        ["first layer", "second layer", "third layer"],
        [page(1, 1.0), page(2, 0.5), page(3, 1.0)],
        read,
        on_record=lambda record: seen.append(record.page),
    )
    assert asked == [2]
    assert [record.text for record in records] == ["first layer", "page 2 by OCR", "third layer"]
    assert [record.method for record in records] == [TEXT_LAYER, OCR, TEXT_LAYER]
    assert [record.ocr is None for record in records] == [True, False, True]
    assert seen == [1, 2, 3]


# What a record keeps


def test_a_box_is_given_in_the_page_s_points() -> None:
    assert in_points((0.5, 0.25, 1.0, 0.75), (600, 800)) == [300, 200, 600, 600]


def test_a_record_keeps_the_gate_s_evidence_whichever_method_won() -> None:
    record = PageRecord("guide", 7, OCR, "PDFium", "نص", page(7, 0.5), TEXT_LAYER, reading())
    data = record.to_dict()
    assert data["method"] == "ocr"
    assert data["gate"]["verdict"] == "untrusted"
    assert data["gate"]["rejected_by"] == ["token_validity"]
    assert data["gate"]["document"] == "text_layer"
    assert data["gate"]["token_validity"] == 0.5
    assert data["ocr"] == {
        "engine": "stand-in-1.0",
        "dpi": 200,
        "seconds": 120.5,
        "warnings": "",
        "page_size": [600, 800],
        "blocks": [{"category": "Text", "text": "نص", "box": [300, 200, 600, 600], "html": ""}],
    }


def test_a_text_layer_record_has_no_ocr() -> None:
    record = PageRecord("guide", 1, TEXT_LAYER, "PDFium", "نص", page(1, 1.0), TEXT_LAYER)
    assert record.to_dict()["ocr"] is None


# Documents


def test_every_page_the_gate_cannot_trust_is_rendered_and_read(
    make_pdf: Callable[..., Path], tmp_path: Path
) -> None:
    # Latin text only, so the gate finds no Arabic to judge on either page.
    pdf = make_pdf(["first page", "second page"], name="notice.pdf", size=(300, 200))
    engine = StandInEngine()
    reader = CachedReader(engine, tmp_path / "cache")
    records = extract_document(pdf, frozenset(), lambda: reader)
    assert [(record.doc_id, record.page, record.method) for record in records] == [
        ("notice", 1, OCR),
        ("notice", 2, OCR),
    ]
    assert records[0].text == "نص مقروء"
    assert records[0].gate.decision.verdict.value == "no_arabic_text"
    reading = records[0].ocr
    assert reading is not None
    assert (reading.engine, reading.dpi, reading.page_size) == ("stand-in-1.0", 200, (300, 200))
    # 300 by 200 points at 200 DPI is 833.3 by 555.6 pixels, which PDFium rounds up.
    assert engine.images[0].size == (834, 556)
    assert records[0].to_dict()["ocr"]["blocks"][0]["box"] == [75, 0, 225, 50]


def test_no_engine_is_asked_for_when_the_text_layer_serves_every_page(
    make_pdf: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = make_pdf(["first page", "second page"])
    monkeypatch.setattr(
        records_module, "judge_pages", lambda texts, lexicon: [page(1, 1.0), page(2, 1.0)]
    )

    def no_engine() -> CachedReader:
        raise AssertionError("no page needed OCR")

    records = extract_document(pdf, frozenset(), no_engine)
    assert [record.method for record in records] == [TEXT_LAYER, TEXT_LAYER]
    assert records[0].text.strip() == "first page"


# Writing


def test_records_are_written_a_line_per_page_and_read_back(tmp_path: Path) -> None:
    records = [
        PageRecord("guide", 1, TEXT_LAYER, "PDFium", "الأولى", page(1, 1.0), TEXT_LAYER),
        PageRecord("guide", 2, OCR, "PDFium", "الثانية", page(2, 0.5), TEXT_LAYER, reading()),
    ]
    path = write_records(records, tmp_path / "extracted")
    assert path == tmp_path / "extracted" / "guide.jsonl"
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert read_records(path) == [record.to_dict() for record in records]
    assert [child.suffix for child in path.parent.iterdir()] == [".jsonl"]


def test_records_are_written_a_document_at_a_time(tmp_path: Path) -> None:
    records = [
        PageRecord("guide", 1, TEXT_LAYER, "PDFium", "", page(1, 1.0), TEXT_LAYER),
        PageRecord("charter", 1, TEXT_LAYER, "Adobe", "", page(1, 1.0), TEXT_LAYER),
    ]
    with pytest.raises(ValueError, match="a document at a time"):
        write_records(records, tmp_path)
