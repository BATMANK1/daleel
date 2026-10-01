"""Tests for text-layer extraction.

The PDFs come from the make_pdf fixture in conftest.py, which builds them byte
by byte, so the tests need no fixture files. Most use a standard Latin font:
they cover the wrapper's behaviour, while Arabic accuracy is established by the
calibration against ground truth. Arabic word order is tested on a generated
line, whose letters are boxes that a ToUnicode map names, so no Arabic font is
needed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from daleel.ingest import extract
from daleel.ingest.extract import (
    BACKENDS,
    ORDER_PROBE,
    UnsupportedPdfiumError,
    arabic_line_pdf,
    check_reading_order,
    page_text,
    page_texts,
)

# The probe as PDFium 153.0.7999.0, in pypdfium2 5.13.0, reads it: last word first.
REORDERED = " ".join(reversed(ORDER_PROBE.split()))


@pytest.fixture
def three_pages(make_pdf: Callable[..., Path]) -> Path:
    return make_pdf(["first page", "GPA 3.75 (DN)", ""])


def test_pages_are_numbered_from_one(three_pages: Path) -> None:
    assert page_text(three_pages, 1) == "first page"


def test_the_requested_page_is_read(three_pages: Path) -> None:
    assert page_text(three_pages, 2) == "GPA 3.75 (DN)"


def test_blank_page_has_empty_text(three_pages: Path) -> None:
    assert page_text(three_pages, 3) == ""


def test_page_zero_is_an_error(three_pages: Path) -> None:
    with pytest.raises(ValueError, match="numbered from 1"):
        page_text(three_pages, 0)


def test_page_past_the_end_is_an_error(three_pages: Path) -> None:
    with pytest.raises(IndexError, match="has 3 pages"):
        page_text(three_pages, 4)


def test_every_page_in_order(three_pages: Path) -> None:
    assert page_texts(three_pages) == ["first page", "GPA 3.75 (DN)", ""]


def test_line_breaks_are_newlines(make_pdf: Callable[..., Path]) -> None:
    text = page_text(make_pdf(["upper line\nlower line"]), 1)
    assert "\r" not in text
    assert text.split("\n") == ["upper line", "lower line"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_both_backends_read_latin_text_alike(three_pages: Path, backend: str) -> None:
    # Arabic is where they differ; see eval/results/gate_calibration.md.
    assert page_texts(three_pages, backend) == ["first page", "GPA 3.75 (DN)", ""]


def test_an_unknown_backend_is_an_error(three_pages: Path) -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        page_texts(three_pages, "pdftotext")


def test_an_unreadable_page_is_recorded_and_read_as_empty(
    three_pages: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = extract._text_of

    def failing_on_page_two(document, index):
        if index == 1:
            raise RuntimeError("damaged content stream")
        return real(document, index)

    monkeypatch.setattr(extract, "_text_of", failing_on_page_two)
    errors: list[str] = []
    assert page_texts(three_pages, errors=errors) == ["first page", "", ""]
    assert errors == ["page 2: RuntimeError: damaged content stream"]


def test_without_an_error_list_an_unreadable_page_raises(
    three_pages: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def always_failing(document, index):
        raise RuntimeError("damaged content stream")

    monkeypatch.setattr(extract, "_text_of", always_failing)
    with pytest.raises(RuntimeError):
        page_texts(three_pages)


def test_progress_is_reported_after_every_page(three_pages: Path) -> None:
    calls: list[tuple[int, int]] = []
    page_texts(three_pages, on_page=lambda number, count: calls.append((number, count)))
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_an_arabic_line_is_read_in_the_order_it_is_written(tmp_path: Path) -> None:
    # Drawn from its left end, last word first, as Adobe's software draws
    # Arabic: this is the test that fails on a build that reorders words.
    path = tmp_path / "line.pdf"
    path.write_bytes(arabic_line_pdf(ORDER_PROBE))
    assert page_text(path, 1) == ORDER_PROBE


@pytest.fixture
def unchecked() -> Iterator[None]:
    """Forget, before and after the test, that the installed PDFium passed the check."""
    check_reading_order.cache_clear()
    yield
    check_reading_order.cache_clear()


@pytest.mark.usefixtures("unchecked")
def test_a_pdfium_that_reorders_arabic_words_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extract, "_read_probe", lambda: REORDERED)
    with pytest.raises(UnsupportedPdfiumError, match="out of order") as refused:
        check_reading_order()
    assert 'uv pip install -e ".[dev]"' in str(refused.value)


@pytest.mark.usefixtures("unchecked")
def test_no_page_is_read_with_such_a_pdfium(
    three_pages: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read: list[int] = []
    monkeypatch.setattr(extract, "_read_probe", lambda: REORDERED)
    monkeypatch.setattr(extract, "_text_of", lambda document, index: read.append(index) or "")
    with pytest.raises(UnsupportedPdfiumError):
        page_texts(three_pages)
    with pytest.raises(UnsupportedPdfiumError):
        page_text(three_pages, 1)
    assert read == []


@pytest.mark.usefixtures("unchecked")
def test_the_check_runs_once(monkeypatch: pytest.MonkeyPatch, three_pages: Path) -> None:
    probes: list[str] = []
    monkeypatch.setattr(extract, "_read_probe", lambda: probes.append("read") or ORDER_PROBE)
    page_texts(three_pages)
    page_texts(three_pages)
    assert probes == ["read"]


@pytest.mark.usefixtures("unchecked")
def test_pdfplumber_needs_no_check(three_pages: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # It is kept for comparison, and its order is measured, not assumed.
    monkeypatch.setattr(extract, "_read_probe", lambda: REORDERED)
    assert page_texts(three_pages, "pdfplumber")[0] == "first page"
