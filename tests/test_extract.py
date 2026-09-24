"""Tests for text-layer extraction.

The PDFs come from the make_pdf fixture in conftest.py, which builds them byte
by byte, so the tests need no fixture files. They
use a standard Latin font, since embedding an Arabic font in a test is not
practical: these tests cover the wrapper's behaviour, while Arabic accuracy is
established by the calibration against ground truth.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from daleel.ingest import extract
from daleel.ingest.extract import BACKENDS, page_text, page_texts


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
