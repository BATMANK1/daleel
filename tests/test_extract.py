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

from daleel.ingest.extract import page_text, page_texts


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
