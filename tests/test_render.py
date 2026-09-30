"""Tests for rendering pages to images for OCR."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

from daleel.ocr.render import printed_scale, render_page

# organizational_regulations.pdf declares every page at this size: A4 at a tenth.
REGULATIONS_PAGE = (59.5276, 84.189)


@pytest.mark.parametrize(
    "size",
    [(595.276, 841.89), (612, 792), (960, 540), (300, 200)],
    ids=["a4", "us_letter", "slide", "test_page"],
)
def test_real_pages_print_at_their_declared_size(size: tuple[float, float]) -> None:
    assert printed_scale(*size) == 1.0


def test_a_page_drawn_at_a_tenth_prints_as_a4() -> None:
    assert printed_scale(*REGULATIONS_PAGE) == pytest.approx(10.0, rel=1e-3)


@pytest.mark.parametrize(("dpi", "expected"), [(72, (300, 200)), (144, (600, 400))])
def test_pixels_follow_the_resolution(
    make_pdf: Callable[..., Path], dpi: int, expected: tuple[int, int]
) -> None:
    assert render_page(make_pdf(["HELLO"]), 1, dpi=dpi).image.size == expected


def test_a_page_drawn_at_a_tenth_renders_as_a4_at_300_dpi(
    make_pdf: Callable[..., Path],
) -> None:
    width, height = render_page(make_pdf([""], size=REGULATIONS_PAGE), 1).image.size
    assert width == pytest.approx(2480, abs=1)
    assert height == pytest.approx(3508, abs=1)


def test_the_page_keeps_its_declared_size_in_points(make_pdf: Callable[..., Path]) -> None:
    # The page's own coordinates, even where it prints ten times larger.
    rendered = render_page(make_pdf([""], size=REGULATIONS_PAGE), 1, dpi=72)
    assert rendered.page_size == pytest.approx(REGULATIONS_PAGE, abs=0.01)


def test_the_text_is_drawn(make_pdf: Callable[..., Path]) -> None:
    blank = render_page(make_pdf([""]), 1, dpi=72).image.convert("L")
    text = render_page(make_pdf(["HELLO"]), 1, dpi=72).image.convert("L")
    assert blank.getextrema() == (255, 255)
    assert text.getextrema()[0] < 128


def test_the_png_carries_its_resolution(make_pdf: Callable[..., Path]) -> None:
    png = render_page(make_pdf([""], size=REGULATIONS_PAGE), 1, dpi=300).png()
    with Image.open(io.BytesIO(png)) as image:
        assert image.info["dpi"] == pytest.approx((300, 300), abs=0.01)


def test_page_zero_is_an_error(make_pdf: Callable[..., Path]) -> None:
    with pytest.raises(ValueError, match="numbered from 1"):
        render_page(make_pdf(["HELLO"]), 0)


def test_page_past_the_end_is_an_error(make_pdf: Callable[..., Path]) -> None:
    with pytest.raises(IndexError, match="has 1 pages"):
        render_page(make_pdf(["HELLO"]), 2)


def test_resolution_must_be_positive(make_pdf: Callable[..., Path]) -> None:
    with pytest.raises(ValueError, match="positive"):
        render_page(make_pdf(["HELLO"]), 1, dpi=0)
