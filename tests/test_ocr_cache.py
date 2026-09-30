"""Tests for the OCR cache.

A stand-in engine counts its readings and a counter wraps the renderer, so each
test can tell a page rendered and read again from a page read back.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from daleel.ocr import cache as cache_module
from daleel.ocr.cache import CachedReader
from daleel.ocr.engine import Block, Result
from daleel.ocr.render import render_page


class CountingEngine:
    """Reads every page as the same layout, and counts how often it is asked."""

    def __init__(self, tag: str = "stand-in-1.0", truncated: bool = False) -> None:
        self._tag = tag
        self.truncated = truncated
        self.readings = 0

    def tag(self) -> str:
        return self._tag

    def describe(self) -> str:
        return "a stand-in engine"

    def recognize(self, png: bytes) -> Result:
        self.readings += 1
        return Result(
            text="نص الصفحة",
            seconds=150.0,
            warnings="a warning",
            response='{"usage": {"prompt_tokens": 5172}}',
            blocks=(
                Block("Section-header", "عنوان", (0.5, 0.0, 1.0, 0.25)),
                Block("Picture", "", None),
                Block("Table", "أ ب", (0.0, 0.25, 1.0, 0.5), "<table></table>"),
            ),
            truncated=self.truncated,
        )


@pytest.fixture
def renders(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The pages the cache renders, in order."""
    rendered: list[int] = []

    def counting_render(path: Path, page: int, *, dpi: int = 300):
        rendered.append(page)
        return render_page(path, page, dpi=dpi)

    monkeypatch.setattr(cache_module, "render_page", counting_render)
    return rendered


def test_a_page_read_once_is_read_back_without_rendering(
    make_pdf: Callable[..., Path], tmp_path: Path, renders: list[int]
) -> None:
    pdf = make_pdf(["first page"])
    engine = CountingEngine()
    reader = CachedReader(engine, tmp_path / "cache")
    first = reader.read(pdf, 1)
    again = reader.read(pdf, 1)
    assert (engine.readings, renders) == (1, [1])
    assert again.result == first.result
    assert again.page_size == first.page_size == (300, 200)
    assert (first.cached, again.cached) == (False, True)
    assert (reader.hits, reader.misses) == (1, 1)


def test_the_cache_outlives_the_reader(make_pdf: Callable[..., Path], tmp_path: Path) -> None:
    pdf = make_pdf(["first page"])
    first = CachedReader(CountingEngine(), tmp_path / "cache").read(pdf, 1)
    engine = CountingEngine()
    assert CachedReader(engine, tmp_path / "cache").read(pdf, 1).result == first.result
    assert engine.readings == 0


def test_a_copy_of_the_document_under_another_name_is_read_back(
    make_pdf: Callable[..., Path], tmp_path: Path
) -> None:
    pdf = make_pdf(["first page"])
    CachedReader(CountingEngine(), tmp_path / "cache").read(pdf, 1)
    copy = tmp_path / "renamed.pdf"
    shutil.copy(pdf, copy)
    engine = CountingEngine()
    CachedReader(engine, tmp_path / "cache").read(copy, 1)
    assert engine.readings == 0


@pytest.mark.parametrize(
    "change",
    ["engine", "resolution", "page", "document"],
)
def test_anything_that_changes_the_reading_reads_the_page_again(
    make_pdf: Callable[..., Path], tmp_path: Path, change: str
) -> None:
    pdf = make_pdf(["first page", "second page"])
    CachedReader(CountingEngine(), tmp_path / "cache").read(pdf, 1)
    engine = CountingEngine("stand-in-2.0" if change == "engine" else "stand-in-1.0")
    reader = CachedReader(engine, tmp_path / "cache", dpi=300 if change == "resolution" else 200)
    if change == "document":
        pdf = make_pdf(["first page, changed", "second page"])
    reader.read(pdf, 2 if change == "page" else 1)
    assert engine.readings == 1


def test_an_answer_cut_off_is_not_kept(make_pdf: Callable[..., Path], tmp_path: Path) -> None:
    pdf = make_pdf(["first page"])
    engine = CountingEngine(truncated=True)
    reader = CachedReader(engine, tmp_path / "cache")
    reader.read(pdf, 1)
    reader.read(pdf, 1)
    assert engine.readings == 2
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_a_damaged_entry_is_read_again_and_replaced(
    make_pdf: Callable[..., Path], tmp_path: Path
) -> None:
    pdf = make_pdf(["first page"])
    first = CachedReader(CountingEngine(), tmp_path / "cache").read(pdf, 1)
    (entry,) = (tmp_path / "cache").rglob("*.json")
    entry.write_text('{"engine": "stand-in-1.0", "text": ', encoding="utf-8")
    engine = CountingEngine()
    assert CachedReader(engine, tmp_path / "cache").read(pdf, 1).result == first.result
    assert engine.readings == 1
    assert CachedReader(CountingEngine(), tmp_path / "cache").read(pdf, 1).cached


def test_nothing_half_written_is_left_behind(make_pdf: Callable[..., Path], tmp_path: Path) -> None:
    CachedReader(CountingEngine(), tmp_path / "cache").read(make_pdf(["first page"]), 1)
    files = [path for path in (tmp_path / "cache").rglob("*") if path.is_file()]
    assert [path.suffix for path in files] == [".json"]


def test_a_resolution_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        CachedReader(CountingEngine(), tmp_path, dpi=0)


def test_the_engine_s_names_pass_through(tmp_path: Path) -> None:
    reader = CachedReader(CountingEngine("stand-in-3.1"), tmp_path)
    assert reader.tag() == "stand-in-3.1"
    assert reader.describe() == "a stand-in engine"
