"""Tests for text-layer measurement.

Arabic appears here as explicit \\u escapes rather than literal script. Three
reasons: the tests are about specific codepoints, so naming them is the point;
a reader can see exactly which class is under test without trusting their
editor; and bidirectional rendering makes mixed-direction source lines
genuinely hard to read correctly.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from daleel.ingest.extract import BACKENDS
from daleel.ingest.inventory import (
    CharStats,
    analyse_text,
    format_table,
    inspect_pdf,
)

# نظام ("system") in base letters -- what a student types.
BASE_ARABIC = "\u0646\u0638\u0627\u0645"

# The same word in Presentation Forms-B -- what the main guide's text layer
# emits. Different codepoints, so no term overlap with the above.
PRESENTATION_ARABIC = "\ufee7\ufec8\ufe8e\ufee1"


def test_empty_text_is_all_zeros_and_does_not_divide_by_zero() -> None:
    stats = analyse_text("")
    assert stats.chars == 0
    assert stats.arabic_total == 0
    assert stats.presform_ratio == 0.0
    assert stats.bidi_per_1k == 0.0
    assert stats.c0_per_1k == 0.0


def test_base_arabic_counted_as_letters_not_presentation_forms() -> None:
    stats = analyse_text(BASE_ARABIC)
    assert stats.arabic_letters == 4
    assert stats.presentation_forms == 0
    assert stats.presform_ratio == 0.0


def test_presentation_forms_counted_separately_from_letters() -> None:
    stats = analyse_text(PRESENTATION_ARABIC)
    assert stats.presentation_forms == 4
    assert stats.arabic_letters == 0
    assert stats.presform_ratio == 1.0


def test_presform_ratio_is_share_of_arabic_only() -> None:
    # Three display forms, one base letter, plus Latin that must not count.
    stats = analyse_text(PRESENTATION_ARABIC[:3] + BASE_ARABIC[:1] + "abc")
    assert stats.arabic_total == 4
    assert stats.presform_ratio == pytest.approx(0.75)


def test_bidi_controls_counted_and_not_mistaken_for_letters() -> None:
    # RTL embedding, text, pop directional formatting -- the pattern that
    # surrounds every line of the main guide's extracted text.
    stats = analyse_text("\u202b" + BASE_ARABIC + "\u202c")
    assert stats.bidi_controls == 2
    assert stats.arabic_letters == 4
    assert stats.bidi_per_1k == pytest.approx(1000 * 2 / 6)


def test_tatweel_counted_separately() -> None:
    # The justification stroke, and the likely cause of the lossy ToUnicode
    # mapping in the Word-produced documents.
    stats = analyse_text(BASE_ARABIC[:2] + "\u0640\u0640\u0640" + BASE_ARABIC[2:])
    assert stats.tatweel == 3
    assert stats.arabic_letters == 4


def test_diacritics_counted_separately_from_letters() -> None:
    stats = analyse_text("\u0646\u064e\u0638\u064e")  # two letters, two fatha
    assert stats.arabic_letters == 2
    assert stats.arabic_diacritics == 2


def test_c0_controls_counted_excluding_layout_whitespace() -> None:
    # The calendar's failure mode: raw control characters where letters belong.
    stats = analyse_text("a\u0018b\u0019c\td\ne\rf")
    assert stats.c0_controls == 2
    assert stats.latin_letters == 6


def test_replacement_characters_counted() -> None:
    stats = analyse_text("a\ufffdb\ufffd")
    assert stats.replacement_chars == 2
    assert stats.latin_letters == 2


def test_ascii_digits_survive_analysis() -> None:
    # Every numeric threshold in the corpus depends on these being intact.
    stats = analyse_text("3.75 of 4.00")
    assert stats.ascii_digits == 6


def test_arabic_indic_digits_counted_separately_from_ascii() -> None:
    stats = analyse_text("\u0663\u0667\u0665" + "375")
    assert stats.arabic_indic_digits == 3
    assert stats.ascii_digits == 3


def test_char_count_is_total_length_regardless_of_class() -> None:
    text = "\u202b" + BASE_ARABIC + "\u0640" + "abc123" + "\ufffd"
    assert analyse_text(text).chars == len(text)


def test_mixed_script_text_is_partitioned_not_double_counted() -> None:
    stats = analyse_text(BASE_ARABIC + "GPA 3.75")
    counted = (
        stats.arabic_letters + stats.latin_letters + stats.ascii_digits + stats.presentation_forms
    )
    # Remainder is the space and the decimal point, which belong to no class.
    assert counted == stats.chars - 2


def test_format_table_handles_no_documents() -> None:
    assert format_table([]) == "No PDFs found."


def test_char_stats_ratios_are_independent_of_text_length() -> None:
    small = CharStats(chars=100, bidi_controls=10)
    large = CharStats(chars=10_000, bidi_controls=1_000)
    assert small.bidi_per_1k == large.bidi_per_1k == pytest.approx(100.0)


def test_form_feed_is_page_separator_not_corruption() -> None:
    # pdftotext writes a form feed after every page. Counting it as a control
    # character made every document look slightly corrupt: exactly one "bad"
    # character per page, on all five documents.
    stats = analyse_text("page one\fpage two\f")
    assert stats.c0_controls == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_inventory_records_the_backend_that_read_it(
    make_pdf: Callable[..., Path], backend: str
) -> None:
    inventory = inspect_pdf(make_pdf(["first page", "second page"]), backend=backend)
    assert inventory.backend == backend
    assert inventory.to_dict()["backend"] == backend
    assert inventory.pages == 2
    assert inventory.stats.latin_letters == len("firstpagesecondpage")


def test_the_table_names_the_backend(make_pdf: Callable[..., Path]) -> None:
    table = format_table([inspect_pdf(make_pdf(["first page"]))])
    assert "Text layers read with pypdfium2." in table.splitlines()
