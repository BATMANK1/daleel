"""Tests for scoring OCR output against ground truth.

The Arabic cases are real. Three are errors Tesseract made on page 5 of the
organizational regulations; the rest are printed on page 21, including a
percentage whose reading order an engine can get wrong while reading every
character right.
"""

from __future__ import annotations

import random
from functools import cache

import pytest

from daleel.eval.ocr_metrics import (
    Errors,
    Span,
    char_errors,
    edit_distance,
    locate,
    missed_words,
    normalized_form,
    raw_form,
    word_errors,
    words,
)


def random_text(rng: random.Random, longest: int) -> str:
    return "".join(rng.choice("ab") for _ in range(rng.randint(0, longest)))


@cache
def by_definition(a: str, b: str) -> int:
    """Edit distance straight from its recursive definition, for small inputs."""
    if not a or not b:
        return len(a) + len(b)
    return min(
        by_definition(a[1:], b) + 1,
        by_definition(a, b[1:]) + 1,
        by_definition(a[1:], b[1:]) + (a[0] != b[0]),
    )


# Edit distance


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("", "", 0),
        ("abc", "", 3),
        ("", "abc", 3),
        ("abc", "abc", 0),
        ("kitten", "sitting", 3),
        ("flaw", "lawn", 2),
    ],
)
def test_edit_distance_counts_the_fewest_edits(a: str, b: str, expected: int) -> None:
    assert edit_distance(a, b) == expected


def test_edit_distance_agrees_with_its_definition() -> None:
    rng = random.Random(0)
    for _ in range(500):
        a, b = random_text(rng, 7), random_text(rng, 7)
        assert edit_distance(a, b) == by_definition(a, b) == edit_distance(b, a)


def test_edit_distance_compares_words_as_units() -> None:
    assert edit_distance(["في", "المقرر"], ["فق", "المقرر"]) == 1


# The two forms


@pytest.mark.parametrize(
    ("printed", "read", "raw_edits", "normalized_edits"),
    [
        ("التي", "الى", 2, 1),
        ("في", "فق", 1, 1),
        ("النهائي", "النهائ", 1, 1),
    ],
    ids=["lost_taa_and_maqsura", "wrong_letter", "lost_final_yaa"],
)
def test_tesseract_errors_from_page_5(
    printed: str, read: str, raw_edits: int, normalized_edits: int
) -> None:
    # Normalization forgives alef maqsura read for yaa. It never forgives a
    # letter that is wrong or missing.
    assert char_errors(printed, read).edits == raw_edits
    assert char_errors(printed, read, form=normalized_form).edits == normalized_edits


def test_line_breaks_are_layout_not_errors() -> None:
    # Escapes sit in their own literals: ruff reads \n glued to an Arabic word
    # as Latin text and flags the Arabic letters in it as look-alikes.
    printed = "درجة الاختبار" + "\n" + "النهائي"
    read = "درجة  الاختبار النهائي" + "\n"
    assert char_errors(printed, read).edits == 0


def test_direction_marks_are_not_text() -> None:
    # Tesseract wrapped this cell of page 27's grading table in these marks.
    marked = "‏" + "أقل من" + "‎"
    assert char_errors("أقل من", marked).edits == 0


def test_normalization_forgives_a_dropped_tanween() -> None:
    # Page 21 prints بناءً with tanween on the hamza.
    assert char_errors("بناءً", "بناء").edits == 1
    assert char_errors("بناءً", "بناء", form=normalized_form).edits == 0


def test_a_decomposed_letter_is_the_same_text() -> None:
    # Yaa followed by a combining hamza is canonically the same as the letter
    # yaa with hamza, so neither form counts it as an error. Without NFC first,
    # normalization would strip the combining hamza as a diacritic.
    decomposed = "الزائر"
    assert raw_form(decomposed) == "الزائر"
    assert char_errors("الزائر", decomposed).edits == 0
    assert char_errors("الزائر", decomposed, form=normalized_form).edits == 0


def test_percent_order_costs_characters_but_not_words() -> None:
    # Page 21 prints the percent sign to the left of the number. An engine that
    # returns (70%) has every character right and the order wrong.
    assert char_errors("(%70)", "(70%)").edits == 2
    assert word_errors("(%70)", "(70%)").edits == 0


# Words


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("الكلية/المعهد", ["الكلية", "المعهد"]),
        ("الكلية/ المعهد", ["الكلية", "المعهد"]),
        ("(%70)", ["70"]),
        ("بناءً على", ["بناءً", "على"]),
        ("RCJY.gov.sa", ["RCJY", "gov", "sa"]),
        ("", []),
    ],
    ids=["slash", "slash_and_space", "percentage", "mark_stays_on_word", "latin", "empty"],
)
def test_words_are_runs_of_letters_digits_and_marks(text: str, expected: list[str]) -> None:
    assert words(text) == expected


def test_word_errors_are_normalized_unless_asked_otherwise() -> None:
    assert word_errors("بناءً", "بناء").edits == 0
    assert word_errors("بناءً", "بناء", form=raw_form).edits == 1


def test_missed_words_ignore_order() -> None:
    # A table row read left to right instead of right to left loses no words.
    row = "ممتاز مرتفع 95 100"
    assert missed_words(row, "100 95 مرتفع ممتاز") == Errors(edits=0, length=4)


def test_missed_words_count_repeats() -> None:
    assert missed_words("ممتاز ممتاز جيد", "ممتاز جيد") == Errors(edits=1, length=3)


# Errors


def test_rates_add_edits_and_lengths_before_dividing() -> None:
    total = Errors(edits=1, length=10) + Errors(edits=9, length=90)
    assert total == Errors(edits=10, length=100)
    assert total.rate == 0.1


def test_empty_ground_truth_has_no_rate() -> None:
    with pytest.raises(ValueError, match="not empty"):
        _ = Errors(edits=0, length=0).rate


# Locating a region


@pytest.mark.parametrize(
    ("pattern", "text", "expected"),
    [
        ("abc", "xxabcxx", Span(start=2, end=5, edits=0)),
        ("adc", "xxabcxx", Span(start=2, end=5, edits=1)),
        ("ab", "abxx", Span(start=0, end=2, edits=0)),
        ("ab", "xxab", Span(start=2, end=4, edits=0)),
        ("ab", "xxabxxab", Span(start=2, end=4, edits=0)),
        # "ab" with d missing and "abc" with d misread both cost one edit.
        ("abd", "xxabcxx", Span(start=2, end=4, edits=1)),
        ("abc", "xx", Span(start=0, end=0, edits=3)),
        ("", "abc", Span(start=0, end=0, edits=0)),
    ],
    ids=[
        "exact",
        "one_edit",
        "at_start",
        "at_end",
        "first_of_two",
        "tie_ends_first",
        "absent",
        "empty_pattern",
    ],
)
def test_locate_finds_the_best_stretch(pattern: str, text: str, expected: Span) -> None:
    assert locate(pattern, text) == expected


def test_locate_agrees_with_trying_every_stretch() -> None:
    rng = random.Random(1)
    for _ in range(300):
        pattern, text = random_text(rng, 5), random_text(rng, 8)
        span = locate(pattern, text)
        best = min(
            edit_distance(pattern, text[i:j])
            for i in range(len(text) + 1)
            for j in range(i, len(text) + 1)
        )
        assert span.edits == best
        assert edit_distance(pattern, text[span.start : span.end]) == span.edits


def test_a_region_is_scored_against_the_output_that_matches_it() -> None:
    region = "درجة الاختبار النهائي: الدرجة التي يحصل عليها الطالب"
    page = "اللائحة التنظيمية\n" + region.replace("التي", "الى") + "\nRCJY.gov.sa 02"
    assert char_errors(region, page, region=True).edits == 2
    assert word_errors(region, page, region=True).edits == 1
    # Scored as a whole page, everything else the engine read counts too.
    assert char_errors(region, page).edits > 30
