"""Tests for Arabic comparison normalization.

Words appear as literal Arabic because these tests are about words. Invisible
characters and presentation forms appear as escapes because they cannot be
seen, and a test that relies on what an editor displays can't be trusted.
Several cases are printed inconsistencies recorded in the ground truth, which
normalization must make compare equal.
"""

from __future__ import annotations

import pytest

from daleel.normalize.arabic import (
    collapse_whitespace,
    fold_presentation_forms,
    for_comparison,
    remove_tatweel,
    strip_diacritics,
    strip_invisible,
    unify_alef,
    unify_alef_maqsura,
    unify_taa_marbuta,
)

# نظام in base letters, and the same word in Presentation Forms-B as the main
# guide's text layer stores it: noon initial, zah medial, alef final, then a
# base meem.
NIZAM = "نظام"
NIZAM_PRESENTATION = "\ufee7\ufec8\ufe8e\u0645"


def test_presentation_forms_fold_to_base_letters() -> None:
    assert fold_presentation_forms(NIZAM_PRESENTATION) == NIZAM


def test_lam_alef_ligature_folds_to_two_letters() -> None:
    assert fold_presentation_forms("\ufefb") == "لا"


def test_isolated_harakat_form_leaves_no_space() -> None:
    # NFKC turns the isolated fatha form into a space plus the mark.
    assert fold_presentation_forms("ب\ufe76") == "ب\u064e"


def test_folding_leaves_base_letters_alone() -> None:
    assert fold_presentation_forms("الطالب GPA 3.75") == "الطالب GPA 3.75"


def test_bidi_and_zero_width_characters_are_removed() -> None:
    assert strip_invisible("\u202b" + NIZAM + "\u202c\u200c\ufeff") == NIZAM


def test_control_codes_go_but_layout_whitespace_stays() -> None:
    assert strip_invisible("a\u0018b\tc\nd\re") == "ab\tc\nd\re"


def test_tatweel_is_removed() -> None:
    assert remove_tatweel("مؤس\u0640\u0640\u0640سة") == "مؤسسة"


@pytest.mark.parametrize(
    ("printed", "bare"),
    [
        ("يُمنح", "يمنح"),
        ("يومًا", "يوما"),
        ("أسبوعاً", "أسبوعا"),
        ("ثانٍ", "ثان"),
        ("يؤدِ", "يؤد"),
    ],
    ids=["damma", "tanween_on_meem", "tanween_on_alef", "kasratan", "kasra"],
)
def test_diacritics_are_removed(printed: str, bare: str) -> None:
    # All five are printed on student_guide_2025 page 30.
    assert strip_diacritics(printed) == bare


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("الإختبارات", "الاختبارات"),
        ("أن", "ان"),
        ("آخر", "اخر"),
        ("\u0671" + "لله", "الله"),  # alef wasla, then the rest of the word
    ],
    ids=["hamza_below", "hamza_above", "madda", "wasla"],
)
def test_alef_variants_unify(variant: str, expected: str) -> None:
    assert unify_alef(variant) == expected


def test_alef_maqsura_becomes_yaa() -> None:
    assert unify_alef_maqsura("على") == "علي"


def test_taa_marbuta_becomes_haa() -> None:
    assert unify_taa_marbuta("مؤسسة") == "مؤسسه"


def test_whitespace_collapses_to_single_spaces() -> None:
    assert collapse_whitespace("  نظام \n\t الطالب  ") == "نظام الطالب"


@pytest.mark.parametrize(
    ("one", "other"),
    [
        ("اجازة", "إجازة"),
        ("الاختبارات", "الإختبارات"),
        ("يؤدِ", "يؤد"),
        ("الالكتروني", "الإلكتروني"),
        ("الى", "إلى"),
    ],
    ids=["ijaza", "ikhtibarat", "yuaddi", "electroni", "ila"],
)
def test_printed_inconsistencies_compare_equal(one: str, other: str) -> None:
    # Each pair is spelled both ways in the corpus; ground truth keeps both.
    assert for_comparison(one) == for_comparison(other)


def test_presentation_form_text_matches_base_text() -> None:
    raw = "\u202b" + NIZAM_PRESENTATION + " \u0627\u0644\u0637\u0627\u0644\u0628\u202c"
    assert for_comparison(raw) == for_comparison("نظام الطالب")


@pytest.mark.parametrize(
    "text",
    ["", "نظام", NIZAM_PRESENTATION, "يُمنح الطالب تقدير (ح) أو (DN)", "الإختبارات  على\n"],
    ids=["empty", "plain", "presentation_forms", "mixed_direction", "needs_every_step"],
)
def test_normalization_is_idempotent(text: str) -> None:
    once = for_comparison(text)
    assert for_comparison(once) == once


def test_digits_and_latin_survive() -> None:
    assert for_comparison("GPA 3.75 (DN) 12:00 PM") == "GPA 3.75 (DN) 12:00 PM"
