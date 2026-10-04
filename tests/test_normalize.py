"""Tests for Arabic comparison normalization.

Words appear as literal Arabic because these tests are about words. Invisible
characters and presentation forms appear as escapes because they cannot be
seen, and a test that relies on what an editor displays can't be trusted.
Several cases are printed inconsistencies recorded in the ground truth, which
normalization must make compare equal.

The last tests run on real lines from every document with hand-transcribed
ground truth, kept in tests/fixtures/normalization_samples.json: each line as a
text layer or dots.mocr gives it, beside the same line as the page prints it.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    unify_digits,
    unify_punctuation,
    unify_taa_marbuta,
)

# نظام in base letters, and the same word in Presentation Forms-B as the main
# guide's text layer stores it: noon initial, zah medial, alef final, then a
# base meem.
NIZAM = "نظام"
NIZAM_PRESENTATION = "\ufee7\ufec8\ufe8e\u0645"

# Arabic-Indic numbers are written as escapes where their digits pass for Latin
# letters or punctuation, which ruff flags as look-alikes. GRADE_POINTS is 3.75
# as page 27 of the organizational regulations prints it: three, the Arabic
# decimal separator, seven, five.
GRADE_POINTS = "\u0663\u066b\u0667\u0665"
ELEVEN_HUNDRED = "\u0661\u066c\u0661\u0660\u0660"  # with the Arabic thousands separator
SEVENTY_PERCENT = "\u0667\u0660\u066a"  # with the Arabic percent sign

# The extended Arabic-Indic digits Persian and Urdu use. Several look exactly
# like their Arabic-Indic twins, so they are built from their codepoints.
EXTENDED_DIGITS = "".join(chr(0x06F0 + value) for value in range(10))


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
    [
        "",
        "نظام",
        NIZAM_PRESENTATION,
        "يُمنح الطالب تقدير (ح) أو (DN)",
        "الإختبارات  على\n",
        "الفصل الدراسي الثاني (٤٨٢)" + "\u060c",
    ],
    ids=[
        "empty",
        "plain",
        "presentation_forms",
        "mixed_direction",
        "needs_every_step",
        "digits_and_punctuation",
    ],
)
def test_normalization_is_idempotent(text: str) -> None:
    once = for_comparison(text)
    assert for_comparison(once) == once


def test_digits_and_latin_survive() -> None:
    assert for_comparison("GPA 3.75 (DN) 12:00 PM") == "GPA 3.75 (DN) 12:00 PM"


@pytest.mark.parametrize(
    ("printed", "ascii_form"),
    [
        ("٠١٢٣٤٥٦٧٨٩", "0123456789"),
        (EXTENDED_DIGITS, "0123456789"),
        (GRADE_POINTS, "3.75"),
        (ELEVEN_HUNDRED, "1,100"),
        (SEVENTY_PERCENT, "70%"),
    ],
    ids=["arabic_indic", "extended", "decimal_separator", "thousands_separator", "percent_sign"],
)
def test_arabic_indic_numbers_become_ascii(printed: str, ascii_form: str) -> None:
    # The corpus prints no thousands separator or Arabic percent sign; a query
    # typed on an Arabic keyboard can.
    assert unify_digits(printed) == ascii_form


@pytest.mark.parametrize(
    ("arabic_indic", "ascii_digits"),
    [
        ("٩٠إلى أقل من ٩٥", "90إلى أقل من 95"),
        (GRADE_POINTS, "3.75"),
        ("الفصل الدراسي الثاني (٤٨٢)", "الفصل الدراسي الثاني (482)"),
    ],
    ids=["grading_band", "grade_points", "semester_code"],
)
def test_a_number_matches_in_either_script(arabic_indic: str, ascii_digits: str) -> None:
    # Page 27 of the organizational regulations prints its grading scale in
    # Arabic-Indic digits, and the calendar names the second semester ٤٨٢ in
    # Arabic and 482 in English. Either spelling must find the other.
    assert for_comparison(arabic_indic) == for_comparison(ascii_digits)


@pytest.mark.parametrize(
    "number",
    ["3.75", "4.00", "1100", "(200)", "2026/08/23", "1448/03/10", "12:00 PM"],
)
def test_ascii_numbers_are_left_alone(number: str) -> None:
    # GPAs, fees and dates are the facts the system exists to report, so
    # normalization must never change a digit or its punctuation.
    assert for_comparison(number) == number


def test_arabic_punctuation_becomes_latin() -> None:
    # Page 30 of the student guide joins two clauses with an Arabic semicolon.
    assert unify_punctuation("يحدد في التوصية\u061b وإذا مضى") == "يحدد في التوصية; وإذا مضى"
    assert unify_punctuation("\u060c\u061b\u061f") == ",;?"


def test_a_comma_matches_in_either_script() -> None:
    # Page 21 of the organizational regulations uses the Arabic comma.
    printed = "الهيئة الملكية\u060c أو في أي جامعة"
    assert for_comparison(printed) == for_comparison("الهيئة الملكية, أو في أي جامعة")


@pytest.mark.parametrize(
    "text",
    ["", "   ", "\n\t \r\n", "\u200f\u202b\u202c"],
    ids=["empty", "spaces", "layout_whitespace", "direction_marks"],
)
def test_blank_text_normalizes_to_nothing(text: str) -> None:
    assert for_comparison(text) == ""


SAMPLES = json.loads(
    (Path(__file__).parent / "fixtures" / "normalization_samples.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("sample", SAMPLES["same"], ids=[s["name"] for s in SAMPLES["same"]])
def test_real_text_matches_its_page_once_normalized(sample: dict) -> None:
    # Each raw line differs from the page only in how it is encoded.
    assert sample["raw"] != sample["printed"]
    assert for_comparison(sample["raw"]) == for_comparison(sample["printed"])


@pytest.mark.parametrize(
    "sample", SAMPLES["different"], ids=[s["name"] for s in SAMPLES["different"]]
)
def test_real_damage_survives_normalization(sample: dict) -> None:
    # A broken text layer or a misread page is the gate's and the engine's
    # business: normalization must not make it look right.
    assert for_comparison(sample["raw"]) != for_comparison(sample["printed"])


def test_the_samples_come_from_every_document_with_ground_truth() -> None:
    documents = {sample["doc"] for sample in SAMPLES["same"] + SAMPLES["different"]}
    assert documents == {
        "academic_weeks_1448",
        "guidance_manual",
        "library_services_2024_2025",
        "organizational_regulations",
        "orientation_1446",
        "student_charter",
        "student_guide_2025",
    }
