"""Tests for repairing letters of other scripts a model writes inside Arabic words.

The letters are built from their code points, so the tests read the same in any
editor and no linter mistakes them for look-alikes.
"""

from __future__ import annotations

from daleel.ocr.letters import repair_letters

SO_SUA = chr(0x0E2A)  # THAI CHARACTER SO SUA
TAV = chr(0x05EA)  # HEBREW LETTER TAV
ZHE = chr(0x0416)  # CYRILLIC CAPITAL LETTER ZHE
REPLACEMENT = chr(0xFFFD)


def test_the_thai_letter_in_a_week_is_read_as_sin_and_ba() -> None:
    # Library deck, page 8, and organizational regulations, pages 21 and 33.
    text, notes = repair_letters("خلال أ" + SO_SUA + "وعين من إبلاغه")
    assert text == "خلال أسبوعين من إبلاغه"
    assert notes == ["THAI CHARACTER SO SUA in أ" + SO_SUA + "وعين read as سب"]


def test_the_hebrew_tav_is_read_as_ta() -> None:
    # Library deck, page 4: the same character one UTF-8 byte off.
    text, notes = repair_letters(TAV + "عبئة نموذج عضوية المكتبة")
    assert text == "تعبئة نموذج عضوية المكتبة"
    assert notes == ["HEBREW LETTER TAV in " + TAV + "عبئة read as ت"]


def test_every_known_letter_in_a_text_is_repaired() -> None:
    text, notes = repair_letters("أ" + SO_SUA + "وع و" + TAV + "عبئة")
    assert text == "أسبوع وتعبئة"
    assert len(notes) == 2


def test_a_letter_of_another_script_away_from_arabic_is_left_alone() -> None:
    assert repair_letters(SO_SUA + " page " + TAV) == (SO_SUA + " page " + TAV, [])


def test_another_letter_inside_an_arabic_word_is_reported_and_left() -> None:
    text = "مكتبة" + ZHE + "عامة"
    note = "CYRILLIC CAPITAL LETTER ZHE in " + text + " left as read"
    assert repair_letters(text) == (text, [note])


def test_a_character_the_model_could_not_write_is_reported() -> None:
    text = REPLACEMENT + "عبئة"
    assert repair_letters(text) == (text, ["REPLACEMENT CHARACTER in " + text + " left as read"])


def test_arabic_and_latin_text_comes_back_as_it_was() -> None:
    text = "منصة BlackBoard و Edugate"
    assert repair_letters(text) == (text, [])
