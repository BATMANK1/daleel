"""Tests for text-layer quality metrics.

The central cases are the corpus's real failure modes: presentation-form text
that normalization recovers, a silently wrong word that passes every structural
check, cid placeholders disguised as Latin, and words run together on the page.
"""

from __future__ import annotations

from daleel.ingest.quality import (
    PageQuality,
    arabic_tokens,
    build_lexicon,
    cid_placeholders,
    measure,
    single_letter_share,
    token_validity,
)
from daleel.normalize.arabic import for_comparison

# نظام in Presentation Forms-B, as the main guide's text layer stores it.
NIZAM_PRESENTATION = "\ufee7\ufec8\ufe8e\u0645"


def test_lexicon_is_normalized_like_page_text() -> None:
    lexicon = build_lexicon(["إجازة"])
    assert for_comparison("اجازة") in lexicon


def test_lexicon_drops_words_that_normalize_to_nothing() -> None:
    assert build_lexicon(["\u0640", ""]) == frozenset()


def test_cid_placeholders_are_counted() -> None:
    assert cid_placeholders("(cid:24)(cid:3) abc") == 2
    assert cid_placeholders("abc") == 0


def test_tokens_are_the_arabic_words_only() -> None:
    normalized = for_comparison("نظام الطالب")
    assert arabic_tokens(normalized + " (DN) GPA 3.75") == ["نظام", "الطالب"]


def test_punctuation_separates_tokens() -> None:
    assert arabic_tokens("(" + "ع" + ")") == ["ع"]


def test_words_run_together_on_the_page_stay_one_token() -> None:
    # Printed without spaces on library_services_2024_2025 page 10.
    assert arabic_tokens(for_comparison("والصوروالكثيرمن")) == ["والصوروالكثيرمن"]


def test_token_validity_is_the_share_of_known_words() -> None:
    assert token_validity(["نظام", "وؤيتنا"], frozenset({"نظام"})) == 0.5


def test_token_validity_is_none_without_tokens() -> None:
    assert token_validity([], frozenset({"نظام"})) is None


def test_single_letter_share() -> None:
    assert single_letter_share(["ع", "ح", "نظام", "الطالب"]) == 0.5
    assert single_letter_share([]) is None


def test_presentation_form_text_is_valid_once_normalized() -> None:
    # The student guide's failure mode: every letter is a display form, yet
    # the word is recoverable, so validity must be judged after normalization.
    quality = measure(NIZAM_PRESENTATION, build_lexicon(["نظام"]))
    # Three display forms and a base meem, the mix the guide's layer really stores.
    assert quality.presform_ratio == 0.75
    assert quality.token_validity == 1.0


def test_silently_wrong_word_is_caught_only_by_the_lexicon() -> None:
    # The guidance manual's failure mode, verified against its rendered page:
    # no presentation forms, no controls, nothing structural to see.
    quality = measure("وؤيتنا", build_lexicon(["رؤيتنا"]))
    assert quality.presform_ratio == 0.0
    assert quality.replacement_chars == 0
    assert quality.token_validity == 0.0


def test_cid_placeholders_disguise_themselves_as_content() -> None:
    # They count as Latin letters and digits, so content alone looks healthy.
    quality = measure("(cid:24)(cid:31)", frozenset())
    assert quality.cid_placeholders == 2
    assert quality.content_chars > 0
    assert quality.arabic_tokens == 0
    assert quality.token_validity is None


def test_invisible_characters_are_not_content() -> None:
    quality = measure("\u202b" + "نظام" + "\u202c", frozenset())
    assert quality.content_chars == 4
    assert quality.bidi_per_1k > 0


def test_empty_page_has_no_evidence_either_way() -> None:
    quality = measure("", frozenset())
    assert quality.content_chars == 0
    assert quality.arabic_tokens == 0
    assert quality.token_validity is None
    assert quality.single_letter_share is None


def test_every_metric_appears_in_the_dict() -> None:
    fields = set(PageQuality.__dataclass_fields__)
    assert set(measure("", frozenset()).to_dict()) == fields
