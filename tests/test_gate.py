"""Tests for the quality gate's verdicts.

Pages are described by their measurements directly, so no PDF or lexicon is
needed. The calibration test replays the five ground-truth pages: if the
threshold ever moves, it names the page that changed side.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from daleel.ingest import gate
from daleel.ingest.gate import (
    LEXICON_CUTOFF,
    MIN_ARABIC_WORDS,
    MIN_TOKEN_VALIDITY,
    RULES,
    Verdict,
    decide,
    gate_document,
    load_gate_lexicon,
)
from daleel.ingest.quality import PageQuality

SOUND = PageQuality(
    content_chars=1400,
    presform_ratio=0.0,
    bidi_per_1k=0.0,
    c0_per_1k=0.0,
    replacement_chars=0,
    cid_placeholders=0,
    arabic_tokens=290,
    token_validity=1.0,
    single_letter_share=0.01,
)


def test_a_sound_page_is_trusted() -> None:
    decision = decide(SOUND)
    assert decision.verdict == Verdict.TRUSTED
    assert decision.rejected_by == ()


@pytest.mark.parametrize(
    ("page", "validity", "expected"),
    [
        ("student_guide_2025_p30", 1.00, Verdict.TRUSTED),
        ("academic_weeks_1448_p01", 0.99, Verdict.TRUSTED),
        ("guidance_manual_p02", 0.88, Verdict.UNTRUSTED),
        ("library_services_2024_2025_p10", 0.81, Verdict.UNTRUSTED),
        ("orientation_1446_p03", 0.78, Verdict.UNTRUSTED),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_the_calibration_pages_get_the_verdicts_ground_truth_gives_them(
    page: str, validity: float, expected: Verdict
) -> None:
    # Measured validity at cutoff 1,000; truth from precision against ground truth.
    assert decide(replace(SOUND, token_validity=validity)).verdict == expected


def test_the_threshold_itself_passes() -> None:
    assert decide(replace(SOUND, token_validity=MIN_TOKEN_VALIDITY)).verdict == Verdict.TRUSTED


def test_just_under_the_threshold_fails() -> None:
    decision = decide(replace(SOUND, token_validity=MIN_TOKEN_VALIDITY - 0.001))
    assert decision.verdict == Verdict.UNTRUSTED
    assert decision.rejected_by == ("token_validity",)


def test_cid_placeholders_reject_even_when_every_word_is_real() -> None:
    decision = decide(replace(SOUND, cid_placeholders=1))
    assert decision.verdict == Verdict.UNTRUSTED
    assert decision.rejected_by == ("cid_placeholders",)


def test_replacement_characters_reject_even_when_every_word_is_real() -> None:
    decision = decide(replace(SOUND, replacement_chars=1))
    assert decision.rejected_by == ("replacement_characters",)


def test_every_failed_rule_is_recorded_not_just_the_first() -> None:
    decision = decide(replace(SOUND, cid_placeholders=3, token_validity=0.5))
    assert set(decision.rejected_by) == {"cid_placeholders", "token_validity"}


def test_a_footer_alone_is_not_enough_to_trust_a_page() -> None:
    # guidance_manual page 18: its 3-word footer is perfectly valid, and its
    # content is missing from the text layer entirely.
    decision = decide(replace(SOUND, arabic_tokens=3, token_validity=1.0))
    assert decision.verdict == Verdict.UNTRUSTED
    assert decision.rejected_by == ("too_few_words",)


def test_the_minimum_number_of_words_is_enough() -> None:
    decision = decide(replace(SOUND, arabic_tokens=MIN_ARABIC_WORDS))
    assert decision.verdict == Verdict.TRUSTED


def test_every_calibration_page_clears_the_word_minimum() -> None:
    # The smallest calibration page had 52 Arabic words.
    assert MIN_ARABIC_WORDS < 52


def test_a_page_without_arabic_words_has_nothing_to_judge() -> None:
    decision = decide(replace(SOUND, arabic_tokens=0, token_validity=None))
    assert decision.verdict == Verdict.NO_ARABIC_TEXT
    assert decision.rejected_by == ("no_arabic_words",)


def test_no_arabic_text_still_reports_other_damage() -> None:
    # Like a layer made entirely of cid placeholders.
    decision = decide(replace(SOUND, arabic_tokens=0, token_validity=None, cid_placeholders=12))
    assert decision.verdict == Verdict.NO_ARABIC_TEXT
    assert set(decision.rejected_by) == {"no_arabic_words", "cid_placeholders"}


def test_decision_as_plain_data() -> None:
    decision = decide(replace(SOUND, token_validity=0.5))
    assert decision.to_dict() == {"verdict": "untrusted", "rejected_by": ["token_validity"]}


def test_every_rule_states_its_meaning_and_evidence() -> None:
    for rule in RULES:
        assert rule.meaning
        assert rule.evidence.startswith(("measured", "principle", "judgement"))


def test_rule_names_are_unique() -> None:
    names = [rule.name for rule in RULES]
    assert len(names) == len(set(names))


def test_gate_lexicon_uses_the_calibrated_cutoff(monkeypatch: pytest.MonkeyPatch) -> None:
    # Replace the real loader so the test needs no 69 MB file, and record its call.
    calls = []

    def fake_load(zip_path, *, min_frequency):
        calls.append(min_frequency)
        return frozenset()

    monkeypatch.setattr(gate, "load_lexicon", fake_load)
    load_gate_lexicon()
    assert calls == [LEXICON_CUTOFF]


def test_every_page_of_a_document_gets_a_numbered_verdict(make_pdf: Callable[..., Path]) -> None:
    # Latin text and a blank page: neither holds Arabic words to judge.
    verdicts = gate_document(make_pdf(["first page", ""]), frozenset())
    assert [v.page for v in verdicts] == [1, 2]
    assert {v.decision.verdict for v in verdicts} == {Verdict.NO_ARABIC_TEXT}
