"""Tests for taking what could identify a student out of a question."""

from __future__ import annotations

import pytest

from daleel.serve.privacy import REDACTED, redact


@pytest.mark.parametrize(
    ("asked", "kept"),
    [
        # A student number, and a national ID in Arabic-Indic digits.
        ("رقمي الجامعي 441234567 كم غيابي؟", f"رقمي الجامعي {REDACTED} كم غيابي؟"),
        ("هويتي ١٠٢٣٤٥٦٧٨٩ هل تقبل؟", f"هويتي {REDACTED} هل تقبل؟"),
        # Mobile numbers, written whole or in groups.
        ("جوالي 0551234567", f"جوالي {REDACTED}"),
        ("جوالي +966 55 123 4567 اتصلوا", f"جوالي {REDACTED} اتصلوا"),
        ("رقمي 055-123-4567", f"رقمي {REDACTED}"),
        # An e-mail address, digits and all.
        ("راسلني على s441234567@rcyci.edu.sa رجاء", f"راسلني على {REDACTED} رجاء"),
    ],
)
def test_identifiers_are_replaced(asked: str, kept: str) -> None:
    assert redact(asked) == kept


@pytest.mark.parametrize(
    "asked",
    [
        "كم رسوم الوحدة 250 ريال في 2025؟",
        "متى التسجيل من 1448/04/09 إلى 1448/07/01؟",
        "هل يحرم الطالب إذا تجاوز غيابه 20%؟",
        "ما متطلبات المقرر CS 101 والمقرر 2125؟",
        "What is the fee per credit hour in 2025?",
    ],
)
def test_the_numbers_of_a_question_about_the_rules_are_kept(asked: str) -> None:
    assert redact(asked) == asked
