"""Taking what could identify a student out of the text they send.

A question about the regulations needs no student number, national ID,
phone number or e-mail address, so these are replaced before the question
reaches search, the model or the query log.
"""

from __future__ import annotations

import re

REDACTED = "[redacted]"

# Arabic-Indic and Eastern Arabic-Indic digits, read as 0 to 9 for matching.
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_EMAIL = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")
# Nine or more digits written in groups, as a phone number is, or seven or more in a row.
_GROUPED = re.compile(r"(?<![\d+])\+?\d(?:[ -]?\d){8,}(?!\d)")
_RUN = re.compile(r"(?<!\d)\d{7,}(?!\d)")


def redact(text: str) -> str:
    """The text with e-mail addresses, phone numbers and runs of seven or more digits replaced."""
    for pattern in (_EMAIL, _GROUPED, _RUN):
        # The digits are read as 0 to 9 one for one, so the places found fit the text.
        spans = [found.span() for found in pattern.finditer(text.translate(_DIGITS))]
        for start, end in reversed(spans):
            text = text[:start] + REDACTED + text[end:]
    return text
