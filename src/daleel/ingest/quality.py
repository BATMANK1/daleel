"""Measure how far a page's text layer can be trusted, without deciding.

Every metric here is computable from the extracted text alone, because almost
no page will ever have ground truth. Ground truth exists to check that these
numbers predict what the page really says; turning them into a verdict is the
quality gate's job, not this module's.

Character measurements run on the raw text, where encoding damage is visible.
Word measurements run on normalized text, compared against a lexicon that has
been normalized the same way, so that differences in encoding never count as
differences in words.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from daleel.ingest.inventory import analyse_text
from daleel.normalize.arabic import for_comparison

# pdfminer writes this placeholder for a glyph it cannot map to a character.
# It reads as Latin text, which hides the damage from the Arabic measurements.
CID_PLACEHOLDER = re.compile(r"\(cid:\d+\)")

# Base Arabic letters as they remain after normalization: hamza and its seats,
# the alphabet, and alef maqsura and yaa, but not tatweel.
ARABIC_WORD = re.compile(r"[\u0621-\u063a\u0641-\u064a]+")


@dataclass(frozen=True)
class PageQuality:
    """What a page's text layer looks like, measured. No verdict."""

    content_chars: int
    presform_ratio: float
    bidi_per_1k: float
    c0_per_1k: float
    replacement_chars: int
    cid_placeholders: int
    arabic_tokens: int
    token_validity: float | None
    single_letter_share: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def build_lexicon(words: Iterable[str]) -> frozenset[str]:
    """Normalize a word list exactly as page text is normalized."""
    return frozenset(w for w in (for_comparison(word) for word in words) if w)


def cid_placeholders(raw: str) -> int:
    """Count glyphs the extractor could not map to any character."""
    return len(CID_PLACEHOLDER.findall(raw))


def arabic_tokens(normalized: str) -> list[str]:
    """The Arabic words in normalized text. Digits, Latin and punctuation split them."""
    return ARABIC_WORD.findall(normalized)


def token_validity(tokens: list[str], lexicon: frozenset[str]) -> float | None:
    """Share of tokens that are real words, or None when there are no tokens."""
    if not tokens:
        return None
    return sum(token in lexicon for token in tokens) / len(tokens)


def single_letter_share(tokens: list[str]) -> float | None:
    """Share of tokens that are a lone letter, or None when there are no tokens."""
    if not tokens:
        return None
    return sum(len(token) == 1 for token in tokens) / len(tokens)


def measure(raw: str, lexicon: frozenset[str]) -> PageQuality:
    """Measure one page's extracted text against a normalized lexicon."""
    stats = analyse_text(raw)
    tokens = arabic_tokens(for_comparison(raw))
    return PageQuality(
        content_chars=(
            stats.arabic_total
            + stats.latin_letters
            + stats.ascii_digits
            + stats.arabic_indic_digits
        ),
        presform_ratio=stats.presform_ratio,
        bidi_per_1k=stats.bidi_per_1k,
        c0_per_1k=stats.c0_per_1k,
        replacement_chars=stats.replacement_chars,
        cid_placeholders=cid_placeholders(raw),
        arabic_tokens=len(tokens),
        token_validity=token_validity(tokens, lexicon),
        single_letter_share=single_letter_share(tokens),
    )
