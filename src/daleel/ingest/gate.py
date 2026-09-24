"""Decide whether a page's text layer can be trusted.

daleel.ingest.quality measures; this module decides. Each rule names what
failing it means and where its threshold comes from, marked either as
measured, calibrated against hand-transcribed ground truth, or as a principle,
true by definition and needing no calibration.

The calibration behind the measured rule used five pages, two with sound text
layers and three broken ones, with text extracted by pypdfium2 and a lexicon
cut at 1,000 occurrences. The threshold means nothing outside that
combination, which is why load_gate_lexicon fixes the cutoff. Five pages make
the threshold provisional: more ground truth would narrow it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from daleel.ingest.extract import page_texts
from daleel.ingest.lexicon import LEXICON_ZIP, load_lexicon
from daleel.ingest.quality import PageQuality, measure

# Sound pages scored at least 99% and broken ones at most 88%. The threshold
# sits above that gap's midpoint on purpose: passing a broken page puts wrong
# words in front of a reader, while rejecting a sound one costs an OCR run.
MIN_TOKEN_VALIDITY = 0.95
LEXICON_CUTOFF = 1_000


class Verdict(StrEnum):
    TRUSTED = "trusted"  # use the text layer
    UNTRUSTED = "untrusted"  # send the page to OCR
    NO_ARABIC_TEXT = "no_arabic_text"  # nothing the gate can judge; send to OCR


@dataclass(frozen=True)
class GateRule:
    name: str
    fails: Callable[[PageQuality], bool]
    meaning: str
    evidence: str


RULES: tuple[GateRule, ...] = (
    GateRule(
        name="no_arabic_words",
        fails=lambda q: q.arabic_tokens == 0,
        meaning="the text layer holds no Arabic words at all",
        evidence="principle: with no words there is nothing to judge",
    ),
    GateRule(
        name="cid_placeholders",
        fails=lambda q: q.cid_placeholders > 0,
        meaning="the extractor met glyphs it could not map to characters",
        evidence="principle: each placeholder is an unread glyph; none occur in the corpus",
    ),
    GateRule(
        name="replacement_characters",
        fails=lambda q: q.replacement_chars > 0,
        meaning="the extractor met bytes it could not decode",
        evidence="principle: each replacement is an unread character; none occur in the corpus",
    ),
    GateRule(
        name="token_validity",
        fails=lambda q: q.token_validity is not None and q.token_validity < MIN_TOKEN_VALIDITY,
        meaning=f"under {MIN_TOKEN_VALIDITY:.0%} of the Arabic words are real words",
        evidence=(
            "measured on five ground-truth pages at lexicon cutoff 1,000: sound pages "
            ">= 99%, broken pages <= 88%; set above the midpoint to lean toward rejection"
        ),
    ),
)


@dataclass(frozen=True)
class GateDecision:
    verdict: Verdict
    rejected_by: tuple[str, ...]

    def to_dict(self) -> dict[str, str | list[str]]:
        return {"verdict": self.verdict.value, "rejected_by": list(self.rejected_by)}


def decide(quality: PageQuality) -> GateDecision:
    """Judge a measured page, recording every rule it fails."""
    failed = tuple(rule.name for rule in RULES if rule.fails(quality))
    if "no_arabic_words" in failed:
        return GateDecision(Verdict.NO_ARABIC_TEXT, failed)
    if failed:
        return GateDecision(Verdict.UNTRUSTED, failed)
    return GateDecision(Verdict.TRUSTED, ())


def load_gate_lexicon(zip_path: Path = LEXICON_ZIP) -> frozenset[str]:
    """The lexicon at the cutoff the threshold was calibrated for."""
    return load_lexicon(zip_path, min_frequency=LEXICON_CUTOFF)


@dataclass(frozen=True)
class PageVerdict:
    """One page's measurements and the verdict drawn from them."""

    page: int
    quality: PageQuality
    decision: GateDecision

    def to_dict(self) -> dict:
        return {"page": self.page, **self.decision.to_dict(), **self.quality.to_dict()}


def gate_document(path: Path, lexicon: frozenset[str]) -> list[PageVerdict]:
    """Extract, measure and judge every page of a PDF, the way the gate was calibrated.

    Pass the lexicon from load_gate_lexicon: the threshold means nothing with any other.
    """
    verdicts = []
    for number, text in enumerate(page_texts(path), start=1):
        quality = measure(text, lexicon)
        verdicts.append(PageVerdict(page=number, quality=quality, decision=decide(quality)))
    return verdicts
