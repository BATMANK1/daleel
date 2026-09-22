# Route each PDF to the extraction path most likely to work, from its metadata.

"""A route is a prior, not a verdict. It predicts which extraction path to try
first and which failure to expect, from the software that produced the PDF.
The quality gate still checks every document, so a wrong prediction costs time
and never correctness.

The rules are built from the documents measured so far, and each one records
which documents it rests on. They are observations about this corpus, not
general truths about the software they name. Software with no evidence behind
it, such as other Microsoft or Adobe products, falls to the default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class ExtractionPath(StrEnum):
    TEXT_LAYER = "text_layer"
    OCR = "ocr"


class ExpectedFailure(StrEnum):
    PRESENTATION_FORMS = "presentation_forms"
    LOSSY_SUBSTITUTION = "lossy_substitution"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    path: ExtractionPath
    expected_failure: ExpectedFailure
    evidence: str


@dataclass(frozen=True)
class Route:
    path: ExtractionPath
    expected_failure: ExpectedFailure
    rule: str
    # "producer" or "creator": which metadata field the rule matched.
    # None means nothing matched and the default applied.
    matched_on: str | None


RULES: tuple[Rule, ...] = (
    Rule(
        name="pdfium",
        pattern=re.compile(r"\bpdfium\b"),
        path=ExtractionPath.TEXT_LAYER,
        expected_failure=ExpectedFailure.PRESENTATION_FORMS,
        evidence="student_guide_2025.pdf: 77% presentation forms",
    ),
    Rule(
        name="adobe",
        pattern=re.compile(r"\badobe\b.*\b(pdf library|illustrator)\b"),
        path=ExtractionPath.TEXT_LAYER,
        expected_failure=ExpectedFailure.PRESENTATION_FORMS,
        evidence=(
            "academic_weeks_1448.pdf: 71% presentation forms, matched on producer; "
            "the unofficial calendar: 61%, matched on creator"
        ),
    ),
    Rule(
        name="microsoft_office",
        pattern=re.compile(r"\bmicrosoft\b.*\b(word|powerpoint)\b"),
        path=ExtractionPath.OCR,
        expected_failure=ExpectedFailure.LOSSY_SUBSTITUTION,
        evidence=(
            "guidance_manual.pdf: letters substituted, verified against the page; "
            "library_services_2024_2025.pdf: scrambled text"
        ),
    ),
)

DEFAULT = Route(
    path=ExtractionPath.TEXT_LAYER,
    expected_failure=ExpectedFailure.UNKNOWN,
    rule="default",
    matched_on=None,
)

# Trademark symbols appear in real metadata ("Microsoft® Word 2019") and carry
# no information, so they are removed before matching.
_SYMBOLS = str.maketrans("", "", "®™©")


def _normalise(value: str) -> str:
    return " ".join(value.translate(_SYMBOLS).lower().split())


def _match(value: str) -> Rule | None:
    text = _normalise(value)
    for rule in RULES:
        if rule.pattern.search(text):
            return rule
    return None


def route(*, producer: str, creator: str) -> Route:
    """Predict the extraction path for a PDF from its producer and creator.

    The producer is tried first because it is the software that wrote the text
    encoding, which is what determines how extraction fails. The creator is the
    fallback for producers that name nothing recognisable, such as a bare
    version string.
    """
    for field, value in (("producer", producer), ("creator", creator)):
        rule = _match(value)
        if rule is not None:
            return Route(
                path=rule.path,
                expected_failure=rule.expected_failure,
                rule=rule.name,
                matched_on=field,
            )
    return DEFAULT
