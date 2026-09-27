"""Tests for producer-based extraction routing.

The corpus table uses metadata copied verbatim from real documents, so it
encodes what the inventory measured, not what the code happens to do. The rule
tests below reuse those strings wherever they can, and say so when they can't.
"""

from __future__ import annotations

import pytest

from daleel.ingest.router import ExpectedFailure, ExtractionPath, route


def outcome(*, producer: str, creator: str) -> tuple[ExtractionPath, ExpectedFailure, str | None]:
    """Everything a route decides: path, expected failure, and the field that matched."""
    result = route(producer=producer, creator=creator)
    return result.path, result.expected_failure, result.matched_on


@pytest.mark.parametrize(
    ("producer", "creator", "path", "failure", "matched"),
    [
        # student_guide_2025.pdf: 77% presentation forms through pdfplumber.
        # PDFium is the PDF viewer that wrote this copy before Student Affairs
        # posted it, not the design tool behind it.
        (
            "PDFium",
            "PDFium",
            ExtractionPath.TEXT_LAYER,
            ExpectedFailure.PRESENTATION_FORMS,
            "producer",
        ),
        # academic_weeks_1448.pdf, the official calendar: 71% presentation forms
        # through pdfplumber. Note the lowercase "library": a real document
        # already needs case-insensitive matching, not just a hypothetical one.
        (
            "Adobe PDF library 18.00",
            "Adobe Illustrator 30.4 (Windows)",
            ExtractionPath.TEXT_LAYER,
            ExpectedFailure.PRESENTATION_FORMS,
            "producer",
        ),
        # guidance_manual.pdf: 0% presentation forms, yet words come out wrong.
        (
            "Microsoft® Word 2019",
            "Microsoft® Word 2019",
            ExtractionPath.OCR,
            ExpectedFailure.LOSSY_SUBSTITUTION,
            "producer",
        ),
        # library_services_2024_2025.pdf: 0% presentation forms, scrambled text.
        (
            "Microsoft® PowerPoint® 2019",
            "Microsoft® PowerPoint® 2019",
            ExtractionPath.OCR,
            ExpectedFailure.LOSSY_SUBSTITUTION,
            "producer",
        ),
        # orientation_1446.pdf: at first a prediction only, since it shares the
        # guidance manual's producer. Ground truth has since confirmed it: its
        # words match the page at 33% precision.
        (
            "Microsoft® Word 2019",
            "Microsoft® Word 2019",
            ExtractionPath.OCR,
            ExpectedFailure.LOSSY_SUBSTITUTION,
            "producer",
        ),
        # student_conduct_code.pdf: 75% presentation forms through pdfplumber.
        # Out of sample: the rule was written from library version 18.00.
        (
            "Adobe PDF library 15.00",
            "Adobe Illustrator 24.0 (Windows)",
            ExtractionPath.TEXT_LAYER,
            ExpectedFailure.PRESENTATION_FORMS,
            "producer",
        ),
        # student_charter.pdf: 76% presentation forms through pdfplumber. Also
        # out of sample, with a creator a decade older than any the rule saw.
        (
            "Adobe PDF library 15.00",
            "Adobe Illustrator CC 2017 (Windows)",
            ExtractionPath.TEXT_LAYER,
            ExpectedFailure.PRESENTATION_FORMS,
            "producer",
        ),
        # organizational_regulations.pdf, as downloaded from the college's site,
        # carries no metadata at all, so it takes the default. The gate trusts
        # 43 of its 44 pages that have a text layer.
        (
            "",
            "",
            ExtractionPath.TEXT_LAYER,
            ExpectedFailure.UNKNOWN,
            None,
        ),
        # student_portal_guide.pdf was compressed by an online tool before it
        # was published, which replaced its metadata. It takes the default too,
        # and the gate trusts 37 of its 41 pages that have a text layer.
        (
            "GPL Ghostscript 10.00.0",
            "pdfresizer.com",
            ExtractionPath.TEXT_LAYER,
            ExpectedFailure.UNKNOWN,
            None,
        ),
    ],
    ids=[
        "student_guide",
        "academic_weeks_official",
        "guidance_manual",
        "library_services",
        "orientation",
        "student_conduct_code",
        "student_charter",
        "organizational_regulations",
        "student_portal_guide",
    ],
)
def test_corpus_documents_route_as_measured(
    producer: str,
    creator: str,
    path: ExtractionPath,
    failure: ExpectedFailure,
    matched: str | None,
) -> None:
    assert outcome(producer=producer, creator=creator) == (path, failure, matched)


def test_creator_is_the_fallback_when_producer_is_unrecognised() -> None:
    # Real strings: the unofficial student calendar, since removed from the
    # corpus. Its producer is a bare version string; only the creator names it.
    assert outcome(
        producer="3.0.38 (5.1.23)",
        creator="Adobe Illustrator 28.2 (Windows)",
    ) == (
        ExtractionPath.TEXT_LAYER,
        ExpectedFailure.PRESENTATION_FORMS,
        "creator",
    )


def test_producer_takes_precedence_over_creator() -> None:
    # Synthetic: no document in the corpus has this combination. This encodes
    # a design choice (the producer wrote the text encoding), not a measurement.
    assert outcome(
        producer="Adobe PDF library 18.00",
        creator="Microsoft® Word 2019",
    ) == (
        ExtractionPath.TEXT_LAYER,
        ExpectedFailure.PRESENTATION_FORMS,
        "producer",
    )


def test_unknown_producer_defaults_to_text_layer() -> None:
    # Synthetic: software never seen in the corpus.
    assert outcome(producer="LibreOffice 7.5", creator="Writer") == (
        ExtractionPath.TEXT_LAYER,
        ExpectedFailure.UNKNOWN,
        None,
    )


def test_empty_metadata_defaults_to_text_layer() -> None:
    # Real: organizational_regulations.pdf, as downloaded from the college's
    # site, omits both fields, and missing fields are read as empty strings.
    assert outcome(producer="", creator="") == (
        ExtractionPath.TEXT_LAYER,
        ExpectedFailure.UNKNOWN,
        None,
    )


def test_other_microsoft_software_is_not_assumed_lossy() -> None:
    # Synthetic. The evidence covers Word and PowerPoint only, so the rule must
    # not stretch to Microsoft software in general.
    assert outcome(producer="Microsoft: Print To PDF", creator="") == (
        ExtractionPath.TEXT_LAYER,
        ExpectedFailure.UNKNOWN,
        None,
    )


@pytest.mark.parametrize(
    ("variant", "proper", "path", "failure"),
    [
        (
            "pdfium",
            "PDFium",
            ExtractionPath.TEXT_LAYER,
            ExpectedFailure.PRESENTATION_FORMS,
        ),
        (
            "MICROSOFT WORD",
            "Microsoft Word",
            ExtractionPath.OCR,
            ExpectedFailure.LOSSY_SUBSTITUTION,
        ),
    ],
    ids=["pdfium", "microsoft_word"],
)
def test_matching_is_case_insensitive(
    variant: str,
    proper: str,
    path: ExtractionPath,
    failure: ExpectedFailure,
) -> None:
    # Synthetic spellings. The real documents' spellings are tested above.
    expected = (path, failure, "producer")
    assert outcome(producer=proper, creator="") == expected
    assert outcome(producer=variant, creator="") == expected


@pytest.mark.parametrize(
    ("with_symbols", "without_symbols"),
    [
        ("Microsoft® Word 2019", "Microsoft Word 2019"),
        ("Microsoft® PowerPoint® 2019", "Microsoft PowerPoint 2019"),
    ],
    ids=["word", "powerpoint"],
)
def test_trademark_symbols_do_not_affect_matching(with_symbols: str, without_symbols: str) -> None:
    # The strings with symbols are exactly as the inventory reported them.
    expected = (
        ExtractionPath.OCR,
        ExpectedFailure.LOSSY_SUBSTITUTION,
        "producer",
    )
    assert outcome(producer=with_symbols, creator="") == expected
    assert outcome(producer=without_symbols, creator="") == expected
