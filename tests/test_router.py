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
    ("producer", "creator", "path", "failure"),
    [
        # student_guide_2025.pdf: 77% presentation forms.
        (
            "PDFium",
            "PDFium",
            ExtractionPath.TEXT_LAYER,
            ExpectedFailure.PRESENTATION_FORMS,
        ),
        # academic_weeks_1448.pdf, the official calendar: 71% presentation forms.
        # Note the lowercase "library": a real document already needs
        # case-insensitive matching, not just a hypothetical one.
        (
            "Adobe PDF library 18.00",
            "Adobe Illustrator 30.4 (Windows)",
            ExtractionPath.TEXT_LAYER,
            ExpectedFailure.PRESENTATION_FORMS,
        ),
        # guidance_manual.pdf: 0% presentation forms, yet words come out wrong.
        (
            "Microsoft® Word 2019",
            "Microsoft® Word 2019",
            ExtractionPath.OCR,
            ExpectedFailure.LOSSY_SUBSTITUTION,
        ),
        # library_services_2024_2025.pdf: 0% presentation forms, scrambled text.
        (
            "Microsoft® PowerPoint® 2019",
            "Microsoft® PowerPoint® 2019",
            ExtractionPath.OCR,
            ExpectedFailure.LOSSY_SUBSTITUTION,
        ),
        # orientation_1446.pdf is left out on purpose. It has the same producer
        # as guidance_manual, but its lossy substitution is still a prediction
        # nobody has checked against ground truth. Add it once that's done.
    ],
    ids=[
        "student_guide",
        "academic_weeks_official",
        "guidance_manual",
        "library_services",
    ],
)
def test_corpus_documents_route_as_measured(
    producer: str,
    creator: str,
    path: ExtractionPath,
    failure: ExpectedFailure,
) -> None:
    # Every document in the current corpus names its software in the producer.
    assert outcome(producer=producer, creator=creator) == (path, failure, "producer")


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
    # Synthetic but realistic: PDFs can omit both fields, and the inventory
    # reads a missing field as an empty string.
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
