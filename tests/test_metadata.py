# Tests for reading producer and creator from PDF metadata.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from daleel.ingest.metadata import PdfMetadata, metadata_from, metadata_value


def test_none_metadata_returns_empty_string() -> None:
    assert metadata_value(None, "Producer") == ""


def test_missing_key_returns_empty_string() -> None:
    metadata = {"Creator": "Microsoft Word"}
    assert metadata_value(metadata, "Producer") == ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"PDFium", "PDFium"),
        (b"Microsoft\xc2\xae Word", "Microsoft® Word"),
        (b"Bad\xffBytes", "Bad\ufffdBytes"),
    ],
    ids=["ascii", "utf8_trademark", "invalid_utf8_is_replaced"],
)
def test_bytes_value_gets_decoded(raw: bytes, expected: str) -> None:
    assert metadata_value({"Producer": raw}, "Producer") == expected


def test_surrounding_whitespace_is_stripped() -> None:
    metadata = {"Producer": "   Adobe PDF library 18.00 \n\t"}
    assert metadata_value(metadata, "Producer") == "Adobe PDF library 18.00"


def test_existing_string_returns_unchanged() -> None:
    assert metadata_value({"Producer": "PDFium"}, "Producer") == "PDFium"


def test_non_string_raw_values_are_cast_to_string() -> None:
    assert metadata_value({"Version": 1.5}, "Version") == "1.5"
    assert metadata_value({"IsEncrypted": False}, "IsEncrypted") == "False"


def test_metadata_from_reads_producer_and_creator() -> None:
    # A stand-in with a .metadata mapping is all metadata_from needs, which is
    # what lets this run in CI, where there are no PDFs.
    fake_pdf = SimpleNamespace(metadata={"Producer": "PDFium", "Creator": "PDFium"})
    assert metadata_from(fake_pdf) == PdfMetadata(producer="PDFium", creator="PDFium")


def test_metadata_from_gives_empty_strings_when_fields_are_missing() -> None:
    # The router's empty-metadata test assumes missing fields arrive as "".
    # This is the test that makes that assumption true.
    fake_pdf = SimpleNamespace(metadata=None)
    assert metadata_from(fake_pdf) == PdfMetadata(producer="", creator="")
