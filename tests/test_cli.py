# Tests for the command line interface.

"""The commands read PDFs, which CI doesn't have, so these tests cover everything
that can be checked without them: the pure functions that turn metadata into
output, argument parsing, and the error paths of both commands.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daleel.cli import build_parser, format_route_table, main, route_row
from daleel.ingest.metadata import PdfMetadata

COMMANDS = ["inventory", "route"]


def test_route_row_reports_the_route_for_a_document() -> None:
    # Real strings: guidance_manual.pdf.
    meta = PdfMetadata(producer="Microsoft® Word 2019", creator="Microsoft® Word 2019")
    assert route_row("guidance_manual.pdf", meta) == {
        "file": "guidance_manual.pdf",
        "producer": "Microsoft® Word 2019",
        "creator": "Microsoft® Word 2019",
        "path": "ocr",
        "expected_failure": "lossy_substitution",
        "rule": "microsoft_office",
        "matched_on": "producer",
    }


def test_route_row_for_unknown_software_shows_no_match() -> None:
    # Synthetic: software never seen in the corpus.
    row = route_row("other.pdf", PdfMetadata(producer="LibreOffice 7.5", creator="Writer"))
    assert row["rule"] == "default"
    assert row["matched_on"] is None


def test_route_table_shows_a_dash_where_nothing_matched() -> None:
    row = route_row("other.pdf", PdfMetadata(producer="LibreOffice 7.5", creator="Writer"))
    last_line = format_route_table([row]).splitlines()[-1]
    assert last_line.split()[-1] == "-"


def test_route_command_is_registered() -> None:
    assert build_parser().parse_args(["route", "data/raw"]).command == "route"


@pytest.mark.parametrize("command", COMMANDS)
def test_missing_directory_exits_with_2(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([command, str(tmp_path / "missing")]) == 2
    assert "does not exist" in capsys.readouterr().err


@pytest.mark.parametrize("command", COMMANDS)
def test_file_instead_of_directory_exits_with_2(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    not_a_directory = tmp_path / "notes.txt"
    not_a_directory.write_text("not a directory", encoding="utf-8")
    assert main([command, str(not_a_directory)]) == 2
    assert "is not a directory" in capsys.readouterr().err


@pytest.mark.parametrize("command", COMMANDS)
def test_directory_without_pdfs_exits_with_1(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([command, str(tmp_path)]) == 1
    assert "no PDFs" in capsys.readouterr().err
