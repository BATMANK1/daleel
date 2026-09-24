"""Tests for the command line interface.

The commands read PDFs, which CI doesn't have, so these tests cover everything
that can be checked without them: the pure functions that turn metadata into
output, argument parsing, and the error paths of both commands.
"""

from __future__ import annotations

import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from daleel.cli import (
    build_parser,
    format_route_table,
    gate_page_row,
    gate_summary,
    main,
    route_row,
)
from daleel.ingest.gate import PageVerdict, decide
from daleel.ingest.metadata import PdfMetadata
from daleel.ingest.quality import PageQuality

COMMANDS = ["inventory", "route", "gate"]


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


def _page(number: int, validity: float | None, tokens: int = 100) -> PageVerdict:
    quality = PageQuality(
        content_chars=500,
        presform_ratio=0.0,
        bidi_per_1k=0.0,
        c0_per_1k=0.0,
        replacement_chars=0,
        cid_placeholders=0,
        arabic_tokens=tokens,
        token_validity=validity,
        single_letter_share=0.0,
    )
    return PageVerdict(page=number, quality=quality, decision=decide(quality))


def test_gate_command_is_registered() -> None:
    assert build_parser().parse_args(["gate", "data/raw"]).command == "gate"


def test_gate_summary_counts_verdicts_and_checks_the_route() -> None:
    pages = [_page(1, 1.0), _page(2, 0.99), _page(3, 0.5), _page(4, None, tokens=0)]
    summary = gate_summary("guide.pdf", "text_layer", pages)
    assert summary["trusted"] == 2
    assert summary["untrusted"] == 1
    assert summary["no_arabic_text"] == 1
    # Two of four trusted is not more than half, so the gate says OCR.
    assert summary["gate_path"] == "ocr"
    assert summary["agrees"] == "no"


def test_gate_summary_agrees_when_most_pages_are_trusted() -> None:
    summary = gate_summary(
        "guide.pdf", "text_layer", [_page(1, 1.0), _page(2, 0.99), _page(3, 0.5)]
    )
    assert (summary["gate_path"], summary["agrees"]) == ("text_layer", "yes")


def test_gate_page_row_shows_validity_and_reasons() -> None:
    row = gate_page_row("guide.pdf", _page(7, 0.5))
    assert row["page"] == 7
    assert row["validity"] == "50%"
    assert row["rejected_by"] == "token_validity"


def test_trusted_page_row_has_no_reasons() -> None:
    assert gate_page_row("guide.pdf", _page(1, 1.0))["rejected_by"] is None


def test_gate_explains_a_missing_lexicon(
    make_pdf: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_pdf(["first page"])
    assert main(["gate", str(tmp_path), "--lexicon", str(tmp_path / "absent.zip")]) == 2
    assert "fetch_lexicon.py" in capsys.readouterr().err


def test_gate_runs_end_to_end(
    make_pdf: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_pdf(["first page", ""], name="guide.pdf")
    lexicon = tmp_path / "lexicon.zip"
    with zipfile.ZipFile(lexicon, "w") as archive:
        archive.writestr("MSA_freq_lists.tsv", "\u0641\u064a\t5000\n")
    assert main(["gate", str(tmp_path), "--lexicon", str(lexicon)]) == 0
    out = capsys.readouterr().out
    assert "guide.pdf" in out
    assert "no arabic text" in out


def test_inventory_reads_with_pypdfium2_by_default() -> None:
    assert build_parser().parse_args(["inventory", "data/raw"]).backend == "pypdfium2"


def test_inventory_rejects_an_unknown_backend() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["inventory", "data/raw", "--backend", "pdftotext"])
