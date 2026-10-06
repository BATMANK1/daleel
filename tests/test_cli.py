"""Tests for the command line interface.

The commands read PDFs, which CI doesn't have, so these tests cover everything
that can be checked without them: the pure functions that turn metadata into
output, argument parsing, and every command's error paths. Small generated PDFs
stand in where a command must read one, and a stand-in for dots.mocr where it
must read a page by OCR.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
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
from daleel.ingest import extract
from daleel.ingest import records as records_module
from daleel.ingest.calendar import CalendarPage, CalendarRow
from daleel.ingest.gate import PageVerdict, decide
from daleel.ingest.metadata import PdfMetadata
from daleel.ingest.quality import PageQuality
from daleel.ingest.records import read_records
from daleel.ocr import dots
from daleel.ocr.engine import Block, Result

COMMANDS = ["inventory", "route", "gate", "extract"]


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
    # Two of the three pages with Arabic text are trusted, so the gate says text
    # layer. The page with none is counted, but says nothing about the layer.
    assert summary["gate_path"] == "text_layer"
    assert summary["agrees"] == "yes"


def test_pages_without_arabic_text_do_not_count_against_the_layer() -> None:
    # student_charter.pdf: its cover, a photo, two title pages and its back cover
    # have no Arabic in their text layer; its three pages of content are trusted.
    pages = [_page(n, None, tokens=0) for n in (1, 2, 3, 7, 8)]
    pages += [_page(n, 1.0) for n in (4, 5, 6)]
    summary = gate_summary("student_charter.pdf", "text_layer", pages)
    assert (summary["trusted"], summary["no_arabic_text"]) == (3, 5)
    assert (summary["gate_path"], summary["agrees"]) == ("text_layer", "yes")


def test_gate_says_ocr_when_most_pages_with_text_are_untrusted() -> None:
    # guidance_manual.pdf: one trusted page among nineteen.
    pages = [_page(1, 1.0)] + [_page(n, 0.5) for n in range(2, 20)]
    summary = gate_summary("guidance_manual.pdf", "ocr", pages)
    assert (summary["gate_path"], summary["agrees"]) == ("ocr", "yes")


def test_a_document_without_arabic_text_goes_to_ocr() -> None:
    pages = [_page(n, None, tokens=0) for n in (1, 2)]
    assert gate_summary("scan.pdf", "text_layer", pages)["gate_path"] == "ocr"


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


@pytest.mark.skipif(shutil.which("head") is None, reason="needs the head command")
def test_output_cut_short_by_a_pipe_ends_quietly(tmp_path: Path) -> None:
    # A real pipe into a real `head`, with enough output to overflow the pipe's
    # buffer, so the reader is gone before the writer has finished.
    script = tmp_path / "flood.py"
    script.write_text(
        "import sys\n"
        "from daleel import cli\n"
        "def flood(args):\n"
        "    for number in range(200_000):\n"
        "        print('line', number)\n"
        "    return 0\n"
        "cli._COMMANDS['route'] = flood\n"
        "sys.exit(cli.main(['route', '.']))\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        f'"{sys.executable}" "{script}" | head -n 1',
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout == "line 0\n"
    assert "Traceback" not in result.stderr


def _lexicon(folder: Path) -> Path:
    path = folder / "lexicon.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("MSA_freq_lists.tsv", "\u0641\u064a\t5000\n")
    return path


class StandInDots:
    """Stands in for dots.mocr on a vLLM server, counting the pages it reads."""

    readings = 0

    def __init__(self, settings: dots.Settings) -> None:
        self.settings = settings

    def tag(self) -> str:
        return "dots-stand-in"

    def describe(self) -> str:
        return "a stand-in for dots.mocr"

    def recognize(self, png: bytes) -> Result:
        StandInDots.readings += 1
        return Result(text="نص", seconds=150.0)


class StandInDotsWithAPicture(StandInDots):
    """Reads a slide whose steps sit inside a picture, as dots.mocr read the library's page 14."""

    def text_tag(self) -> str:
        return "dots-stand-in-text"

    def recognize(self, png: bytes) -> Result:
        title = Block("Section-header", "طريقة حجز القاعات الدراسية", (0.3, 0.16, 0.7, 0.27))
        picture = Block("Picture", "", (0.26, 0.35, 0.8, 0.78))
        return Result(text=title.text, seconds=21.0, blocks=(title, picture))

    def read_text(self, png: bytes) -> Result:
        return Result(text="مسح كود حجز القاعات الدراسية", seconds=14.0)


def test_extract_command_is_registered() -> None:
    assert build_parser().parse_args(["extract", "data/raw"]).command == "extract"


def test_extract_names_a_document_it_cannot_find(
    make_pdf: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_pdf(["first page"], name="guide.pdf")
    arguments = [
        "extract",
        str(tmp_path),
        "--only",
        "charter",
        "--lexicon",
        str(_lexicon(tmp_path)),
    ]
    assert main(arguments) == 2
    assert "no charter in" in capsys.readouterr().err


def test_extract_explains_a_missing_lexicon(
    make_pdf: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_pdf(["first page"])
    assert main(["extract", str(tmp_path), "--lexicon", str(tmp_path / "absent.zip")]) == 2
    assert "fetch_lexicon.py" in capsys.readouterr().err


def test_extract_writes_records_and_reads_no_page_twice(
    make_pdf: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(dots, "DotsEngine", StandInDots)
    monkeypatch.setattr(StandInDots, "readings", 0)
    pdfs = tmp_path / "raw"
    pdfs.mkdir()
    (pdfs / "notice.pdf").write_bytes(make_pdf(["first page", "second page"]).read_bytes())
    arguments = [
        "extract",
        str(pdfs),
        "--lexicon",
        str(_lexicon(tmp_path)),
        "--out",
        str(tmp_path / "extracted"),
        "--cache",
        str(tmp_path / "cache"),
    ]

    assert main(arguments) == 0
    first = capsys.readouterr()
    records = read_records(tmp_path / "extracted" / "notice.jsonl")
    assert [(record["page"], record["method"]) for record in records] == [(1, "ocr"), (2, "ocr")]
    assert StandInDots.readings == 2
    assert "notice page 1: read in 150 s" in first.err
    assert "notice.pdf" in first.out

    assert main(arguments) == 0
    assert StandInDots.readings == 2
    assert "notice page 2: from the cache" in capsys.readouterr().err


def test_extract_reports_a_page_read_again_for_its_text(
    make_pdf: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(dots, "DotsEngine", StandInDotsWithAPicture)
    quality = PageQuality(
        content_chars=500,
        presform_ratio=0.0,
        bidi_per_1k=0.0,
        c0_per_1k=0.0,
        replacement_chars=0,
        cid_placeholders=0,
        arabic_tokens=25,
        token_validity=0.5,
        single_letter_share=0.0,
    )
    verdicts = [PageVerdict(page=1, quality=quality, decision=decide(quality))]
    monkeypatch.setattr(records_module, "judge_pages", lambda texts, lexicon: verdicts)
    pdfs = tmp_path / "raw"
    pdfs.mkdir()
    (pdfs / "notice.pdf").write_bytes(make_pdf(["first page"]).read_bytes())
    arguments = ["extract", str(pdfs), "--lexicon", str(_lexicon(tmp_path))]
    arguments += ["--out", str(tmp_path / "extracted"), "--cache", str(tmp_path / "cache")]

    assert main(arguments) == 0
    assert (
        "notice page 1: read in 21 s; its text held 4 of the 25 Arabic words in its text "
        "layer, so it was read again for its text in 14 s: 1 paragraph added"
    ) in capsys.readouterr().err
    assert main(arguments) == 0
    assert "page 1: from the cache; " in capsys.readouterr().err


@pytest.mark.parametrize("command", ["inventory", "gate", "extract", "calendar"])
def test_no_command_reads_a_text_layer_with_a_pdfium_that_reorders_arabic(
    command: str,
    make_pdf: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # As PDFium 153.0.7999.0, in pypdfium2 5.13.0, reads the generated line.
    monkeypatch.setattr(extract, "_read_probe", lambda: "حذف فترة نهاية")
    extract.check_reading_order.cache_clear()
    pdf = make_pdf(["first page"])
    target = pdf if command == "calendar" else tmp_path
    arguments = [command, str(target)]
    if command != "inventory":
        arguments += ["--lexicon", str(_lexicon(tmp_path))]
    try:
        assert main(arguments) == 2
    finally:
        extract.check_reading_order.cache_clear()
    assert "out of order" in capsys.readouterr().err


def test_calendar_command_is_registered() -> None:
    assert build_parser().parse_args(["calendar", "calendar.pdf"]).command == "calendar"


def test_calendar_names_a_pdf_it_cannot_find(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["calendar", str(tmp_path / "absent.pdf")]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_calendar_wants_a_file_not_a_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["calendar", str(tmp_path)]) == 2
    assert "is not a file" in capsys.readouterr().err


def test_calendar_explains_a_missing_lexicon(
    make_pdf: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf = make_pdf(["first page"])
    assert main(["calendar", str(pdf), "--lexicon", str(tmp_path / "absent.zip")]) == 2
    assert "fetch_lexicon.py" in capsys.readouterr().err


def test_calendar_says_when_a_pdf_has_no_event_cards(
    make_pdf: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf = make_pdf(["Start of First Semester"])
    out = tmp_path / "calendar"
    arguments = ["calendar", str(pdf), "--lexicon", str(_lexicon(tmp_path)), "--out", str(out)]
    assert main(arguments) == 1
    assert "no event cards" in capsys.readouterr().err
    assert not out.exists()


def test_calendar_reports_a_page_it_cannot_read(
    make_pdf: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unreadable(path: Path, lexicon: frozenset[str]) -> list[CalendarPage]:
        raise ValueError("page 2: its text has 4 characters for PDFium's 3")

    monkeypatch.setattr("daleel.cli.read_calendar", unreadable)
    pdf = make_pdf(["a calendar"])
    assert main(["calendar", str(pdf), "--lexicon", str(_lexicon(tmp_path))]) == 1
    assert "page 2: its text has 4 characters" in capsys.readouterr().err


def _calendar_page() -> CalendarPage:
    """Two cards of the calendar's page 1, as read_calendar gives them, the
    second with a line that fit none of its columns."""
    meem, heh = chr(0x0645), chr(0x0647)
    start = CalendarRow(
        "بداية الفصل الدراسي الأول",
        "Start of First Semester",
        "الأحد",
        "Sun",
        "2026/08/23" + meem,
        "1448/03/10" + heh,
    )
    exams = CalendarRow(
        "الاختبارات النهائية",
        "Final Exams",
        "الأحد- الخميس",
        "Sun-Thu",
        "2026/12/31-20" + meem,
        "1448/07/22-11" + heh,
        unplaced=("12:00 PM",),
    )
    return CalendarPage(1, "الفصل الدراسي الأول (481)", "First Semester (481)", (start, exams))


def test_calendar_writes_each_page_s_rows_and_warns_of_a_line_it_could_not_place(
    make_pdf: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("daleel.cli.read_calendar", lambda path, lexicon: [_calendar_page()])
    pdf = make_pdf(["a calendar"], name="academic_weeks_1448.pdf")
    out = tmp_path / "calendar"
    arguments = ["calendar", str(pdf), "--lexicon", str(_lexicon(tmp_path)), "--out", str(out)]

    assert main(arguments) == 0
    written = out / "academic_weeks_1448_p01.csv"
    with written.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["title_en"] for row in rows] == ["Start of First Semester", "Final Exams"]
    assert rows[1]["date_gregorian"] == "2026/12/31-20" + chr(0x0645)
    captured = capsys.readouterr()
    assert "First Semester (481)" in captured.out
    assert "page 1, Final Exams: a line that fits no column: '12:00 PM'" in captured.err

    assert main([*arguments, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == [
        {
            "page": 1,
            "semester": "First Semester (481)",
            "rows": 2,
            "unplaced": 1,
            "csv": str(written),
        }
    ]


def _write_records(folder: Path, doc_id: str, texts: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{doc_id}.jsonl").open("w", encoding="utf-8") as file:
        for page, text in enumerate(texts, start=1):
            gate = {"verdict": "trusted", "token_validity": 1.0}
            record = {"doc_id": doc_id, "page": page, "method": "text_layer", "text": text}
            file.write(json.dumps({**record, "gate": gate}, ensure_ascii=False) + "\n")


def test_chunk_command_is_registered() -> None:
    assert build_parser().parse_args(["chunk"]).command == "chunk"


def test_chunk_names_a_folder_it_cannot_find(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["chunk", "--records", str(tmp_path / "absent")]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_chunk_says_how_to_make_records_it_cannot_find(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["chunk", "--records", str(tmp_path)]) == 1
    assert "daleel extract" in capsys.readouterr().err


def test_chunk_refuses_a_document_outside_the_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_records(tmp_path / "records", "unknown", ["نص من صفحة"])
    out = tmp_path / "chunks.jsonl"
    assert main(["chunk", "--records", str(tmp_path / "records"), "--out", str(out)]) == 1
    assert "not a document of the corpus" in capsys.readouterr().err
    assert not out.exists()


def test_chunk_writes_every_document_s_chunks_and_counts_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clauses = "\n".join(["المادة الأولى", ".1 يلتزم الطالب بالحضور", ".2 يلتزم الطالب بالأنظمة"])
    _write_records(tmp_path / "records", "student_conduct_code", [clauses, "نص الصفحة الثانية هنا"])
    _write_records(tmp_path / "records", "student_charter", ["حقوق الطالب وواجباته"])
    out = tmp_path / "processed" / "chunks.jsonl"
    arguments = ["chunk", "--records", str(tmp_path / "records"), "--out", str(out)]

    assert main(arguments) == 0
    chunks = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "student_charter_p1_c1",
        "student_conduct_code_p1_c1",
        "student_conduct_code_p2_c1",
    ]
    assert chunks[1]["section_heading"] == "المادة الأولى"
    assert "3 chunks written to" in capsys.readouterr().out

    assert main([*arguments, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[1] == {
        "document": "student_conduct_code",
        "pages": 2,
        "chunks": 2,
        "clauses": 2,
        "tables": 0,
        "median_words": 6,
        "max_words": 8,
    }


def test_chunk_wants_the_tables_typed_by_hand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_records(tmp_path / "records", "student_charter", ["حقوق الطالب وواجباته"])
    arguments = ["chunk", "--records", str(tmp_path / "records"), "--out", str(tmp_path / "c")]
    assert main([*arguments, "--manual", str(tmp_path / "absent.json")]) == 2
    assert "no tables typed by hand" in capsys.readouterr().err


def test_chunk_reads_the_calendar_from_its_text_without_its_pdf(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_records(tmp_path / "records", "academic_weeks_1448", ["بداية الفصل الدراسي الأول"])
    out = tmp_path / "chunks.jsonl"
    arguments = ["chunk", "--records", str(tmp_path / "records"), "--out", str(out)]
    assert main([*arguments, "--raw", str(tmp_path / "raw")]) == 0
    assert "the calendar is chunked from its text" in capsys.readouterr().err
    assert "بداية الفصل الدراسي الأول" in out.read_text(encoding="utf-8")


def test_chunk_needs_the_lexicon_to_read_the_calendar_s_cards(
    make_pdf: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf = make_pdf(["a calendar"], name="academic_weeks_1448.pdf")
    _write_records(tmp_path / "records", "academic_weeks_1448", ["بداية الفصل الدراسي الأول"])
    arguments = [
        "chunk",
        "--records",
        str(tmp_path / "records"),
        "--raw",
        str(pdf.parent),
        "--lexicon",
        str(tmp_path / "absent.zip"),
    ]
    assert main(arguments) == 2
    assert "fetch_lexicon.py" in capsys.readouterr().err


def test_chunk_makes_a_chunk_of_each_calendar_card(
    make_pdf: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("daleel.cli.read_calendar", lambda path, lexicon: [_calendar_page()])
    pdf = make_pdf(["a calendar"], name="academic_weeks_1448.pdf")
    _write_records(tmp_path / "records", "academic_weeks_1448", ["نص الصفحة الأولى من التقويم"])
    out = tmp_path / "chunks.jsonl"
    arguments = [
        "chunk",
        "--records",
        str(tmp_path / "records"),
        "--out",
        str(out),
        "--raw",
        str(pdf.parent),
        "--lexicon",
        str(_lexicon(tmp_path)),
    ]
    assert main(arguments) == 0
    chunks = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [chunk["section_heading"] for chunk in chunks] == ["الفصل الدراسي الأول (481)"] * 2
    assert "Final Exams" in chunks[1]["text"]
    assert "2 chunks written to" in capsys.readouterr().out
