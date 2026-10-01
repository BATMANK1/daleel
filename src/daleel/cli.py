"""Command line entry point for daleel."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from daleel.ingest.extract import BACKENDS
from daleel.ingest.gate import (
    PageVerdict,
    Verdict,
    document_path,
    gate_document,
    load_gate_lexicon,
)
from daleel.ingest.inventory import format_table, inventory_dir, to_json
from daleel.ingest.lexicon import LEXICON_ZIP
from daleel.ingest.metadata import PdfMetadata, read_metadata
from daleel.ingest.records import RECORDS, PageRecord, extract_document, write_records
from daleel.ingest.router import ExtractionPath, route
from daleel.ocr import dots
from daleel.ocr.cache import CACHE, CachedReader

# --- shared ------------------------------------------------------------------


def _directory_error(path: Path) -> int | None:
    """Report and return an exit code if `path` is not a usable directory."""
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2
    if not path.is_dir():
        print(f"error: {path} is not a directory", file=sys.stderr)
        return 2
    return None


def _report_no_pdfs(path: Path) -> int:
    print(
        f"error: no PDFs in {path}. See data/README.md for how to obtain them",
        file=sys.stderr,
    )
    return 1


def _pdfs_in(path: Path) -> list[Path]:
    return sorted(p for p in path.glob("*.pdf") if p.is_file())


def _format_rows(rows: Sequence[dict], columns: Sequence[tuple[str, str]]) -> str:
    """Render rows as an aligned table, with "-" for missing values."""
    headers = [label for _, label in columns]
    cells = [["-" if row[key] is None else str(row[key]) for key, _ in columns] for row in rows]
    widths = [
        max([len(header), *(len(line[i]) for line in cells)]) for i, header in enumerate(headers)
    ]
    lines = [
        "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)),
        "  ".join("-" * w for w in widths),
    ]
    lines += ["  ".join(c.ljust(w) for c, w in zip(line, widths, strict=True)) for line in cells]
    return "\n".join(lines)


# --- inventory ---------------------------------------------------------------


def _add_inventory_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "inventory",
        help="measure what each PDF's text layer actually contains",
        description=(
            "Extract the text layer of every PDF in a directory and report the "
            "character classes that distinguish sound text from silent corruption. "
            "Reports only: pass/fail verdicts belong to the quality gate."
        ),
    )
    parser.add_argument("path", type=Path, help="directory containing PDFs")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument(
        "--out", type=Path, default=None, help="write output to a file as well as stdout"
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-page progress on stderr")
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=BACKENDS[0],
        help="text extractor (default: %(default)s); pdfplumber is kept for comparison",
    )


def _run_inventory(args: argparse.Namespace) -> int:
    error = _directory_error(args.path)
    if error is not None:
        return error

    inventories = inventory_dir(args.path, progress=not args.quiet, backend=args.backend)
    if not inventories:
        return _report_no_pdfs(args.path)

    rendered = to_json(inventories) if args.json else format_table(inventories)
    print(rendered)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)

    return 0


# --- route -------------------------------------------------------------------

_ROUTE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("file", "file"),
    ("producer", "producer"),
    ("path", "path"),
    ("expected_failure", "expected failure"),
    ("rule", "rule"),
    ("matched_on", "matched on"),
)


def route_row(name: str, meta: PdfMetadata) -> dict[str, str | None]:
    """One document's route as plain data, ready for a table or for JSON."""
    result = route(producer=meta.producer, creator=meta.creator)
    return {
        "file": name,
        "producer": meta.producer,
        "creator": meta.creator,
        "path": result.path.value,
        "expected_failure": result.expected_failure.value,
        "rule": result.rule,
        "matched_on": result.matched_on,
    }


def format_route_table(rows: Sequence[dict[str, str | None]]) -> str:
    """Render route rows as an aligned table, with "-" where nothing matched."""
    return _format_rows(rows, _ROUTE_COLUMNS)


def _add_route_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "route",
        help="predict each PDF's extraction path from its metadata",
        description=(
            "Read each PDF's producer and creator, and predict which extraction path "
            "to try first and which failure to expect. A prediction only: the quality "
            "gate still checks every document."
        ),
    )
    parser.add_argument("path", type=Path, help="directory containing PDFs")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")


def _run_route(args: argparse.Namespace) -> int:
    error = _directory_error(args.path)
    if error is not None:
        return error

    pdfs = _pdfs_in(args.path)
    if not pdfs:
        return _report_no_pdfs(args.path)

    rows = [route_row(pdf.name, read_metadata(pdf)) for pdf in pdfs]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(format_route_table(rows))
    return 0


# --- gate --------------------------------------------------------------------

_GATE_SUMMARY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("file", "file"),
    ("route", "route"),
    ("pages", "pages"),
    ("trusted", "trusted"),
    ("untrusted", "untrusted"),
    ("no_arabic_text", "no arabic text"),
    ("gate_path", "gate says"),
    ("agrees", "agrees"),
)

_GATE_PAGE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("file", "file"),
    ("page", "page"),
    ("verdict", "verdict"),
    ("validity", "validity"),
    ("arabic_tokens", "arabic words"),
    ("rejected_by", "rejected by"),
)


def gate_summary(name: str, route_path: str, verdicts: Sequence[PageVerdict]) -> dict:
    """One document's verdicts counted, and whether they agree with the router.

    Pages with no Arabic in their text layer are counted in their own column,
    and go to OCR regardless of the path the document takes.
    """
    counts = Counter(verdict.decision.verdict for verdict in verdicts)
    gate_path = document_path(verdicts).value
    return {
        "file": name,
        "route": route_path,
        "pages": len(verdicts),
        "trusted": counts[Verdict.TRUSTED],
        "untrusted": counts[Verdict.UNTRUSTED],
        "no_arabic_text": counts[Verdict.NO_ARABIC_TEXT],
        "gate_path": gate_path,
        "agrees": "yes" if gate_path == route_path else "no",
    }


def gate_page_row(name: str, verdict: PageVerdict) -> dict:
    """One page's verdict as plain data for the per-page table."""
    validity = verdict.quality.token_validity
    return {
        "file": name,
        "page": verdict.page,
        "verdict": verdict.decision.verdict.value,
        "validity": None if validity is None else f"{validity:.0%}",
        "arabic_tokens": verdict.quality.arabic_tokens,
        "rejected_by": ", ".join(verdict.decision.rejected_by) or None,
    }


def _add_gate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "gate",
        help="judge whether each page's text layer can be trusted",
        description=(
            "Extract every page with pypdfium2, measure it, and judge whether its text "
            "layer can be trusted or the page needs OCR. The threshold was calibrated on "
            "five hand-transcribed pages; see the README."
        ),
    )
    parser.add_argument("path", type=Path, help="directory containing PDFs")
    parser.add_argument("--pages", action="store_true", help="one row per page, not per document")
    parser.add_argument("--json", action="store_true", help="every page's measurements as JSON")
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=LEXICON_ZIP,
        help="frequency-list zip (default: %(default)s)",
    )


def _run_gate(args: argparse.Namespace) -> int:
    error = _directory_error(args.path)
    if error is not None:
        return error

    pdfs = _pdfs_in(args.path)
    if not pdfs:
        return _report_no_pdfs(args.path)

    if not args.lexicon.is_file():
        print(
            f"error: no lexicon at {args.lexicon}. Fetch it with: python3 scripts/fetch_lexicon.py",
            file=sys.stderr,
        )
        return 2

    lexicon = load_gate_lexicon(args.lexicon)
    summaries, page_rows, records = [], [], []
    for pdf in pdfs:
        meta = read_metadata(pdf)
        route_path = route(producer=meta.producer, creator=meta.creator).path.value
        verdicts = gate_document(pdf, lexicon)
        summaries.append(gate_summary(pdf.name, route_path, verdicts))
        for verdict in verdicts:
            page_rows.append(gate_page_row(pdf.name, verdict))
            records.append({"file": pdf.name, **verdict.to_dict()})

    if args.json:
        print(json.dumps(records, indent=2, ensure_ascii=False))
    elif args.pages:
        print(_format_rows(page_rows, _GATE_PAGE_COLUMNS))
    else:
        print(_format_rows(summaries, _GATE_SUMMARY_COLUMNS))
    return 0


# --- extract -----------------------------------------------------------------

_EXTRACT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("file", "file"),
    ("pages", "pages"),
    ("document", "document"),
    ("text_layer", "text layer"),
    ("ocr", "OCR"),
    ("records", "records"),
)


def extract_summary(name: str, records: Sequence[PageRecord], out: Path | None) -> dict:
    """One document's pages counted by the method that extracted them, and where they went."""
    methods = Counter(record.method for record in records)
    return {
        "file": name,
        "pages": len(records),
        "document": records[0].document.value if records else None,
        "text_layer": methods[ExtractionPath.TEXT_LAYER],
        "ocr": methods[ExtractionPath.OCR],
        "records": None if out is None else str(out),
    }


def _add_extract_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "extract",
        help="extract every page's text, from its text layer or by OCR",
        description=(
            "Judge every page with the quality gate and extract its text: from the text "
            "layer where the gate trusts the page and its document, otherwise by OCR with "
            "dots.mocr on a running vLLM server. Each document's pages are written to "
            "OUT/<document>.jsonl, one record per line. OCR readings are cached, so a "
            "second run reads no page again and an interrupted run resumes where it stopped."
        ),
    )
    parser.add_argument("path", type=Path, help="directory containing PDFs")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="only these documents, named as their files are without .pdf",
    )
    parser.add_argument(
        "--out", type=Path, default=RECORDS, help="folder for the records (default: %(default)s)"
    )
    parser.add_argument(
        "--cache", type=Path, default=CACHE, help="folder for OCR readings (default: %(default)s)"
    )
    parser.add_argument(
        "--server",
        default=dots.DEFAULT.server,
        help="the vLLM server serving dots.mocr (default: %(default)s)",
    )
    parser.add_argument(
        "--quantization",
        default=dots.DEFAULT.quantization,
        help="as passed to vllm serve, such as fp8 (default: %(default)s)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="scale larger pages down to this many pixels, as the server allows",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="resolution to render pages at for OCR (default: %(default)s, as dots.mocr's)",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=LEXICON_ZIP,
        help="frequency-list zip (default: %(default)s)",
    )
    parser.add_argument("--json", action="store_true", help="emit the summary as JSON")


def _run_extract(args: argparse.Namespace) -> int:
    error = _directory_error(args.path)
    if error is not None:
        return error

    pdfs = _pdfs_in(args.path)
    if not pdfs:
        return _report_no_pdfs(args.path)

    if args.only:
        by_name = {pdf.stem: pdf for pdf in pdfs}
        unknown = [name for name in args.only if name not in by_name]
        if unknown:
            print(
                f"error: no {', '.join(unknown)} in {args.path}; "
                f"its documents are {', '.join(by_name)}",
                file=sys.stderr,
            )
            return 2
        pdfs = [by_name[name] for name in args.only]

    if not args.lexicon.is_file():
        print(
            f"error: no lexicon at {args.lexicon}. Fetch it with: python3 scripts/fetch_lexicon.py",
            file=sys.stderr,
        )
        return 2
    if args.dpi < 1:
        print(f"error: a resolution must be positive, not {args.dpi}", file=sys.stderr)
        return 2
    try:
        settings = dots.Settings(
            server=args.server, quantization=args.quantization, max_pixels=args.max_pixels
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    lexicon = load_gate_lexicon(args.lexicon)
    reader: CachedReader | None = None

    # The server is only asked for when a page needs OCR, so a document the
    # text layer serves throughout is extracted without one.
    def ocr() -> CachedReader:
        nonlocal reader
        if reader is None:
            reader = CachedReader(dots.DotsEngine(settings), args.cache, dpi=args.dpi)
            print(f"OCR: {reader.describe()}; pages at {args.dpi} DPI", file=sys.stderr)
        return reader

    def report(record: PageRecord) -> None:
        if record.ocr is None:
            return
        reading = record.ocr
        how = "from the cache" if reading.cached else f"read in {reading.seconds:.0f} s"
        again = reading.text_reading
        if again is not None:
            when = "from the cache" if again.cached else f"in {again.seconds:.0f} s"
            added = f"{again.added} paragraph{'' if again.added == 1 else 's'} added"
            how += (
                f"; its text held {reading.arabic_tokens} of the "
                f"{record.gate.quality.arabic_tokens} Arabic words in its text layer, "
                f"so it was read again for its text {when}: {added}"
            )
        print(f"{record.doc_id} page {record.page}: {how}", file=sys.stderr, flush=True)

    summaries = []
    for pdf in pdfs:
        try:
            records = extract_document(pdf, lexicon, ocr, on_record=report)
        except dots.DotsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        out = write_records(records, args.out) if records else None
        summaries.append(extract_summary(pdf.name, records, out))

    if args.json:
        print(json.dumps(summaries, indent=2, ensure_ascii=False))
    else:
        print(_format_rows(summaries, _EXTRACT_COLUMNS))
    return 0


# --- entry point -------------------------------------------------------------

_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "inventory": _run_inventory,
    "route": _run_route,
    "gate": _run_gate,
    "extract": _run_extract,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daleel",
        description="Cited Arabic answers over RCJY college regulation documents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_inventory_parser(subparsers)
    _add_route_parser(subparsers)
    _add_gate_parser(subparsers)
    _add_extract_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        status = _COMMANDS[args.command](args)
        # Flush inside the try, so a reader that has already gone away is
        # noticed here rather than as a traceback when the interpreter exits.
        sys.stdout.flush()
    except BrokenPipeError:
        # The reader stopped early, as `head` does. That is not a failure of
        # this program, so point the rest of the output at /dev/null and stop,
        # as the Python documentation recommends, with the conventional status 1.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
