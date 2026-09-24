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
from daleel.ingest.gate import PageVerdict, Verdict, gate_document, load_gate_lexicon
from daleel.ingest.inventory import format_table, inventory_dir, to_json
from daleel.ingest.lexicon import LEXICON_ZIP
from daleel.ingest.metadata import PdfMetadata, read_metadata
from daleel.ingest.router import route

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

    The gate says "text_layer" when more than half of the pages are trusted.
    """
    counts = Counter(verdict.decision.verdict for verdict in verdicts)
    trusted = counts[Verdict.TRUSTED]
    gate_path = "text_layer" if trusted * 2 > len(verdicts) else "ocr"
    return {
        "file": name,
        "route": route_path,
        "pages": len(verdicts),
        "trusted": trusted,
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


# --- entry point -------------------------------------------------------------

_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "inventory": _run_inventory,
    "route": _run_route,
    "gate": _run_gate,
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
