"""Command line entry point for daleel."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from daleel.ingest.inventory import format_table, inventory_dir, to_json
from daleel.ingest.metadata import PdfMetadata, read_metadata
from daleel.ingest.router import route


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
        f"error: no PDFs in {path} -- see data/README.md for how to obtain them",
        file=sys.stderr,
    )
    return 1


def _add_inventory_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "inventory",
        help="measure what each PDF's text layer actually contains",
        description=(
            "Extract the text layer of every PDF in a directory and report the "
            "character classes that distinguish sound text from silent corruption. "
            "Reports only -- pass/fail verdicts belong to the quality gate."
        ),
    )
    parser.add_argument("path", type=Path, help="directory containing PDFs")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument(
        "--out", type=Path, default=None, help="write output to a file as well as stdout"
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-page progress on stderr")


def _run_inventory(args: argparse.Namespace) -> int:
    if not args.path.exists():
        print(f"error: {args.path} does not exist", file=sys.stderr)
        return 2
    if not args.path.is_dir():
        print(f"error: {args.path} is not a directory", file=sys.stderr)
        return 2

    inventories = inventory_dir(args.path, progress=not args.quiet)
    if not inventories:
        print(
            f"error: no PDFs in {args.path} -- see data/README.md for how to obtain them",
            file=sys.stderr,
        )
        return 1

    rendered = to_json(inventories) if args.json else format_table(inventories)
    print(rendered)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)

    return 0


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
    headers = [label for _, label in _ROUTE_COLUMNS]
    cells = [
        [row[key] if row[key] is not None else "-" for key, _ in _ROUTE_COLUMNS] for row in rows
    ]
    widths = [
        max([len(header), *(len(line[i]) for line in cells)]) for i, header in enumerate(headers)
    ]
    lines = [
        "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)),
        "  ".join("-" * w for w in widths),
    ]
    lines += ["  ".join(c.ljust(w) for c, w in zip(line, widths, strict=True)) for line in cells]
    return "\n".join(lines)


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

    pdfs = sorted(p for p in args.path.glob("*.pdf") if p.is_file())
    if not pdfs:
        return _report_no_pdfs(args.path)

    rows = [route_row(pdf.name, read_metadata(pdf)) for pdf in pdfs]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(format_route_table(rows))
    return 0


_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "inventory": _run_inventory,
    "route": _run_route,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daleel",
        description="Cited Arabic answers over RCJY college regulation documents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_inventory_parser(subparsers)
    _add_route_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
