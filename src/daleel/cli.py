# Command line entery for daleel

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from daleel.ingest.inventory import format_table, inventory_dir, to_json


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daleel",
        description="Cited Arabic answers over RCJY college regulation documents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_inventory_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        return _run_inventory(args)
    return 2  # unreachable: argparse rejects unknown commands


if __name__ == "__main__":
    raise SystemExit(main())
