#!/usr/bin/env python3
"""Run an OCR engine on every page that has ground truth, and score what it reads.

    python3 scripts/ocr_eval.py tesseract [--lang ara] [--psm 3] [--dpi 300]
    python3 scripts/ocr_eval.py paddle [--device cpu] [--min-score 0] [--stretch 1] [--dpi 300]
    python3 scripts/ocr_eval.py dots [--server http://localhost:8000] [--quantization none]
                                     [--max-pixels N] [--dpi 200]
    python3 scripts/ocr_eval.py saved data/interim/ocr/<run>

Each run gives one engine's row of T2. Every page is rendered once with
daleel.ocr.render and read once, and every piece of its ground truth in
data/interim/ground_truth/ is scored against that output:

- page    <doc>_pNN.txt: the whole page, so reading order counts
- region  <doc>_pNN_rK.txt: part of a page, scored against the stretch of
          output that matches it best
- table   any .csv, less its column names: no single reading order, so only
          the words missed count
- partial the .txt of a page whose table is in a .csv: its blocks sit around
          the table, so, like the table, only the words missed count

Where the ground truth covers a whole page, the words the engine added count
too: words it wrote that the page never printed, which is how an invented
table cell shows. Missed words are counted on every piece, added words on
whole pages and on the tables of whole pages.

CER raw, CER normalized and WER normalized are summed over pages and regions
before dividing, so each is weighted by its length. Before the timed run the
engine reads the first page once, untimed, so a one-off start-up cost such as
loading models is not charged to a page. What the engine read is saved under
data/interim/ocr/ for inspection, with a served model's whole response beside
it; like the ground truth, it stays out of git.

dots.mocr needs a vLLM server already serving it on this machine, and pages
for it are rendered at 200 DPI by default, as its own parser renders them.

saved scores what an earlier run saved, without reading the pages again, for
when the scoring changes but the engines have not. It cannot time anything.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from daleel.eval.ocr_metrics import (
    Errors,
    added_words,
    char_errors,
    missed_words,
    normalized_form,
    raw_form,
    word_errors,
)
from daleel.ocr import dots, tesseract
from daleel.ocr.engine import Engine
from daleel.ocr.render import render_page

RAW = Path("data/raw")
GROUND_TRUTH = Path("data/interim/ground_truth")
OUTPUT = Path("data/interim/ocr")
PIECE = re.compile(r"(?P<doc>.+)_p(?P<page>\d{2})(?:_r(?P<region>\d+))?\.(?P<ext>txt|csv)")
NONE = Errors(edits=0, length=0)

Page = tuple[str, int]


@dataclass(frozen=True)
class Piece:
    path: Path
    doc: str
    page: int
    kind: str
    # Part of its page only, so words the engine read elsewhere on the page are
    # not added words.
    region: bool = False

    @property
    def name(self) -> str:
        return self.path.stem

    def reference(self) -> str:
        """The text the page prints: for a table, every row but the column names."""
        if self.path.suffix == ".csv":
            with self.path.open(encoding="utf-8", newline="") as f:
                # The first row names the columns (ANNOTATION.md 6.2) and was
                # never printed, so no engine could read it.
                return "\n".join(" ".join(row) for row in list(csv.reader(f))[1:])
        return self.path.read_text(encoding="utf-8")


def find_pieces(folder: Path) -> list[Piece]:
    found = [(path, PIECE.fullmatch(path.name)) for path in sorted(folder.iterdir())]
    matched = [(path, m) for path, m in found if m is not None]
    tables = {(m["doc"], m["page"]) for _, m in matched if m["ext"] == "csv" and not m["region"]}
    pieces = []
    for path, m in matched:
        if m["ext"] == "csv":
            kind = "table"
        elif m["region"]:
            kind = "region"
        elif (m["doc"], m["page"]) in tables:
            kind = "partial"
        else:
            kind = "page"
        region = m["region"] is not None
        pieces.append(Piece(path=path, doc=m["doc"], page=int(m["page"]), kind=kind, region=region))
    return pieces


def whole_pages(pieces: list[Piece]) -> dict[Page, str]:
    """The full text of every page whose ground truth covers all of it."""
    pages: dict[Page, list[Piece]] = {}
    for piece in pieces:
        pages.setdefault((piece.doc, piece.page), []).append(piece)
    return {
        page: "\n".join(piece.reference() for piece in found)
        for page, found in pages.items()
        if not any(piece.region for piece in found)
    }


def cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def pct(errors: Errors) -> str:
    return f"{errors.rate:6.1%}" if errors.length else "     -"


def add_dpi(parser: argparse.ArgumentParser, default: int) -> None:
    # Added to each engine's parser, since a default set on an argument shared
    # through parents=[...] would change it for every engine.
    parser.add_argument(
        "--dpi", type=int, default=default, help=f"render resolution (default {default})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    engines = parser.add_subparsers(dest="engine", required=True)
    tess = engines.add_parser("tesseract", help="Tesseract, through its CLI")
    tess.add_argument("--lang", default=tesseract.DEFAULT.lang)
    tess.add_argument("--psm", type=int, default=tesseract.DEFAULT.psm)
    add_dpi(tess, 300)
    paddle = engines.add_parser("paddle", help="PaddleOCR (the paddle extra)")
    paddle.add_argument("--device", default="cpu", help="cpu, or gpu:0 with paddlepaddle-gpu")
    paddle.add_argument("--cpu-threads", type=int, default=10)
    paddle.add_argument("--mkldnn", action=argparse.BooleanOptionalAction, default=True)
    paddle.add_argument("--min-score", type=float, default=0.0, help="drop lines scored below")
    paddle.add_argument("--stretch", type=float, default=1.0, help="widen the page by this factor")
    add_dpi(paddle, 300)
    served = engines.add_parser("dots", help="dots.mocr, on a running vLLM server")
    served.add_argument("--server", default=dots.DEFAULT.server, help="the vLLM server's address")
    served.add_argument(
        "--quantization",
        default=dots.DEFAULT.quantization,
        help="as passed to vllm serve, such as fp8 (default none)",
    )
    served.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="scale larger pages down to this many pixels, as the server allows",
    )
    # dots.mocr's parser renders at 200 DPI; at 300 a page costs 2.25 times the image tokens.
    add_dpi(served, 200)
    saved = engines.add_parser("saved", help="score what an earlier run saved, reading nothing")
    saved.add_argument("folder", type=Path, help="the run's folder, under data/interim/ocr")
    return parser


def build_engine(args: argparse.Namespace) -> Engine:
    if args.engine == "tesseract":
        settings = tesseract.Settings(lang=args.lang, psm=args.psm, dpi=args.dpi)
        return tesseract.TesseractEngine(settings)
    if args.engine == "dots":
        settings = dots.Settings(
            server=args.server, quantization=args.quantization, max_pixels=args.max_pixels
        )
        return dots.DotsEngine(settings)
    from daleel.ocr import paddle

    settings = paddle.Settings(
        device=args.device,
        cpu_threads=args.cpu_threads,
        enable_mkldnn=args.mkldnn,
        min_score=args.min_score,
        stretch=args.stretch,
    )
    return paddle.PaddleEngine(settings)


def read_pages(
    engine: Engine, pngs: dict[Page, bytes], folder: Path
) -> tuple[dict[Page, str], dict[Page, float]]:
    """What the engine reads on each page, and how long each took, saved as it goes."""
    first = next(iter(pngs))
    engine.recognize(pngs[first])  # the untimed first read
    folder.mkdir(parents=True, exist_ok=True)
    outputs, seconds = {}, {}
    for (doc, page), png in pngs.items():
        result = engine.recognize(png)
        outputs[doc, page], seconds[doc, page] = result.text, result.seconds
        (folder / f"{doc}_p{page:02d}.txt").write_text(result.text, encoding="utf-8")
        if result.response:
            response = folder / f"{doc}_p{page:02d}.response.json"
            response.write_text(result.response, encoding="utf-8")
        if result.warnings:
            print(f"{doc} page {page}: {result.warnings}")
    return outputs, seconds


def saved_outputs(folder: Path, pages: list[Page]) -> dict[Page, str]:
    """What an earlier run saved for each page."""
    outputs = {}
    for doc, page in pages:
        path = folder / f"{doc}_p{page:02d}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")
        outputs[doc, page] = path.read_text(encoding="utf-8")
    return outputs


def report(
    pieces: list[Piece], outputs: dict[Page, str], seconds: dict[Page, float] | None
) -> None:
    """The scores of every piece of ground truth, and of all of them together."""
    whole = whole_pages(pieces)
    print(
        f"{'ground truth':42} {'kind':7} {'chars':>6} {'CER raw':>7} {'CER norm':>8} "
        f"{'WER norm':>8} {'missed':>6} {'added':>6} {'sec':>5}"
    )
    cer_raw = cer_norm = wer_norm = missed_all = added_all = NONE
    counted: set[Page] = set()
    for piece in pieces:
        page = (piece.doc, piece.page)
        reference, output = piece.reference(), outputs[page]
        missed = missed_words(reference, output)
        missed_all += missed
        row_raw = row_norm = row_wer = added = NONE
        if piece.kind in ("page", "region"):
            region = piece.kind == "region"
            row_raw = char_errors(reference, output, form=raw_form, region=region)
            row_norm = char_errors(reference, output, form=normalized_form, region=region)
            row_wer = word_errors(reference, output, form=normalized_form, region=region)
            cer_raw, cer_norm, wer_norm = cer_raw + row_raw, cer_norm + row_norm, wer_norm + row_wer
        # A whole page's added words are counted once: on the page, or on its table.
        if page in whole and piece.kind in ("page", "table") and page not in counted:
            counted.add(page)
            added = added_words(whole[page], output)
            added_all += added
        sec = f"{seconds[page]:5.1f}" if seconds else "    -"
        print(
            f"{piece.name:42} {piece.kind:7} {len(raw_form(reference)):>6} {pct(row_raw):>7} "
            f"{pct(row_norm):>8} {pct(row_wer):>8} {pct(missed):>6} {pct(added):>6} {sec}"
        )
    mean = f"{sum(seconds.values()) / len(seconds):5.1f}" if seconds else "    -"
    print(
        f"{'all pages and regions':42} {'':7} {cer_raw.length:>6} {pct(cer_raw):>7} "
        f"{pct(cer_norm):>8} {pct(wer_norm):>8} {pct(missed_all):>6} {pct(added_all):>6} {mean}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not GROUND_TRUTH.is_dir():
        print(f"error: no ground truth in {GROUND_TRUTH}; see eval/ANNOTATION.md", file=sys.stderr)
        return 2
    pieces = find_pieces(GROUND_TRUTH)
    pages = sorted({(piece.doc, piece.page) for piece in pieces})

    if args.engine == "saved":
        try:
            outputs = saved_outputs(args.folder, pages)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"output saved in {args.folder}, scored again without reading the pages")
        print()
        report(pieces, outputs, seconds=None)
        print()
        print(f"{len(pieces)} pieces of ground truth on {len(pages)} pages")
        return 0

    missing = sorted({doc for doc, _ in pages if not (RAW / f"{doc}.pdf").is_file()})
    if missing:
        print(f"error: missing from {RAW}: {', '.join(missing)}", file=sys.stderr)
        return 2

    try:
        engine = build_engine(args)
    except (
        tesseract.TesseractError,
        dots.DotsError,
        ImportError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    tag = f"{engine.tag()}-{args.dpi}dpi"
    print(f"{engine.describe()}; pages rendered at {args.dpi} DPI")
    print(
        f"pypdfium2 {version('pypdfium2')}, Python {platform.python_version()}, {cpu()}, "
        f"{os.cpu_count()} threads, OMP_THREAD_LIMIT {os.environ.get('OMP_THREAD_LIMIT', 'unset')}"
    )
    print()

    pngs = {
        (doc, page): render_page(RAW / f"{doc}.pdf", page, dpi=args.dpi).png()
        for doc, page in pages
    }
    try:
        outputs, seconds = read_pages(engine, pngs, OUTPUT / tag)
    except (tesseract.TesseractError, dots.DotsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report(pieces, outputs, seconds)
    print()
    print(f"{len(pieces)} pieces of ground truth on {len(pages)} pages; output in {OUTPUT / tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
