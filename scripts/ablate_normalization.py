#!/usr/bin/env python3
"""Table T3: what the extractor, and each step of the analyzer, buys BM25.

    python3 scripts/ablate_normalization.py [--split dev] [--gold PATH] [options]

Six rows, each BM25 over the gold set's answerable questions:

1. pdftotext, terms as written: poppler's text layer for every page, chunked
   as the pipeline chunks, and matched word for word. This is what a naive
   pipeline indexes.
2. pdftotext, with the full analyzer: the same chunks, normalized, without
   stopwords, stemmed. Row 2 against row 6 is what the extraction buys.
3. The pipeline's chunks (data/processed/chunks.jsonl), terms as written.
4. Normalized for comparison.
5. And without stopwords.
6. And stemmed: the analyzer the pipeline uses.

Rows 3 to 6 add one step each, so the difference between neighbours is what
that step buys. The held column is the share of the evidence some chunk of
its page holds at all: text that extraction garbled holds no quote, so no
ranking can find it. pdftotext reads every page from its text layer, where
the pipeline reads 80 pages by OCR and rebuilds the calendar's cards and the
fee table, so row 1 also shows what that work recovers.

Needs pdftotext (poppler-utils) and the PDFs in data/raw/, and prints a
Markdown table with the poppler version under it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from daleel.chunk.chunker import CHUNKS, document_chunks, read_chunks
from daleel.eval.gold import CORPUS, GOLD_DRAFT, SPLITS, load_gold
from daleel.eval.retrieval import evaluate, summary
from daleel.retrieve.analyzer import Analyzer
from daleel.retrieve.bm25 import ChunkIndex

RAW = Path("data/raw")
COLUMNS = ("held", "recall@5", "recall@10", "complete@5", "mrr@10")
AS_WRITTEN = Analyzer(normalize=False, stopwords=False, stem=False)
ROWS = (
    ("pdftotext, terms as written", "pdftotext", AS_WRITTEN),
    ("pdftotext, full analyzer", "pdftotext", Analyzer()),
    ("pipeline, terms as written", "pipeline", AS_WRITTEN),
    ("+ normalization", "pipeline", Analyzer(stopwords=False, stem=False)),
    ("+ stopwords", "pipeline", Analyzer(stem=False)),
    ("+ light stemming", "pipeline", Analyzer()),
)


def pdftotext_pages(path: Path) -> list[str]:
    """Every page of a PDF as pdftotext reads it, split at its form feeds."""
    result = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", str(path), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    pages = result.stdout.split("\f")
    # pdftotext ends every page with a form feed, the last one included.
    return pages[:-1] if pages and not pages[-1].strip() else pages


def pdftotext_chunks(raw: Path) -> list[dict]:
    """The corpus read by pdftotext and cut by the pipeline's chunker."""
    chunks = []
    for doc_id, count in CORPUS.items():
        pages = pdftotext_pages(raw / f"{doc_id}.pdf")
        if len(pages) != count:
            raise ValueError(f"pdftotext read {len(pages)} pages of {doc_id}, not {count}")
        records = [
            {"doc_id": doc_id, "page": page, "method": "pdftotext", "text": text, "gate": {}}
            for page, text in enumerate(pages, start=1)
        ]
        chunks += document_chunks(records)
    return chunks


def poppler_version() -> str:
    banner = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True).stderr
    return banner.splitlines()[0] if banner else "pdftotext"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Table T3, the normalization ablation.")
    parser.add_argument("--gold", type=Path, default=GOLD_DRAFT)
    parser.add_argument("--split", choices=SPLITS, help="only the questions of this split")
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--json", action="store_true", help="emit the measures as JSON")
    args = parser.parse_args(argv)
    if shutil.which("pdftotext") is None:
        print("error: no pdftotext; install poppler-utils", file=sys.stderr)
        return 2
    for path in (args.gold, args.chunks, args.raw):
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2

    questions = [
        question
        for question in load_gold(args.gold)
        if args.split is None or question.get("split") == args.split
    ]
    corpora = {"pdftotext": pdftotext_chunks(args.raw), "pipeline": read_chunks(args.chunks)}
    rows = {}
    for name, corpus, analyzer in ROWS:
        chunks = corpora[corpus]
        index = ChunkIndex(chunks, analyzer)
        rows[name] = summary(evaluate(questions, chunks, index.search))

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    asked = next(iter(rows.values()))["questions"]
    where = f"the {args.split} split" if args.split else "the whole set"
    print(f"| BM25, {asked:.0f} answerable questions of {where} | " + " | ".join(COLUMNS) + " |")
    print("|---" * (len(COLUMNS) + 1) + "|")
    for name, measures in rows.items():
        print(f"| {name} | " + " | ".join(f"{measures[c]:.3f}" for c in COLUMNS) + " |")
    print(
        f"\n{len(corpora['pdftotext'])} chunks from pdftotext ({poppler_version()}), "
        f"{len(corpora['pipeline'])} from the pipeline; gold set {args.gold}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
